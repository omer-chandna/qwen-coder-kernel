"""OpenAI-compatible chat server for qwen35 GGUF files on the AdaKernel runtime (for OpenCode, Continue, Aider...).

    python -m adakernel.serve [--model PATH] [--port 8090] [--ctx 32768]

POST /v1/chat/completions (streaming or not, with tools), GET /v1/models. Decoding is greedy speculative decoding
(MTP drafts + prompt lookup); temperature / top_p are accepted and ignored. Prompts are rendered with the GGUF's own
chat template and tool calls are parsed from the model's <tool_call><function=..><parameter=..> output.

Agent loops resend the whole conversation every turn, so the server keeps recurrent-state snapshots at the end of
the system prompt and at the start of the newest message: a follow-up request only processes what changed since.
One request runs at a time.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment

from adakernel.qwen_tokenizer import QwenTokenizer, StreamDecoder

DEFAULT_MODEL = Path.home() / ".ollama/models/blobs/sha256-c7f6915f1d9ccc059903e3bead2493f8e4389a7718e8c1c4eb62ffaf33843408"
MODEL_ID = "qwen3.8-9b"


# ---------------------------------------------------------------------------------------------- prompt rendering


def _raise(msg):
    raise ValueError(msg)


def _tojson(x, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type", "text") == "text")


def normalize_messages(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages -> what the Qwen template expects: one leading system message, tool-call arguments as
    mappings, text-only content."""
    system: list[str] = []
    rest: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            t = _text(m.get("content")).strip()
            if t:
                if rest:  # a late system message becomes a user turn rather than breaking the template
                    rest.append({"role": "user", "content": t})
                else:
                    system.append(t)
            continue
        out = {"role": role, "content": _text(m.get("content"))}
        if role == "assistant":
            if m.get("reasoning_content"):
                out["reasoning_content"] = m["reasoning_content"]
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"input": args}
                if not isinstance(args, dict):
                    args = {"input": args}
                calls.append({"name": fn.get("name", ""), "arguments": args})
            if calls:
                out["tool_calls"] = calls
        rest.append(out)
    msgs = ([{"role": "system", "content": "\n\n".join(system)}] if system else []) + rest
    if not any(m["role"] == "user" for m in msgs):
        msgs.append({"role": "user", "content": "Continue."})
    return msgs


# ---------------------------------------------------------------------------------------------- output parsing

_TC_OPEN = "<tool_call>"
_FN_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.S)


def _convert(value: str, schema: dict | None):
    t = (schema or {}).get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t == "string":
        return value
    if value.lower() == "null" and t != "string":
        return None
    if t in ("integer", "number"):
        try:
            f = float(value)
            return int(f) if t == "integer" and f.is_integer() else f
        except ValueError:
            return value
    if t == "boolean":
        return value.strip().lower() in ("true", "1", "yes")
    if t in ("object", "array") or t is None:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            if t is None:
                return value
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
    return value


def parse_tool_calls(text: str, tools: list[dict]) -> list[dict]:
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    calls = []
    for block in text.split(_TC_OPEN)[1:]:
        block = block.split("</tool_call>")[0]
        m = _FN_RE.search(block)
        if not m:
            continue
        name = m.group(1).strip()
        props = schemas.get(name, {})
        args = {}
        for pm in _PARAM_RE.finditer(m.group(2)):
            key = pm.group(1).strip()
            val = pm.group(2)
            if val.startswith("\n"):
                val = val[1:]
            if val.endswith("\n"):
                val = val[:-1]
            args[key] = _convert(val, props.get(key))
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


class OutputParser:
    """Splits streamed model text into user-visible content and a trailing tool-call section."""

    def __init__(self):
        self.content = ""
        self.tool_text = ""
        self.in_tools = False
        self._held = ""

    def feed(self, text: str) -> str:
        if self.in_tools:
            self.tool_text += text
            return ""
        buf = self._held + text
        i = buf.find(_TC_OPEN)
        if i >= 0:
            self.in_tools = True
            self.tool_text = buf[i:]
            self._held = ""
            emit = buf[:i]
        else:
            # hold back a suffix that could be the start of "<tool_call>"
            keep = 0
            for k in range(min(len(_TC_OPEN) - 1, len(buf)), 0, -1):
                if _TC_OPEN.startswith(buf[-k:]):
                    keep = k
                    break
            emit, self._held = (buf[:-keep], buf[-keep:]) if keep else (buf, "")
        self.content += emit
        return emit

    def flush(self) -> str:
        out, self._held = self._held, ""
        self.content += out
        return out


# ---------------------------------------------------------------------------------------------- engine


class ChatFormat:
    """Tokenizer + the GGUF's chat template + stop tokens (no GPU)."""

    def __init__(self, model_path: Path):
        self.path = Path(model_path)
        self.tok = QwenTokenizer(self.path)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        self.template = env.from_string(self.tok.chat_template)
        self.im_start = self.tok.id("<|im_start|>")
        self.stop_ids = {self.tok.id("<|im_end|>")}
        for name in ("<|endoftext|>",):
            if name in self.tok.vocab:
                self.stop_ids.add(self.tok.vocab[name])
        if self.tok.eos_id is not None:
            self.stop_ids.add(int(self.tok.eos_id))

    def render(self, messages, tools) -> str:
        return self.template.render(messages=normalize_messages(messages), tools=tools or None,
                                    add_generation_prompt=True)

    def snap_points(self, ids: list[int]) -> list[int]:
        """Positions (each an <|im_start|>) whose prefix a follow-up request is likely to share:
        - the end of the system prompt (tools + instructions never change within a session),
        - the first reply after the last real user message: when the next user message arrives the template
          re-renders the replies after it without their (empty) think block, so later snapshots stop matching,
        - the start of the generation prompt, for the next step of a tool loop."""
        starts = [i for i, t in enumerate(ids) if t == self.im_start]
        pts = []
        if len(starts) >= 2:
            pts.append(starts[1])
        last_user = None
        for j, i in enumerate(starts[:-1]):
            head = self.tok.decode(ids[i + 1:i + 5])
            if head.startswith("user") and not head.startswith("user\n<tool_response>"):
                last_user = j
        if last_user is not None:
            pts.append(starts[last_user + 1])
        if starts:
            pts.append(starts[-1])
        return sorted(set(pts))


class Engine(ChatFormat):
    def __init__(self, model_path: Path, ctx: int, draft_vocab: int = 65536):
        from adakernel.qwen35 import Qwen35Model

        super().__init__(model_path)
        t0 = time.time()
        self.model = Qwen35Model(self.path, max_seq=ctx)
        self.ctx = ctx
        self.draft_vocab = draft_vocab
        self.lock = threading.Lock()
        print(f"[serve] loaded {self.path.name[:24]}... ctx {ctx} in {time.time() - t0:.1f}s, "
              f"VRAM {self.model.info['vram_mb']:.0f} MB", flush=True)

    def run(self, body: dict, on_text=None) -> dict:
        """Generate one chat completion. on_text(str) streams visible content; returns the final result."""
        tools = body.get("tools") or []
        if body.get("tool_choice") == "none":
            tools = []
        prompt = self.render(body.get("messages") or [], tools)
        ids = self.tok.encode(prompt)
        room = self.ctx - len(ids) - 16
        if room < 1:
            raise ContextError(f"prompt is {len(ids)} tokens; the server context is {self.ctx}")
        max_new = body.get("max_completion_tokens") or body.get("max_tokens") or room
        max_new = max(1, min(int(max_new), room))
        stops = body.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        dec = StreamDecoder(self.tok)
        parser = OutputParser()
        state = {"finish": "length", "n": 0}

        def on_tokens(toks):
            text_ids = []
            for t in toks:
                state["n"] += 1
                if t in self.stop_ids:
                    state["finish"] = "stop"
                    break
                text_ids.append(t)
            text = dec.feed(text_ids)
            if text:
                vis = parser.feed(text)
                if stops and not parser.in_tools:
                    hit = min((parser.content.find(s) for s in stops if s in parser.content), default=-1)
                    if hit >= 0:
                        cut = len(parser.content) - hit
                        vis = vis[: max(0, len(vis) - cut)]
                        parser.content = parser.content[:hit]
                        state["finish"] = "stop"
                if vis and on_text:
                    on_text(vis)
            return state["finish"] == "stop"

        with self.lock:
            t0 = time.time()
            r = self.model.generate_cached(ids, max_new, on_tokens, snap_at=self.snap_points(ids),
                                           draft_vocab=self.draft_vocab)
            wall = time.time() - t0
        tail = parser.flush() if not parser.in_tools else ""
        if tail and on_text:
            on_text(tail)
        calls = parse_tool_calls(parser.tool_text, tools) if parser.in_tools else []
        finish = "tool_calls" if calls else state["finish"]
        n_gen = len(r["tokens"])
        print(f"[serve] prompt {len(ids)} tok ({r['cached_tokens']} cached, {r['prompt_ms'] / 1000:.2f}s) | "
              f"gen {n_gen} tok {r['decode_tps']:.0f} tok/s | {len(calls)} tool calls | {wall:.2f}s", flush=True)
        return {"content": parser.content, "tool_calls": calls, "finish": finish,
                "usage": {"prompt_tokens": len(ids), "completion_tokens": n_gen, "total_tokens": len(ids) + n_gen,
                          "prompt_tokens_details": {"cached_tokens": r["cached_tokens"]}}}


class ContextError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------- HTTP


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    engine: Engine = None  # set in main

    def log_message(self, fmt, *args):  # quiet default access log
        pass

    def _json(self, code: int, obj) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def _sse(self, obj) -> None:
        self._chunk(b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode("utf-8")) + b"\n\n")

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "adakernel",
                 "context_length": self.engine.ctx}]})
        elif self.path in ("/", "/health"):
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except json.JSONDecodeError as e:
            self._json(400, {"error": {"message": f"bad JSON: {e}", "type": "invalid_request_error"}})
            return
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        model = body.get("model") or MODEL_ID
        try:
            if body.get("stream"):
                self._stream(body, cid, created, model)
            else:
                res = self.engine.run(body)
                msg = {"role": "assistant", "content": res["content"] or (None if res["tool_calls"] else "")}
                if res["tool_calls"]:
                    msg["tool_calls"] = res["tool_calls"]
                self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": model,
                                 "choices": [{"index": 0, "message": msg, "finish_reason": res["finish"]}],
                                 "usage": res["usage"]})
        except ContextError as e:
            self._json(400, {"error": {"message": str(e), "type": "invalid_request_error",
                                       "code": "context_length_exceeded"}})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            print("[serve] client disconnected; generation stopped", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[serve] error: {e!r}", flush=True)
            if not getattr(self, "_streaming", False):
                self._json(500, {"error": {"message": str(e), "type": "server_error"}})

    def _stream(self, body, cid, created, model):
        def chunk(delta, finish=None):
            return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        started = {"v": False}

        def start():
            if started["v"]:
                return
            started["v"] = True
            self._streaming = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._sse(chunk({"role": "assistant", "content": ""}))

        def on_text(t):
            start()
            self._sse(chunk({"content": t}))

        res = self.engine.run(body, on_text)  # ContextError propagates before any bytes are sent
        start()
        for i, tc in enumerate(res["tool_calls"]):
            self._sse(chunk({"tool_calls": [{"index": i, **tc}]}))
        self._sse(chunk({}, res["finish"]))
        if (body.get("stream_options") or {}).get("include_usage"):
            self._sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [], "usage": res["usage"]})
        self._sse(b"[DONE]")
        self._chunk(b"")


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # clients dropping keep-alive connections is normal
        import sys

        if not isinstance(sys.exc_info()[1], (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            super().handle_error(request, client_address)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090, help="8090: OpenCode auto-discovers port 8000 as vLLM")
    ap.add_argument("--ctx", type=int, default=32768, help="context length in tokens (KV cache size)")
    ap.add_argument("--draft-vocab", type=int, default=65536)
    ap.add_argument("--pf-cols", type=int, default=0,
                    help="prompt chunk length (default 1024); 512 saves ~170 MB of VRAM, prompts read ~15%% slower")
    args = ap.parse_args()
    if args.pf_cols:
        import os

        os.environ["AK_Q35_PF_COLS"] = str(args.pf_cols)  # read by the runtime on the first long prompt
    Handler.engine = Engine(Path(args.model), args.ctx, args.draft_vocab)
    srv = Server((args.host, args.port), Handler)
    print(f"[serve] OpenAI-compatible API on http://{args.host}:{args.port}/v1  (model id: {MODEL_ID})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
