"""CPU-only checks for the OpenAI server: tokenizer parity with llama.cpp, template rendering, prefix stability
across agent turns (what the prefix cache relies on) and tool-call parsing."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from adakernel.serve import DEFAULT_MODEL, ChatFormat, OutputParser, parse_tool_calls

MODEL = Path(os.environ.get("AK_QWEN35_GGUF", str(DEFAULT_MODEL)))
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="qwen35 GGUF not found")

TOOLS = [
    {"type": "function", "function": {
        "name": "read", "description": "Read a file",
        "parameters": {"type": "object", "properties": {
            "filePath": {"type": "string"}, "offset": {"type": "number"}, "limit": {"type": "integer"}},
            "required": ["filePath"]}}},
    {"type": "function", "function": {
        "name": "edit", "description": "Edit a file",
        "parameters": {"type": "object", "properties": {
            "filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"},
            "replaceAll": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "todowrite", "description": "Write todos",
        "parameters": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object"}}}}}},
]


@pytest.fixture(scope="module")
def fmt():
    return ChatFormat(MODEL)


def _conv():
    sys = [{"role": "system", "content": "You are opencode, a coding agent."},
           {"role": "system", "content": "Working directory: C:/proj"}]
    u1 = {"role": "user", "content": [{"type": "text", "text": "Fix the bug in main.py"}]}
    a1 = {"role": "assistant", "content": "Let me look.", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"filePath": "main.py"})}}]}
    t1 = {"role": "tool", "tool_call_id": "c1", "content": "print('hi'\n"}
    a2 = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c2", "type": "function", "function": {"name": "edit", "arguments": json.dumps(
            {"filePath": "main.py", "oldString": "print('hi'", "newString": "print('hi')"})}}]}
    t2 = {"role": "tool", "tool_call_id": "c2", "content": "ok"}
    a3 = {"role": "assistant", "content": "Fixed the missing parenthesis."}
    u2 = {"role": "user", "content": "Thanks, now add a test."}
    return sys, [u1, a1, t1, a2, t2, a3, u2]


def test_render_has_tools_and_calls(fmt):
    sys, msgs = _conv()
    text = fmt.render(sys + msgs[:3], TOOLS)
    assert text.startswith("<|im_start|>system\n# Tools")
    assert '"name": "read"' in text
    assert "Working directory: C:/proj" in text
    assert "<tool_call>\n<function=read>\n<parameter=filePath>\nmain.py\n</parameter>" in text
    assert "<|im_start|>user\n<tool_response>\nprint('hi'\n</tool_response><|im_end|>" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _common(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def test_prefix_cache_points_survive_followups(fmt):
    """Every turn of an agent session must share, with the next turn's prompt, at least the snapshot points the
    server takes - otherwise the cache cannot help."""
    sys, msgs = _conv()
    turns = [sys + msgs[: n] for n in (1, 3, 5, 7)]
    ids = [fmt.tok.encode(fmt.render(t, TOOLS)) for t in turns]
    for a, b in zip(ids, ids[1:]):
        pts = fmt.snap_points(a)
        common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        reusable = [p for p in pts if p <= common]
        assert reusable, (pts, common)
        if len(b) > len(a) and b[: len(a)] == a:
            continue
    # within the tool loop (no new user message) the newest point is reusable
    p = fmt.snap_points(ids[1])
    common = _common(ids[1], ids[2])
    assert max(p) <= common
    # after the new user message, the reply-after-last-user point of the previous turn still matches
    p = fmt.snap_points(ids[2])
    common = _common(ids[2], ids[3])
    assert sorted(p)[-2] <= common


def test_tool_call_parsing():
    text = ("<tool_call>\n<function=edit>\n<parameter=filePath>\na.py\n</parameter>\n<parameter=oldString>\n"
            "x = 1\n\ny = 2\n</parameter>\n<parameter=replaceAll>\nfalse\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=read>\n<parameter=filePath>\nb.py\n</parameter>\n<parameter=limit>\n40\n"
            "</parameter>\n</function>\n</tool_call>")
    calls = parse_tool_calls(text, TOOLS)
    assert [c["function"]["name"] for c in calls] == ["edit", "read"]
    a0 = json.loads(calls[0]["function"]["arguments"])
    assert a0 == {"filePath": "a.py", "oldString": "x = 1\n\ny = 2", "replaceAll": False}
    assert json.loads(calls[1]["function"]["arguments"]) == {"filePath": "b.py", "limit": 40}
    todo = parse_tool_calls('<tool_call>\n<function=todowrite>\n<parameter=todos>\n[{"content": "a"}]\n'
                            "</parameter>\n</function>\n</tool_call>", TOOLS)
    assert json.loads(todo[0]["function"]["arguments"]) == {"todos": [{"content": "a"}]}


def test_output_parser_holds_partial_tag():
    p = OutputParser()
    out = "".join(p.feed(s) for s in ["I'll read it.\n\n<to", "ol_c", "all>\n<function=read>"])
    assert out == "I'll read it.\n\n"
    assert p.in_tools and p.tool_text.startswith("<tool_call>")
    q = OutputParser()
    assert q.feed("a < b <t") == "a < b "
    assert q.feed("x") == "<tx"


@pytest.mark.skipif(not os.environ.get("LLAMA_CPP_BIN"), reason="set LLAMA_CPP_BIN to a llama.cpp bin folder")
def test_tokenizer_matches_llama_cpp(fmt):
    sys, msgs = _conv()
    text = fmt.render(sys + msgs, TOOLS) + "naïve café 日本語 😀 x²  \r\n\t end"
    text += (Path(__file__).resolve().parents[1] / "adakernel" / "serve.py").read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False, newline="") as f:
        f.write(text)
    try:
        r = subprocess.run([str(Path(os.environ["LLAMA_CPP_BIN"]) / "llama-tokenize"), "-m", str(MODEL), "-f", f.name, "--ids",
                            "--log-disable", "--no-escape"], capture_output=True, text=True, encoding="utf-8",
                           stdin=subprocess.DEVNULL, check=True)
    finally:
        os.unlink(f.name)
    ref = json.loads([ln for ln in r.stdout.splitlines() if ln.strip().startswith("[")][-1])
    assert fmt.tok.encode(text) == ref
