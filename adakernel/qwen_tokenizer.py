"""In-process byte-level BPE tokenizer for Qwen GGUF files (pre-tokenizer types qwen2 / qwen35).

Reads tokens, token types and merges straight from the GGUF header and reproduces llama.cpp's tokenization with
special-token parsing on (control and user-defined tokens such as <|im_start|> and <tool_call> map to their ids).
Python's `re` has no \\p{..} classes, so the letter / mark / number classes are built once from unicodedata.
"""

from __future__ import annotations

import functools
import re
import struct
import unicodedata
from pathlib import Path

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
_NORMAL = 1


def read_tokenizer_arrays(model_path: str | Path) -> dict:
    """tokenizer.ggml.{tokens, token_type, merges} plus scalar tokenizer.* keys."""
    want = {"tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"}
    out: dict = {}
    with open(model_path, "rb") as f:
        magic, _version, _n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != 0x46554747:
            raise ValueError("not a GGUF file")

        def rstr() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", errors="replace")

        def val(t, keep):
            if t in _SCALAR:
                fmt = _SCALAR[t]
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            if t == 8:
                return rstr()
            if t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if et in _SCALAR:
                    fmt = _SCALAR[et]
                    raw = f.read(struct.calcsize(fmt) * n)
                    return list(struct.unpack(f"<{n}{fmt[1]}", raw)) if keep else None
                vals = [val(et, keep) for _ in range(n)]
                return vals if keep else None
            raise ValueError(f"unknown gguf value type {t}")

        for _ in range(n_kv):
            key = rstr()
            (t,) = struct.unpack("<I", f.read(4))
            v = val(t, key in want)
            if key in want or (key.startswith("tokenizer.") and v is not None):
                out[key] = v
    return out


@functools.lru_cache(maxsize=1)
def _classes() -> tuple[str, str, str]:
    """Regex character-class bodies for \\p{L}, \\p{M} and \\p{N}."""
    groups: dict[str, list[tuple[int, int]]] = {"L": [], "M": [], "N": []}
    for cp in range(0x110000):
        c = unicodedata.category(chr(cp))[0]
        g = groups.get(c)
        if g is None:
            continue
        if g and g[-1][1] == cp - 1:
            g[-1] = (g[-1][0], cp)
        else:
            g.append((cp, cp))

    def body(ranges):
        parts = []
        for a, b in ranges:
            ea, eb = re.escape(chr(a)), re.escape(chr(b))
            parts.append(ea if a == b else f"{ea}-{eb}")
        return "".join(parts)

    return body(groups["L"]), body(groups["M"]), body(groups["N"])


def _pretokenizer(kind: str) -> re.Pattern:
    L, M, N = _classes()
    contractions = r"'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]"
    if kind == "qwen35":
        pat = (f"{contractions}|[^\\r\\n{L}{N}]?[{L}{M}]+|[{N}]| ?[^\\s{L}{M}{N}]+[\\r\\n]*|\\s*[\\r\\n]+"
               r"|\s+(?!\S)|\s+")
    else:  # qwen2
        pat = f"{contractions}|[^\\r\\n{L}{N}]?[{L}]+|[{N}]| ?[^\\s{L}{N}]+[\\r\\n]*|\\s*[\\r\\n]+" r"|\s+(?!\S)|\s+"
    return re.compile(pat)


def _byte_encoder() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class QwenTokenizer:
    def __init__(self, model_path: str | Path):
        arr = read_tokenizer_arrays(model_path)
        self.tokens: list[str] = arr["tokenizer.ggml.tokens"]
        types = arr.get("tokenizer.ggml.token_type") or [_NORMAL] * len(self.tokens)
        merges = arr["tokenizer.ggml.merges"]
        self.pre = arr.get("tokenizer.ggml.pre", "qwen2")
        self.eos_id = arr.get("tokenizer.ggml.eos_token_id")
        self.chat_template = arr.get("tokenizer.chat_template")
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.ranks = {}
        for i, mg in enumerate(merges):
            a, _, b = mg.partition(" ")
            self.ranks[(a, b)] = i
        self.special = {t: i for i, t in enumerate(self.tokens) if types[i] != _NORMAL and t}
        alts = sorted(self.special, key=len, reverse=True)
        self._split = re.compile("(" + "|".join(re.escape(t) for t in alts) + ")") if alts else None
        self._pre = _pretokenizer(self.pre)
        self._be = _byte_encoder()
        self._bd = {v: k for k, v in self._be.items()}
        self._cache: dict[str, list[int]] = {}

    def id(self, text: str) -> int:
        return self.vocab[text]

    def _bpe(self, word: str) -> list[int]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        direct = self.vocab.get(word)
        if direct is not None:
            self._cache[word] = [direct]
            return [direct]
        parts = list(word)
        ranks = self.ranks
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            a, b = parts[bi], parts[bi + 1]
            merged, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        ids = []
        for p in parts:
            t = self.vocab.get(p)
            if t is None:  # unmergeable byte sequence: fall back to single byte symbols
                ids.extend(self.vocab[ch] for ch in p)
            else:
                ids.append(t)
        if len(self._cache) < 500_000:
            self._cache[word] = ids
        return ids

    def _encode_plain(self, text: str, out: list[int]) -> None:
        be = self._be
        for m in self._pre.finditer(text):
            word = "".join(be[b] for b in m.group().encode("utf-8"))
            out.extend(self._bpe(word))

    def encode(self, text: str, special: bool = True) -> list[int]:
        out: list[int] = []
        if not special or self._split is None:
            self._encode_plain(text, out)
            return out
        for i, piece in enumerate(self._split.split(text)):
            if not piece:
                continue
            if i % 2:
                out.append(self.special[piece])
            else:
                self._encode_plain(piece, out)
        return out

    def token_bytes(self, i: int) -> bytes:
        tok = self.tokens[i]
        if i in self._special_ids:
            return tok.encode("utf-8")
        bd = self._bd
        out = bytearray()
        for ch in tok:
            b = bd.get(ch)
            if b is None:
                out += ch.encode("utf-8")
            else:
                out.append(b)
        return bytes(out)

    @functools.cached_property
    def _special_ids(self) -> set[int]:
        return set(self.special.values())

    def decode(self, ids) -> str:
        return b"".join(self.token_bytes(int(i)) for i in ids).decode("utf-8", errors="replace")


class StreamDecoder:
    """Incremental detokenizer: feed ids, get back the newly completed text (holds incomplete UTF-8 sequences)."""

    def __init__(self, tok: QwenTokenizer):
        self.tok = tok
        self.buf = b""

    def feed(self, ids) -> str:
        self.buf += b"".join(self.tok.token_bytes(int(i)) for i in ids)
        cut = len(self.buf)
        # keep back up to 3 bytes of an unfinished multi-byte character
        for back in range(1, min(4, len(self.buf)) + 1):
            b = self.buf[-back]
            if b & 0xC0 == 0x80:
                continue
            need = 2 if b & 0xE0 == 0xC0 else 3 if b & 0xF0 == 0xE0 else 4 if b & 0xF8 == 0xF0 else 1
            if need > back:
                cut = len(self.buf) - back
            break
        text = self.buf[:cut].decode("utf-8", errors="replace")
        self.buf = self.buf[cut:]
        return text
