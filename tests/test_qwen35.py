"""qwen35 hybrid runtime: graph vs eager, speculative == greedy, batched prefill, prefix cache, and (with
LLAMA_CPP_BIN set to a llama.cpp bin folder) llama.cpp greedy parity. Skipped unless the model file is present.

Set AK_QWEN35_GGUF to a qwen35 GGUF; defaults to the Ollama blob of eriktonon/qwen3.8-uncensored:9b.
Needs about 5 GB of free VRAM.
"""

import functools
import os
import subprocess
from pathlib import Path

import pytest

DEFAULT = Path.home() / ".ollama/models/blobs/sha256-c7f6915f1d9ccc059903e3bead2493f8e4389a7718e8c1c4eb62ffaf33843408"
MODEL = Path(os.environ.get("AK_QWEN35_GGUF", DEFAULT))

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="qwen35 GGUF not present")
LLAMA_BIN = os.environ.get("LLAMA_CPP_BIN")


@functools.lru_cache(maxsize=1)
def _tok():
    from adakernel.qwen_tokenizer import QwenTokenizer

    return QwenTokenizer(MODEL)


def tokenize(_path, text):
    return _tok().encode(text)


@pytest.fixture(scope="module")
def model():
    from adakernel.qwen35 import Qwen35Model

    m = Qwen35Model(MODEL, max_seq=2048)
    yield m
    m.close()


def test_graph_matches_eager(model):

    ids = tokenize(MODEL, "The capital of France is")
    a = model.generate(ids, 24, graph=False)["tokens"]
    b = model.generate(ids, 24, graph=True)["tokens"]
    assert (a == b).all()


@pytest.mark.parametrize("n_draft", [1, 3, 5, 11])
def test_mtp_speculative_equals_greedy(model, n_draft):

    if not model.info["has_mtp"]:
        pytest.skip("file has no MTP block")
    ids = tokenize(MODEL, "Write a short story about a robot who learns to paint.")
    ref = model.generate(ids, 48)["tokens"]
    spec = model.generate_mtp(ids, 48, n_draft=n_draft)
    assert (spec["tokens"] == ref).all()
    assert spec["acceptance"] > 0.3


def _code_prompt() -> str:
    src = (Path(__file__).resolve().parents[1] / "adakernel" / "qwen35.py").read_text(encoding="utf-8")
    return ("<|im_start|>user\nAdd a docstring to every method of this module and return the full file.\n\n```python\n"
            + src[:2500] + "```<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.parametrize("lookup", [False, True])
def test_spec_equals_greedy(model, lookup):

    if not model.info["has_mtp"]:
        pytest.skip("file has no MTP block")
    ids = tokenize(MODEL, _code_prompt())
    ref = model.generate(ids, 96)["tokens"]
    spec = model.generate_spec(ids, 96, lookup=lookup)
    assert (spec["tokens"] == ref).all()
    assert spec["tokens_per_round"] > 1.5
    if lookup:
        assert spec["lookup_rounds"] > 0


def test_batched_prefill_matches_gemv_prefill(model):
    import numpy as np

    ids = tokenize(MODEL, _code_prompt())
    assert len(ids) > 300
    batched = model.logits(ids)
    model.set_prefill(False)
    try:
        gemv = model.logits(ids)
    finally:
        model.set_prefill(True)
    assert int(np.argmax(batched)) == int(np.argmax(gemv))
    top_b = set(np.argsort(batched)[-5:].tolist())
    top_g = set(np.argsort(gemv)[-5:].tolist())
    assert len(top_b & top_g) >= 4
    assert np.corrcoef(batched, gemv)[0, 1] > 0.999


@pytest.mark.skipif(not LLAMA_BIN, reason="set LLAMA_CPP_BIN to a llama.cpp build/bin folder")
def test_matches_llama_cpp_greedy(model):
    prompt = "def quicksort(arr):\n    \"\"\"Sort a list.\"\"\"\n"
    ours = _tok().decode(model.generate(tokenize(MODEL, prompt), 32)["tokens"].tolist())
    r = subprocess.run(
        [str(Path(LLAMA_BIN) / "llama-completion"), "-m", str(MODEL), "-p", prompt, "-n", "32", "--temp", "0",
         "-ngl", "99", "-fa", "on", "-no-cnv", "--no-warmup", "--no-display-prompt"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL, timeout=300, check=True,
    )
    assert r.stdout.strip().startswith(ours.strip())


def test_prefix_cache_equals_fresh(model):
    """A multi-turn chat through generate_cached (snapshots, resumed prefill) gives the same tokens as fresh
    generate() calls on the full prompts, and the cache is actually used."""
    from adakernel.serve import ChatFormat

    fmt = ChatFormat(MODEL)
    msgs = [{"role": "system", "content": "You are a terse coding assistant."},
            {"role": "user", "content": _code_prompt().split("user\n", 1)[1].split("<|im_end|>")[0]}]
    for turn in range(3):
        ids = fmt.tok.encode(fmt.render(msgs, None))
        seen = []
        got = model.generate_cached(ids, 40, on_tokens=lambda t: seen.extend(t) and False,
                                    snap_at=fmt.snap_points(ids))
        if turn:
            assert got["cached_tokens"] > 0
        assert seen[: len(got["tokens"])] == got["tokens"].tolist()
        ref = model.generate(ids, len(got["tokens"]))["tokens"]
        assert (got["tokens"] == ref).all(), turn
        msgs += [{"role": "assistant", "content": fmt.tok.decode(got["tokens"].tolist())},
                 {"role": "user", "content": ["Shorter please.", "Now in one line."][turn % 2]}]
        # generate() reset the cache; replay this turn so the next one can hit it
        model.generate_cached(ids, 1, snap_at=fmt.snap_points(ids))


def test_cached_stops_early(model):

    ids = tokenize(MODEL, "Count: 1, 2, 3,")
    n = {"v": 0}

    def stop(t):
        n["v"] += len(t)
        return n["v"] >= 5

    r = model.generate_cached(ids, 200, on_tokens=stop)
    assert 5 <= len(r["tokens"]) < 5 + 12
