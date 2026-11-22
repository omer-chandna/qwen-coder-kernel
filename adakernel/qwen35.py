"""Qwen3.5 hybrid runtime (GGUF architecture "qwen35"): ctypes binding for native/cuda/qwen35_model.cu.

Gated DeltaNet linear-attention layers with gated full attention every fourth layer. Token ids come from
adakernel.qwen_tokenizer (same ids as llama.cpp's tokenizer).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from adakernel.native import GenStats, Native

# Profile buckets, in the order of the Kind enum in qwen35_model.cu.
PROFILE_KINDS = [
    "embed", "add_norm_quant", "gemv_qkv_z", "gemv_alpha_beta", "conv1d", "gated_delta", "gated_norm",
    "gemv_ssm_out", "gemv_attn_in", "attn_prep", "attn_part", "attn_combine", "gemv_attn_out", "gemv_gate_up",
    "silu_quant", "gemv_down", "gemv_logits", "argmax",
]


class Q35Info(ctypes.Structure):
    _fields_ = [
        ("arch", ctypes.c_char * 32),
        ("n_layers", ctypes.c_int32),
        ("n_attn_layers", ctypes.c_int32),
        ("d_model", ctypes.c_int32),
        ("d_ff", ctypes.c_int32),
        ("vocab", ctypes.c_int32),
        ("n_heads", ctypes.c_int32),
        ("n_kv_heads", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("rope_dims", ctypes.c_int32),
        ("ssm_k_heads", ctypes.c_int32),
        ("ssm_v_heads", ctypes.c_int32),
        ("ssm_state", ctypes.c_int32),
        ("max_seq", ctypes.c_int32),
        ("rope_theta", ctypes.c_float),
        ("rms_eps", ctypes.c_float),
        ("weight_gb_per_token", ctypes.c_float),
        ("vram_mb", ctypes.c_float),
        ("host_mb", ctypes.c_float),
        ("has_mtp", ctypes.c_int32),
        ("mtp_gb_per_draft", ctypes.c_float),
        ("max_cols", ctypes.c_int32),
    ]


class Q35SpecStats(ctypes.Structure):
    _fields_ = [
        ("rounds", ctypes.c_int32),
        ("drafted", ctypes.c_int32),
        ("accepted", ctypes.c_int32),
        ("acceptance", ctypes.c_float),
        ("tokens_per_round", ctypes.c_float),
        ("lookup_rounds", ctypes.c_int32),
        ("lookup_drafted", ctypes.c_int32),
        ("lookup_accepted", ctypes.c_int32),
        ("mean_draft", ctypes.c_float),
    ]


class Q35SpecOpts(ctypes.Structure):
    _fields_ = [
        ("max_draft", ctypes.c_int32),
        ("fixed", ctypes.c_int32),
        ("no_lookup", ctypes.c_int32),
        ("ngram_min", ctypes.c_int32),
        ("ngram_max", ctypes.c_int32),
        ("draft_vocab", ctypes.c_int32),
    ]


TOKEN_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int)
CACHE_SLOTS = 4


class Q35RunOpts(ctypes.Structure):
    _fields_ = [
        ("start", ctypes.c_int32),
        ("snap_at", ctypes.c_int32 * CACHE_SLOTS),
        ("snap_slot", ctypes.c_int32 * CACHE_SLOTS),
        ("cb", TOKEN_CB),
        ("user", ctypes.c_void_p),
    ]


def _bind(lib) -> None:
    if getattr(lib, "_q35_bound", False):
        return
    vp = ctypes.c_void_p
    lib.ak_q35_generate_spec_ex.argtypes = [vp, vp, ctypes.c_int, vp, ctypes.c_int, ctypes.POINTER(Q35SpecOpts),
                                            ctypes.POINTER(Q35RunOpts), ctypes.POINTER(GenStats),
                                            ctypes.POINTER(Q35SpecStats), ctypes.POINTER(ctypes.c_int32)]
    lib.ak_q35_generate_spec_ex.restype = ctypes.c_int
    lib.ak_q35_cache_load.argtypes = [vp, ctypes.c_int]
    lib.ak_q35_cache_load.restype = ctypes.c_int
    lib.ak_q35_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(vp), ctypes.POINTER(Q35Info)]
    lib.ak_q35_load.restype = ctypes.c_int
    lib.ak_q35_destroy.argtypes = [vp]
    lib.ak_q35_destroy.restype = ctypes.c_int
    lib.ak_q35_generate.argtypes = [vp, vp, ctypes.c_int, vp, ctypes.c_int, ctypes.c_int, ctypes.POINTER(GenStats)]
    lib.ak_q35_generate.restype = ctypes.c_int
    lib.ak_q35_generate_mtp.argtypes = [vp, vp, ctypes.c_int, vp, ctypes.c_int, ctypes.c_int, ctypes.POINTER(GenStats),
                                        ctypes.POINTER(Q35SpecStats)]
    lib.ak_q35_generate_mtp.restype = ctypes.c_int
    lib.ak_q35_generate_spec.argtypes = [vp, vp, ctypes.c_int, vp, ctypes.c_int, ctypes.POINTER(Q35SpecOpts),
                                         ctypes.POINTER(GenStats), ctypes.POINTER(Q35SpecStats)]
    lib.ak_q35_generate_spec.restype = ctypes.c_int
    lib.ak_q35_set_prefill.argtypes = [vp, ctypes.c_int]
    lib.ak_q35_set_prefill.restype = ctypes.c_int
    lib.ak_q35_set_attn.argtypes = [vp, ctypes.c_int]
    lib.ak_q35_set_attn.restype = ctypes.c_int
    lib.ak_q35_set_qgemm.argtypes = [vp, ctypes.c_int]
    lib.ak_q35_set_qgemm.restype = ctypes.c_int
    lib.ak_q35_set_int8.argtypes = [vp, ctypes.c_int, ctypes.c_int]
    lib.ak_q35_set_int8.restype = ctypes.c_int
    lib.ak_q35_logits.argtypes = [vp, vp, ctypes.c_int, ctypes.c_int, vp]
    lib.ak_q35_logits.restype = ctypes.c_int
    lib.ak_q35_profile.argtypes = [vp, vp, vp, ctypes.c_int]
    lib.ak_q35_profile.restype = ctypes.c_int
    lib.ak_q35_profile_prefill.argtypes = [vp, ctypes.c_int, ctypes.c_int, vp, vp, ctypes.c_int]
    lib.ak_q35_profile_prefill.restype = ctypes.c_int
    lib._q35_bound = True


class Qwen35Model:
    def __init__(self, path: str | Path, max_seq: int = 4096, native: Native | None = None):
        self.native = native or Native()
        _bind(self.native.lib)
        self.path = Path(path)
        handle = ctypes.c_void_p()
        info = Q35Info()
        self.native.check(
            self.native.lib.ak_q35_load(str(self.path).encode("utf-8"), max_seq, ctypes.byref(handle), ctypes.byref(info))
        )
        self.handle = handle
        self.max_seq = max_seq
        self.info = {name: getattr(info, name) for name, _ in Q35Info._fields_}
        self.info["arch"] = info.arch.decode()
        self._kv = np.zeros(0, dtype=np.int32)  # tokens whose KV entries (positions 0..len-1) are current
        self._slots: list[np.ndarray | None] = [None] * CACHE_SLOTS  # prefix each recurrent snapshot belongs to
        self._slot_used = [0] * CACHE_SLOTS
        self._clock = 0

    def close(self) -> None:
        if self.handle:
            self.native.lib.ak_q35_destroy(self.handle)
            self.handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def generate(self, prompt, n_gen: int, graph: bool = True) -> dict:
        tokens = np.ascontiguousarray(prompt, dtype=np.int32)
        out = np.empty(n_gen, dtype=np.int32)
        stats = GenStats()
        self.reset_cache()
        self.native.check(
            self.native.lib.ak_q35_generate(
                self.handle, Native._ptr(tokens), int(tokens.size), Native._ptr(out), int(n_gen), int(graph),
                ctypes.byref(stats),
            )
        )
        return {
            "tokens": out,
            "prompt_ms": stats.prefill_ms,
            "decode_ms": stats.decode_ms,
            "decode_tps": stats.decode_tps,
            "graph_capture_ms": stats.graph_capture_ms,
            "vram_mb": stats.vram_peak_mb,
        }

    def generate_mtp(self, prompt, n_gen: int, n_draft: int = 3) -> dict:
        """Greedy speculative decoding with the file's MTP head as the draft. Same tokens as generate()."""
        tokens = np.ascontiguousarray(prompt, dtype=np.int32)
        out = np.empty(n_gen, dtype=np.int32)
        stats = GenStats()
        spec = Q35SpecStats()
        self.reset_cache()
        self.native.check(
            self.native.lib.ak_q35_generate_mtp(
                self.handle, Native._ptr(tokens), int(tokens.size), Native._ptr(out), int(n_gen), int(n_draft),
                ctypes.byref(stats), ctypes.byref(spec),
            )
        )
        return self._spec_result(out, stats, spec)

    def generate_spec(self, prompt, n_gen: int, max_draft: int = 11, lookup: bool = True, ngram_min: int = 2,
                      ngram_max: int = 6, draft_vocab: int = 65536) -> dict:
        """Greedy speculative decoding, choosing per round between prompt-lookup and MTP drafts and the draft length
        from measured costs. MTP drafts score only token ids < draft_vocab (the verify pass always scores all).
        Same tokens as generate()."""
        tokens = np.ascontiguousarray(prompt, dtype=np.int32)
        out = np.empty(n_gen, dtype=np.int32)
        opts = Q35SpecOpts(max_draft=max_draft, fixed=0, no_lookup=int(not lookup), ngram_min=ngram_min,
                           ngram_max=ngram_max, draft_vocab=draft_vocab)
        stats = GenStats()
        spec = Q35SpecStats()
        self.reset_cache()
        self.native.check(
            self.native.lib.ak_q35_generate_spec(
                self.handle, Native._ptr(tokens), int(tokens.size), Native._ptr(out), int(n_gen), ctypes.byref(opts),
                ctypes.byref(stats), ctypes.byref(spec),
            )
        )
        return self._spec_result(out, stats, spec)

    # ------------------------------------------------------------------ prefix-cached streaming generation

    def _slot_valid(self, i: int) -> bool:
        s = self._slots[i]
        return s is not None and s.size <= self._kv.size and bool(np.array_equal(self._kv[: s.size], s))

    def generate_cached(self, prompt, n_gen: int, on_tokens=None, snap_at=(), max_draft: int = 11,
                        lookup: bool = True, draft_vocab: int = 65536) -> dict:
        """generate_spec that reuses earlier work and streams.

        The longest saved recurrent-state snapshot whose prefix starts `prompt` (and whose KV entries are still in
        place) is restored, so only the rest of the prompt is processed. snap_at lists up to two prompt positions
        after which the state is saved for later calls - e.g. the end of the system prompt and the start of the
        newest chat message, points a follow-up request is likely to share. on_tokens(list[int]) gets each round's
        new tokens and may return True to stop. Tokens equal generate()'s for the same prompt."""
        if isinstance(snap_at, int):
            snap_at = (snap_at,)
        tokens = np.ascontiguousarray(prompt, dtype=np.int32)
        n_prompt = int(tokens.size)
        n_gen = int(min(n_gen, self.max_seq - n_prompt - 12))
        if n_gen < 1:
            raise ValueError(f"prompt of {n_prompt} tokens leaves no room in max_seq={self.max_seq}")
        best, start = -1, 0
        for i in range(CACHE_SLOTS):
            s = self._slots[i]
            if s is None or s.size >= n_prompt or s.size <= start or not self._slot_valid(i):
                continue
            if np.array_equal(tokens[: s.size], s):
                best, start = i, int(s.size)
        if best >= 0:
            self.native.check(self.native.lib.ak_q35_cache_load(self.handle, best))
            self._clock += 1
            self._slot_used[best] = self._clock
        def survives(i):  # slot still usable after this call rewrites positions >= start
            s = self._slots[i]
            return (s is not None and self._slot_valid(i)
                    and (s.size <= start or (s.size <= n_prompt and np.array_equal(tokens[: s.size], s))))

        # Slots matching this call's requested points are kept (protected from eviction) rather than re-saved.
        points = sorted({int(a) for a in snap_at if 0 < int(a) < n_prompt})
        taken = {best}
        for at in points:
            for i in range(CACHE_SLOTS):
                if survives(i) and self._slots[i].size == at:
                    taken.add(i)
        snaps: list[tuple[int, int]] = []
        for at in points:
            if at <= start or any(survives(i) and self._slots[i].size == at for i in range(CACHE_SLOTS)):
                continue
            free = [i for i in range(CACHE_SLOTS) if i not in taken and not survives(i)]
            if not free:
                free = sorted((i for i in range(CACHE_SLOTS) if i not in taken), key=lambda i: self._slot_used[i])
            if not free:
                break
            snaps.append((at, free[0]))
            taken.add(free[0])
        # from here on positions >= start are being rewritten
        self._kv = tokens[:start].copy()
        for _, i in snaps:
            self._slots[i] = None

        out = np.empty(n_gen, dtype=np.int32)
        err: list[BaseException] = []

        def cb(_user, ptr, n):
            if on_tokens is None:
                return 0
            try:
                return 1 if on_tokens([ptr[i] for i in range(n)]) else 0
            except BaseException as e:  # noqa: BLE001 - re-raised after the native call returns
                err.append(e)
                return 1

        cfn = TOKEN_CB(cb)
        run = Q35RunOpts(start=start, cb=cfn, user=None)
        for j, (at, i) in enumerate(snaps):
            run.snap_at[j] = at
            run.snap_slot[j] = i
        opts = Q35SpecOpts(max_draft=max_draft, fixed=0, no_lookup=int(not lookup), ngram_min=2, ngram_max=6,
                           draft_vocab=draft_vocab)
        stats = GenStats()
        spec = Q35SpecStats()
        n_out = ctypes.c_int32(0)
        st = self.native.lib.ak_q35_generate_spec_ex(
            self.handle, Native._ptr(tokens), n_prompt, Native._ptr(out), n_gen, ctypes.byref(opts),
            ctypes.byref(run), ctypes.byref(stats), ctypes.byref(spec), ctypes.byref(n_out))
        if st != 0:
            self._kv = np.zeros(0, dtype=np.int32)
            self._slots = [None] * CACHE_SLOTS
            self.native.check(st)
        k = int(n_out.value)
        for at, i in snaps:
            self._slots[i] = tokens[:at].copy()
            self._clock += 1
            self._slot_used[i] = self._clock
        # every emitted token but the last has been fed through the model
        self._kv = np.concatenate([tokens, out[: max(k - 1, 0)]])
        if err:
            raise err[0]
        r = self._spec_result(out[:k], stats, spec)
        r["cached_tokens"] = start
        r["prompt_tokens"] = n_prompt
        return r

    def reset_cache(self) -> None:
        self._kv = np.zeros(0, dtype=np.int32)
        self._slots = [None] * CACHE_SLOTS

    @staticmethod
    def _spec_result(out, stats, spec) -> dict:
        return {
            "tokens": out,
            "prompt_ms": stats.prefill_ms,
            "decode_ms": stats.decode_ms,
            "decode_tps": stats.decode_tps,
            "rounds": spec.rounds,
            "acceptance": spec.acceptance,
            "tokens_per_round": spec.tokens_per_round,
            "lookup_rounds": spec.lookup_rounds,
            "lookup_acceptance": spec.lookup_accepted / spec.lookup_drafted if spec.lookup_drafted else 0.0,
            "mean_draft": spec.mean_draft,
        }

    def set_prefill(self, batched: bool) -> None:
        """True (default): long prompts go through fp16 GEMM chunks; False: GEMV chunks of max_cols tokens."""
        self.native.check(self.native.lib.ak_q35_set_prefill(self.handle, int(batched)))

    def set_attention(self, tensor_core: bool) -> None:
        """True (default): tensor-core tiled attention; False: the older per-column scalar kernels."""
        self.native.check(self.native.lib.ak_q35_set_attn(self.handle, int(tensor_core)))

    def set_qgemm(self, max_cols: int) -> None:
        """Prefill chunks of up to max_cols tokens use the fused quantized GEMM (0: always cuBLAS, -1: always fused)."""
        self.native.check(self.native.lib.ak_q35_set_qgemm(self.handle, int(max_cols)))

    def set_int8(self, on: bool, max_cols: int = -1) -> None:
        """True (default): prefill matrix products on int8 tensor cores (decode-path Q8_1 accuracy) for chunks of up
        to max_cols tokens (-1: any); False: the fp16 fused kernel / cuBLAS (set_qgemm)."""
        self.native.check(self.native.lib.ak_q35_set_int8(self.handle, int(on), int(max_cols)))

    def logits(self, prompt, graph: bool = True) -> np.ndarray:
        tokens = np.ascontiguousarray(prompt, dtype=np.int32)
        out = np.empty(self.info["vocab"], dtype=np.float32)
        self.reset_cache()
        self.native.check(
            self.native.lib.ak_q35_logits(self.handle, Native._ptr(tokens), int(tokens.size), int(graph), Native._ptr(out))
        )
        return out

    def profile(self, T: int = 1, pos: int = 0) -> dict:
        """GPU ms by kernel kind of one pass: T = 1 a decode step, T > 12 one batched prefill chunk at pos."""
        n = len(PROFILE_KINDS)
        ms = np.zeros(n, dtype=np.float32)
        cnt = np.zeros(n, dtype=np.int32)
        self.reset_cache()
        self.native.check(self.native.lib.ak_q35_profile_prefill(self.handle, int(T), int(pos), Native._ptr(ms),
                                                                 Native._ptr(cnt), n))
        return {k: (float(ms[i]), int(cnt[i])) for i, k in enumerate(PROFILE_KINDS) if cnt[i]}
