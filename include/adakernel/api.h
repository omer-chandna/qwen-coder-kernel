#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define AK_API __declspec(dllexport)
#else
#define AK_API __attribute__((visibility("default")))
#endif

typedef enum ak_status {
    AK_OK = 0,
    AK_ERR_CUDA = 1,
    AK_ERR_INVALID = 2,
    AK_ERR_OOM = 3,
    AK_ERR_TIMEOUT = 4,
    AK_ERR_UNSUPPORTED = 5
} ak_status;

enum {
    AK_BACKEND_EAGER = 0,
    AK_BACKEND_CUDA_GRAPH = 1,
    AK_BACKEND_PERSISTENT = 2,
    AK_BACKEND_HYBRID = 3,
    AK_BACKEND_FUSED = 4,
    AK_BACKEND_CUBLAS = 5,
    AK_BACKEND_CUBLAS_GRAPH = 6
};

enum {
    AK_OP_NOP = 0,
    AK_OP_RMSNORM = 1,
    AK_OP_RESIDUAL = 2,
    AK_OP_FUSED_ADD_RMSNORM = 3,
    AK_OP_ROPE = 4,
    AK_OP_MATVEC = 5,
    AK_OP_SILU_MUL = 6,
    AK_OP_ATTN_HEAD = 7,
    AK_OP_Q8_MATVEC = 8,
    AK_OP_Q4K_MATVEC = 9,
    AK_OP_Q6K_MATVEC = 15,
    AK_OP_QUANT_Q8_1 = 16,   /* p0 = x (float), p1 = xq (int8), p3 = xd (float), n0 = K */
    AK_OP_Q8_DP4A = 17,      /* p0 = W, p1 = xq, p2 = y, p3 = xd, n0 = rows, n1 = K. Optional second matrix of
                                the same type and K: p4 = W2, n2 = rows of W (W2 supplies rows n2..n0-1,
                                written to y + n2..). One launch for Q+K or gate+up. */
    AK_OP_Q4K_DP4A = 18,
    AK_OP_Q6K_DP4A = 19,
    AK_OP_EMBED_Q = 20,
    /* Fused decode ops (native/cuda/decode_ops.cuh), used by the GGUF runtime. */
    AK_OP_ADD_NORM_QUANT = 21, /* p0 x, p1 residual or 0, p2 norm w, p4 xq, p3 xd, n0 n, f0 eps */
    AK_OP_SILU_QUANT = 22,     /* p0 gate, p1 up, p2 xq, p3 xd, n0 n */
    AK_OP_QK_NORM_ROPE = 23,   /* p0 q, p1 norm w, p2 kc or 0, p3 pos, p4 v, p5 vc, n0 heads, n1 hd, n2 max_seq,
                                  f0 eps, i0 = float bits of rope theta */
    AK_OP_ATTN_DECODE = 24,    /* p0 q, p1 kc, p2 vc, p3 oq, p4 pos, p5 scores, p6 od, n0 kv head, n1 hd,
                                  n2 max_seq, n3 n_kv; i0 > 0: all i0 heads in one grid launch, i1 = n_rep */
    AK_OP_ARGMAX_PART = 25,    /* p0 logits, p1 packed best (u64), n0 n, i0 base index */
    /* Split-key attention (decode_ops.cuh). f0 > 0.5: one grid launch for every (kv head, slice) / head;
       otherwise a single one, indices packed in i1. */
    AK_OP_ATTN_PART = 26,      /* p0 q, p1 kc, p2 vc, p3 partials, p4 pos, n0 heads, n1 hd, n2 max_seq, n3 n_kv,
                                  i0 chunk, i1 = n_split | g << 8 | s << 16 */
    AK_OP_ATTN_COMBINE = 27,   /* p3 partials, p4 pos, p5 oq, p6 od, n0 heads, n1 hd, n2 max_seq, i0 chunk,
                                  i1 = n_split | h << 8 */      /* p0 = quantized table, p1 = token (int*), p2 = x, n0 = d, i0 = 0 Q8_0 / 1 Q4_K / 2 Q6_K */
    AK_OP_EMBED = 10,
    AK_OP_ARGMAX = 11,
    AK_OP_QK_NORM = 12,
    AK_OP_KV_WRITE = 13,
    AK_OP_FINALIZE = 14
};

typedef struct ak_task {
    int32_t op;
    int32_t preds_left;
    int32_t preds_init;
    int32_t succ_begin;
    int32_t n_succ;
    int32_t n0;
    int32_t n1;
    int32_t n2;
    int32_t n3;
    float f0;
    int32_t i0;
    int32_t i1;
    uint64_t p0;
    uint64_t p1;
    uint64_t p2;
    uint64_t p3;
    uint64_t p4;
    uint64_t p5;
    uint64_t p6;
} ak_task;

typedef struct ak_device_info {
    char name[256];
    char pci_bus_id[32];
    int32_t cc_major;
    int32_t cc_minor;
    int32_t sm_count;
    int32_t warp_size;
    int32_t max_threads_per_block;
    int32_t max_threads_per_sm;
    int32_t regs_per_sm;
    int32_t shared_mem_per_sm;
    int32_t shared_mem_per_block;
    int32_t shared_mem_per_block_optin;
    int32_t l2_cache_bytes;
    int32_t memory_clock_khz;
    int32_t memory_bus_width_bits;
    int32_t cooperative_launch;
    int32_t persistent_max_blocks;
    int32_t pci_bus;
    int32_t pci_device;
    int32_t clock_sm_khz;
    float total_vram_mb;
    float free_vram_mb;
    float theoretical_bandwidth_gbps;
} ak_device_info;

typedef struct ak_model_config {
    int32_t n_layers;
    int32_t d_model;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t head_dim;
    int32_t d_ff;
    int32_t vocab;
    int32_t max_seq;
    float rms_eps;
    float rope_theta;
    int32_t qk_norm;
    int32_t n_gpu_layers;
    int32_t chunk_rows;
    int32_t offload_overlap;
    int32_t persistent_blocks;
    int32_t kv_type;
    int32_t page_tokens;
} ak_model_config;

typedef struct ak_fp32_weights {
    const float* tok_embd;
    const float* attn_norm;
    const float* wq;
    const float* wk;
    const float* wv;
    const float* wo;
    const float* q_norm;
    const float* k_norm;
    const float* ffn_norm;
    const float* wgate;
    const float* wup;
    const float* wdown;
    const float* out_norm;
    const float* wout;
    int32_t tied_embeddings;
} ak_fp32_weights;

typedef struct ak_gen_stats {
    float prefill_ms;
    float decode_ms;
    float ttft_ms;
    float graph_capture_ms;
    int32_t prompt_tokens;
    int32_t generated_tokens;
    int32_t backend;
    int32_t launches_per_token;
    float decode_tps;
    float prefill_tps;
    float vram_peak_mb;
    float weight_vram_mb;
    float host_weight_mb;
    int32_t n_gpu_layers;
    int32_t persistent_blocks;
    int32_t status;
    int32_t graph_nodes;
} ak_gen_stats;

#define AK_BENCH_CAP 32

typedef struct ak_bench_row {
    char name[32];
    float median_us;
    float min_us;
    float p95_us;
    float max_abs_err;
    float rel_err;
    int32_t correct;
    int32_t iters;
} ak_bench_row;

typedef struct ak_micro_report {
    ak_bench_row rows[AK_BENCH_CAP];
    int32_t n_rows;
    int32_t sm_count;
    int32_t regs_persistent;
    int32_t persistent_blocks;
    float achieved_bandwidth_gbps;
    float theoretical_bandwidth_gbps;
    float launch_eager_us;
    float launch_graph_us;
    float launch_persistent_us;
    float coop_supported;
} ak_micro_report;

typedef struct ak_model ak_model;

AK_API const char* ak_last_error(void);
AK_API ak_status ak_init(int device);
AK_API ak_status ak_device_query(ak_device_info* out);
AK_API ak_status ak_sync(void);

AK_API ak_status ak_q8_quantize(const float* x, void* dst, int n);
AK_API ak_status ak_q8_dequantize(const void* src, float* y, int n);
AK_API ak_status ak_q4k_quantize(const float* x, void* dst, int n);
AK_API ak_status ak_q4k_dequantize(const void* src, float* y, int n);
AK_API ak_status ak_q6k_dequantize(const void* src, float* y, int n);
AK_API int ak_q8_block_size(void);
AK_API int ak_q4k_block_size(void);
AK_API int ak_q6k_block_size(void);

AK_API ak_status ak_test_rmsnorm(const float* x, const float* w, float* y, int rows, int cols, float eps);
AK_API ak_status ak_test_residual(const float* a, const float* b, float* y, int n);
AK_API ak_status ak_test_fused_add_rmsnorm(const float* x, const float* residual, const float* w,
                                           float* y_add, float* y_norm, int cols, float eps);
AK_API ak_status ak_test_rope(float* q, int n_heads, int head_dim, int pos, float theta);
AK_API ak_status ak_test_qk_norm(float* q, const float* w, int n_heads, int head_dim, float eps);
AK_API ak_status ak_test_matvec(const float* W, const float* x, float* y, int rows, int K);
AK_API ak_status ak_test_cublas_matvec(const float* W, const float* x, float* y, int rows, int K);
AK_API ak_status ak_test_silu_mul(const float* gate, const float* up, float* y, int n);
AK_API ak_status ak_test_attn(const float* q, const float* k, const float* v, float* out,
                              int n_heads, int n_kv_heads, int head_dim, int seq, int max_seq);
AK_API ak_status ak_test_q8_matvec(const void* Wq, const float* x, float* y, int rows, int K);
AK_API ak_status ak_test_q4k_matvec(const void* Wq, const float* x, float* y, int rows, int K);
AK_API ak_status ak_test_qgemv_dp4a(int type, const void* Wq, const float* x, float* y, int8_t* xq_out, float* xd_out,
                                    int rows, int K);
AK_API ak_status ak_persistent_execute(const ak_task* tasks, int n_tasks, const int32_t* succ, int n_edges,
                                       int n_blocks, int max_idle);

AK_API ak_status ak_model_create(const ak_model_config* cfg, const ak_fp32_weights* weights, ak_model** out);
AK_API ak_status ak_model_destroy(ak_model* model);
AK_API ak_status ak_model_logits(ak_model* model, const int32_t* prompt, int n_prompt, float* logits);
AK_API ak_status ak_generate(ak_model* model, const int32_t* prompt, int n_prompt, int32_t* out_tokens,
                             int n_gen, int backend, float* per_token_ms, int per_token_cap, ak_gen_stats* stats);
AK_API ak_status ak_speculate(ak_model* target, ak_model* draft, const int32_t* prompt, int n_prompt,
                              int32_t* out_tokens, int n_gen, int gamma, int32_t* out_accepts, int accept_cap,
                              ak_gen_stats* stats);
AK_API ak_status ak_microbench(int rows, int K, int seq, int n_heads, int iters, int warmup, ak_micro_report* out);

typedef struct ak_qwen_proj_report {
    int32_t n_proj;
    int32_t n_q4;
    int32_t n_q6;
    int32_t n_q8;
    float weight_gb;      /* projection weights read per token; token_embd excluded */
    float old_ms;         /* FP32-activation warp-per-row kernels, all projections */
    float fast_ms;        /* int8-activation dp4a kernels, including one activation quantize */
    float old_gbps;
    float fast_gbps;
    float q4_gbps;        /* fast path, Q4_K tensors only */
    float q6_gbps;
    float q8_gbps;
    float quant_us;       /* one activation quantize launch */
    float max_rel_err;    /* worst ||y_fast - y_old|| / ||y_old|| over projections */
    float kernel_rel_err; /* same, with the old kernel fed the dequantized int8 x: kernel error alone */
    float lpr32_gbps;     /* K-quant rows on a whole warp, interleaved A/B with lpr16 */
    float lpr16_gbps;     /* K-quant rows on a half warp (two rows per warp) */
} ak_qwen_proj_report;

/* Qwen3 GGUF runtime (native/cuda/gguf_model.cu). Weights stay Q8_0 / Q4_K / Q6_K on the device. */
typedef struct ak_gguf_model ak_gguf_model;

typedef struct ak_gguf_info {
    char arch[32];
    int32_t n_layers;
    int32_t d_model;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t head_dim;
    int32_t d_ff;
    int32_t vocab;
    int32_t max_seq;
    int32_t tied;
    int32_t eager_tasks;
    int32_t persistent_tasks;
    float rope_theta;
    float rms_eps;
    float weight_gb_per_token; /* projection + output bytes read per decoded token */
    float vram_mb;
} ak_gguf_info;

/* max_seq <= 0 means 1024. chunk_rows <= 0 means 128 (row chunk per persistent task; best of 32..512 on the 4060). */
AK_API ak_status ak_gguf_load(const char* path, int max_seq, int chunk_rows, ak_gguf_model** out, ak_gguf_info* info);
/* flags: AK_GGUF_REQUANT_Q4K re-quantizes Q8_0 matrices to Q4_K at load (fewer bytes per step; for draft models). */
#define AK_GGUF_REQUANT_Q4K 1
AK_API ak_status ak_gguf_load_ex(const char* path, int max_seq, int chunk_rows, int flags, ak_gguf_model** out,
                                 ak_gguf_info* info);
AK_API ak_status ak_gguf_destroy(ak_gguf_model* model);
/* backend: AK_BACKEND_EAGER, AK_BACKEND_CUDA_GRAPH, AK_BACKEND_PERSISTENT or AK_BACKEND_HYBRID. Greedy. */
AK_API ak_status ak_gguf_generate(ak_gguf_model* model, const int32_t* prompt, int n_prompt, int32_t* out_tokens,
                                  int n_gen, int backend, ak_gen_stats* stats);
/* Logits after the last prompt token. logits must hold info.vocab floats. */
typedef struct ak_spec_stats {
    int32_t status;
    int32_t rounds;
    int32_t drafted;
    int32_t accepted;
    int32_t generated;
    float prompt_ms;
    float decode_ms;
    float decode_tps;       /* generated tokens after the first, per second */
    float acceptance;       /* accepted / drafted */
    float tokens_per_round; /* tokens emitted per target verify pass */
    int32_t lookup_rounds;  /* rounds drafted by prompt lookup instead of the draft model */
    int32_t lookup_drafted;
    int32_t lookup_accepted;
} ak_spec_stats;

typedef struct ak_spec_options {
    int32_t max_draft;  /* 1..8 tokens per round */
    int32_t adaptive;   /* 1: model-draft length from measured costs and acceptance; 0: always max_draft */
    int32_t ngram_min;  /* prompt lookup: shortest suffix that may match (tokens) */
    int32_t ngram_max;
    int32_t use_lookup; /* 1: try prompt lookup before the draft model */
    int32_t stop_token; /* stop after emitting this token; -1 never */
    float temperature;  /* 0: greedy (output identical to greedy decoding); > 0: sample */
    float top_p;        /* nucleus mass, (0, 1] */
    int32_t top_k;      /* 1..64 */
    uint32_t seed;      /* 0: fixed default seed */
} ak_spec_options;

/* Greedy-verify n_tokens (1..9) at positions pos..pos+n-1 in one pass; out[c] = argmax after token c. */
AK_API ak_status ak_gguf_verify(ak_gguf_model* model, int pos, const int32_t* tokens, int n_tokens, int32_t* out);
/* Greedy speculative decoding: draft proposes n_draft (1..8) tokens per round, target verifies them in one pass.
   Output is identical to greedy decoding of the target alone. Both models need the same vocabulary. */
/* Speculation with prompt lookup and adaptive draft length. draft may be NULL (lookup only, or plain decoding
   with use_lookup 0). opt NULL: max_draft 6, adaptive, ngram 3..5, lookup on, no stop token, greedy. Greedy
   output equals plain greedy decoding; with temperature > 0 the draft is checked by rejection sampling, so the
   output is distributed exactly as sampling the target alone (temperature, top-k, top-p). */
AK_API ak_status ak_gguf_speculate2(ak_gguf_model* target, ak_gguf_model* draft, const int32_t* prompt, int n_prompt,
                                    int32_t* out_tokens, int n_gen, const ak_spec_options* opt, ak_spec_stats* stats);
AK_API ak_status ak_gguf_speculate(ak_gguf_model* target, ak_gguf_model* draft, const int32_t* prompt, int n_prompt,
                                   int32_t* out_tokens, int n_gen, int n_draft, ak_spec_stats* stats);
/* 1 (default): prompts go through the batched fp16 tensor-core prefill; 0: one token at a time. */
AK_API ak_status ak_gguf_set_prefill(ak_gguf_model* model, int enabled);
AK_API ak_status ak_gguf_profile(ak_gguf_model* model, float* ms_by_op, int32_t* count_by_op);
AK_API ak_status ak_gguf_logits(ak_gguf_model* model, const int32_t* prompt, int n_prompt, int backend, float* logits);

AK_API ak_status ak_bench_qwen35_projections(const char* path, int iters, ak_qwen_proj_report* out);

/* Qwen3.5 hybrid runtime (native/cuda/qwen35_model.cu): GGUF architecture "qwen35", Gated DeltaNet layers plus
   gated full attention every fourth layer. Matrices Q8_0 / Q4_K / Q6_K on the device, embedding table in pinned
   host memory, MTP block not loaded. */
typedef struct ak_q35_model ak_q35_model;

typedef struct ak_q35_info {
    char arch[32];
    int32_t n_layers;
    int32_t n_attn_layers;
    int32_t d_model;
    int32_t d_ff;
    int32_t vocab;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t head_dim;
    int32_t rope_dims;
    int32_t ssm_k_heads;
    int32_t ssm_v_heads;
    int32_t ssm_state;
    int32_t max_seq;
    float rope_theta;
    float rms_eps;
    float weight_gb_per_token; /* device matrix bytes read per decoded token */
    float vram_mb;
    float host_mb;             /* pinned host bytes (embedding table) */
    int32_t has_mtp;           /* the file's MTP block is loaded */
    float mtp_gb_per_draft;    /* bytes read per MTP draft step (MTP block + LM head) */
    int32_t max_cols;          /* widest verify / prefill pass */
} ak_q35_info;

typedef struct ak_q35_spec_stats {
    int32_t rounds;
    int32_t drafted;
    int32_t accepted;
    float acceptance;          /* accepted / drafted */
    float tokens_per_round;    /* generated tokens per verify pass */
    int32_t lookup_rounds;     /* rounds whose drafts came from prompt lookup */
    int32_t lookup_drafted;
    int32_t lookup_accepted;
    float mean_draft;          /* drafts per round */
} ak_q35_spec_stats;

/* Speculation policy for ak_q35_generate_spec. Zeroed fields take the defaults in brackets. */
typedef struct ak_q35_spec_opts {
    int32_t max_draft;    /* longest draft, 1..max_cols - 1 [11] */
    int32_t fixed;        /* 1: always max_draft MTP drafts, no lookup (ak_q35_generate_mtp) [0] */
    int32_t no_lookup;    /* 1: MTP drafts only [0] */
    int32_t ngram_min;    /* prompt lookup: shortest suffix match [2] */
    int32_t ngram_max;    /* longest suffix match tried first [6] */
    int32_t draft_vocab;  /* MTP drafts score only token ids < draft_vocab [65536]; >= vocab: all */
} ak_q35_spec_opts;

/* max_seq <= 0 means 4096. */
AK_API ak_status ak_q35_load(const char* path, int max_seq, ak_q35_model** out, ak_q35_info* info);
AK_API ak_status ak_q35_destroy(ak_q35_model* model);
/* Greedy. use_graph 1: each step is one CUDA graph replay; 0: eager launches. */
AK_API ak_status ak_q35_generate(ak_q35_model* model, const int32_t* prompt, int n_prompt, int32_t* out_tokens,
                                 int n_gen, int use_graph, ak_gen_stats* stats);
/* Greedy speculative decoding: the MTP head drafts n_draft (1..max_cols - 1) tokens, one target pass verifies them.
   Output equals ak_q35_generate's. */
AK_API ak_status ak_q35_generate_mtp(ak_q35_model* model, const int32_t* prompt, int n_prompt, int32_t* out_tokens,
                                     int n_gen, int n_draft, ak_gen_stats* stats, ak_q35_spec_stats* spec);
/* Greedy speculative decoding with a per-round choice between prompt-lookup drafts (tokens that followed an earlier
   occurrence of the current suffix) and MTP drafts, draft length picked from measured pass costs and acceptance.
   opts may be NULL. Output equals ak_q35_generate's. */
AK_API ak_status ak_q35_generate_spec(ak_q35_model* model, const int32_t* prompt, int n_prompt, int32_t* out_tokens,
                                      int n_gen, const ak_q35_spec_opts* opts, ak_gen_stats* stats,
                                      ak_q35_spec_stats* spec);
/* Called with each round's newly accepted tokens (the first call carries the token after the prompt). Return
   nonzero to stop generating. */
typedef int (*ak_q35_token_cb)(void* user, const int32_t* tokens, int n);

/* Session controls for ak_q35_generate_spec_ex. Zeroed fields: start from scratch, no snapshot, no callback. */
#define AK_Q35_CACHE_SLOTS 4

typedef struct ak_q35_run_opts {
    int32_t start;      /* > 0: the recurrent state (ak_q35_cache_load) and the KV caches already hold prompt[0, start);
                           only prompt[start, n_prompt) is processed */
    int32_t snap_at[AK_Q35_CACHE_SLOTS];    /* increasing positions start < p < n_prompt after which the recurrent
                                               state is saved; 0 ends the list */
    int32_t snap_slot[AK_Q35_CACHE_SLOTS];  /* slot for each, 0 .. AK_Q35_CACHE_SLOTS - 1 */
    ak_q35_token_cb cb;
    void* user;
} ak_q35_run_opts;

/* ak_q35_generate_spec plus prefix reuse, a mid-prompt snapshot and streaming. *n_out receives the number of tokens
   written (fewer than n_gen when the callback stopped early). */
AK_API ak_status ak_q35_generate_spec_ex(ak_q35_model* model, const int32_t* prompt, int n_prompt,
                                         int32_t* out_tokens, int n_gen, const ak_q35_spec_opts* opts,
                                         const ak_q35_run_opts* run, ak_gen_stats* stats, ak_q35_spec_stats* spec,
                                         int32_t* n_out);
/* Restore the recurrent state saved in a slot. The caller must know the KV caches still hold that prefix. */
AK_API ak_status ak_q35_cache_load(ak_q35_model* model, int slot);
/* Prompt processing path: 1 (default) batched fp16 GEMM chunks for long prompts, 0 GEMV chunks of max_cols tokens. */
AK_API ak_status ak_q35_set_prefill(ak_q35_model* model, int batched);
/* Attention kernels: 1 (default) tensor-core tiled, 0 the older per-column scalar ones (also AK_Q35_ATTN_OLD=1). */
AK_API ak_status ak_q35_set_attn(ak_q35_model* model, int tensor_core);
/* Prefill chunks of up to max_cols tokens use the fused quantized-weight GEMM, longer ones fp16 expansion + cuBLAS
   (0: always cuBLAS, < 0: always fused). Also AK_Q35_QGEMM_MAX. */
AK_API ak_status ak_q35_set_qgemm(ak_q35_model* model, int max_cols);
/* Logits after the last prompt token. */
AK_API ak_status ak_q35_logits(ak_q35_model* model, const int32_t* prompt, int n_prompt, int use_graph,
                               float* logits);
/* GPU ms of one eager decode step per kernel kind; arrays hold n_kinds (>= 18) entries. */
AK_API ak_status ak_q35_profile(ak_q35_model* model, float* ms_by_kind, int32_t* count_by_kind, int n_kinds);
/* The same for one pass of T columns at position pos (T > max_cols: one batched prefill chunk, T <= chunk length). */
AK_API ak_status ak_q35_profile_prefill(ak_q35_model* model, int T, int pos, float* ms_by_kind,
                                        int32_t* count_by_kind, int n_kinds);

#ifdef __cplusplus
}
#endif
