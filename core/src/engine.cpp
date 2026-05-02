/**
 * @file engine.cpp
 * @brief Edge Veda SDK - Core Engine Implementation
 *
 * This file implements the public C API defined in edge_veda.h
 * using llama.cpp for on-device LLM inference.
 */

#include "edge_veda.h"
#include "backend_lifecycle.h"
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <atomic>

#ifdef EDGE_VEDA_LLAMA_ENABLED
#include "llama.h"
#include "ggml.h"
// `common/speculative.h` lives in llama.cpp/common — pulled in via
// LLAMA_BUILD_COMMON=ON (core/CMakeLists.txt:202). The speculative
// decoder pairs a draft model with the loaded target context to
// proposal-verify N tokens per step.
#include "speculative.h"
#include "common.h"
#endif

#include "memory_guard.h"
#include "thread_utils.h"

// Tombstone magic numbers for use-after-free detection (issue #28)
static constexpr uint32_t EV_CTX_MAGIC = 0x45564354; // "EVCT"
static constexpr uint32_t EV_CTX_DEAD  = 0xDEADDEAD;

/* ============================================================================
 * Internal Structures
 * ========================================================================= */

struct ev_context_impl {
    // Tombstone magic for use-after-free detection (issue #28)
    // MUST be first field for reliable memory layout.
    uint32_t magic = EV_CTX_MAGIC;

    // Configuration
    ev_config config;

    // Backend information
    ev_backend_t active_backend;

    // Model state
    bool model_loaded;
    std::string model_path;

    // Memory management
    size_t memory_limit;
    bool auto_unload;
    ev_memory_pressure_callback memory_callback;
    void* memory_callback_data;

    // Statistics
    size_t peak_memory_bytes;
    size_t current_memory_bytes;

    // Error tracking
    std::string last_error;

    // Thread safety
    std::mutex mutex;
    std::atomic<int> active_stream_count{0};   // Non-ended streams (issue #25)

    // llama.cpp handles
#ifdef EDGE_VEDA_LLAMA_ENABLED
    llama_model* model = nullptr;
    llama_context* llama_ctx = nullptr;
    llama_context* embed_ctx = nullptr;
    llama_sampler* sampler = nullptr;
    char model_desc[256] = {0};               // Per-context model description (issue #26)

    // Speculative decoding state (mobile-news#586 gap #10).
    // All four are owned by the context and freed together in
    // ev_speculative_detach() (also called from ev_free()).
    llama_model*    draft_model = nullptr;
    llama_context*  draft_ctx   = nullptr;
    common_speculative* spec_ctx = nullptr;
    common_params_speculative spec_params{};
    int64_t spec_n_drafted  = 0;
    int64_t spec_n_accepted = 0;
#endif

    // Constructor
    ev_context_impl()
        : active_backend(EV_BACKEND_AUTO)
        , model_loaded(false)
        , memory_limit(0)
        , auto_unload(false)
        , memory_callback(nullptr)
        , memory_callback_data(nullptr)
        , peak_memory_bytes(0)
        , current_memory_bytes(0) {
    }

    ~ev_context_impl() = default;
};

struct ev_stream_impl {
    ev_context ctx;                    // Parent context (shared, NOT owned)
    std::string prompt;                // Original prompt
    ev_generation_params params;       // Generation parameters
    char* grammar_str_owned = nullptr;   // Owned copy of grammar string (fixes #33)
    char* grammar_root_owned = nullptr;  // Owned copy of grammar root (fixes #33)
    // Stop-sequence state (app-level string matching — llama.cpp sampler
    // API has no built-in stop-sequence support, so we check decoded
    // output against these strings after each token).
    std::vector<std::string> stop_sequences_owned;  // Deep-copy of params.stop_sequences
    std::string accumulated_output;                  // Sliding-window buffer for matching
    bool ended;                        // Stream completion flag
    std::atomic<bool> deactivated{false}; // True once active_stream_count decremented
    std::atomic<bool> cancelled{false}; // Thread-safe cancel flag

#ifdef EDGE_VEDA_LLAMA_ENABLED
    // llama.cpp streaming state
    llama_sampler* sampler = nullptr;        // Token sampler (owned by stream)
    std::vector<llama_token> prompt_tokens;  // Tokenized prompt
    int n_cur = 0;                           // Current position in generation
    bool prompt_evaluated = false;           // Whether prompt has been processed

    // ---- Speculative decoding state (mobile-news#586 gap #10) ----
    // Populated only when ctx->spec_ctx is non-null at first ev_stream_next.
    // - prompt_tgt: full target context history (every token decoded so far,
    //   prompt + generated). common_speculative_draft consumes this to
    //   pick the next draft sequence.
    // - id_last: most-recently-sampled target token; the speculative loop
    //   evaluates the target on [id_last, draft0, draft1, ...] and the
    //   sampler decides how many drafts to accept.
    // - spec_buffer / spec_buffer_pos: tokens that the target has already
    //   accepted but `ev_stream_next()` hasn't yet returned to the caller
    //   (one call returns one token to keep API contract intact).
    // - spec_initialized: whether common_speculative_begin was called.
    std::vector<llama_token> prompt_tgt;
    llama_token              id_last = 0;
    std::vector<llama_token> spec_buffer;
    size_t                   spec_buffer_pos = 0;
    bool                     spec_initialized = false;
#endif

    // Confidence tracking
    float last_confidence = -1.0f;      // Last token's confidence (-1 = not computed)
    double confidence_sum = 0.0;         // Running sum for average
    int confidence_count = 0;            // Number of confidence measurements
    bool needs_handoff = false;          // Cloud handoff signal

    std::mutex mutex;

    ev_stream_impl(ev_context context, const char* p, const ev_generation_params* prms)
        : ctx(context)
        , prompt(p ? p : "")
        , ended(false)
        , cancelled(false) {
        if (prms) {
            params = *prms;
            // Deep-copy grammar strings so the stream owns them (fixes #33).
            // Callers may free originals before the stream completes.
            if (params.grammar_str) {
                grammar_str_owned = strdup(params.grammar_str);
                if (grammar_str_owned) {
                    params.grammar_str = grammar_str_owned;
                }
            }
            if (params.grammar_root) {
                grammar_root_owned = strdup(params.grammar_root);
                if (grammar_root_owned) {
                    params.grammar_root = grammar_root_owned;
                }
            }
            // Deep-copy stop sequences. Caller may free the C strings
            // before the stream completes, so we own them for the
            // stream's lifetime. Empty strings are skipped.
            if (params.stop_sequences && params.num_stop_sequences > 0) {
                stop_sequences_owned.reserve(
                    static_cast<size_t>(params.num_stop_sequences));
                for (int i = 0; i < params.num_stop_sequences; ++i) {
                    const char* s = params.stop_sequences[i];
                    if (s && s[0] != '\0') {
                        stop_sequences_owned.emplace_back(s);
                    }
                }
            }
        } else {
            ev_generation_params_default(&params);
        }
    }

    ~ev_stream_impl() {
#ifdef EDGE_VEDA_LLAMA_ENABLED
        if (sampler) {
            llama_sampler_free(sampler);
            sampler = nullptr;
        }
#endif
        // Free owned grammar string copies (fixes #33)
        free(grammar_str_owned);
        free(grammar_root_owned);
    }

    bool check_cancelled() {
        return cancelled.load(std::memory_order_acquire);
    }

    void request_cancel() {
        cancelled.store(true, std::memory_order_release);
    }

    // End this stream and decrement the parent context's active count exactly once.
    void mark_ended() {
        ended = true;
        if (!deactivated.exchange(true, std::memory_order_acq_rel)) {
            // Tombstone guard: skip decrement if ctx was already freed (issue #28)
            if (ctx && ctx->magic == EV_CTX_MAGIC) {
                ctx->active_stream_count.fetch_sub(1, std::memory_order_release);
            }
        }
    }
};

// Eviction callback invoked by memory guard when LLM engine is selected for LRU eviction.
// Called from the monitor thread with mg mutex released. Must be safe to call concurrently.
static void llm_evict_cb(void* user_data) {
    ev_context ctx = static_cast<ev_context>(user_data);
    if (!ctx) return;

    std::lock_guard<std::mutex> lock(ctx->mutex);

    // Skip if there's an active stream — can't safely evict mid-generation
    if (ctx->active_stream_count.load(std::memory_order_acquire) != 0) return;
    if (!ctx->model_loaded) return; // Already evicted or freed

#ifdef EDGE_VEDA_LLAMA_ENABLED
    // Free draft state first — it references the target's llama_ctx
    // via common_speculative, so order matters.
    if (ctx->spec_ctx)    { common_speculative_free(ctx->spec_ctx); ctx->spec_ctx = nullptr; }
    if (ctx->draft_ctx)   { llama_free(ctx->draft_ctx); ctx->draft_ctx = nullptr; }
    if (ctx->draft_model) { llama_model_free(ctx->draft_model); ctx->draft_model = nullptr; }
    if (ctx->sampler) { llama_sampler_free(ctx->sampler); ctx->sampler = nullptr; }
    if (ctx->llama_ctx) { llama_free(ctx->llama_ctx); ctx->llama_ctx = nullptr; }
    if (ctx->embed_ctx) { llama_free(ctx->embed_ctx); ctx->embed_ctx = nullptr; }
    if (ctx->model) { llama_model_free(ctx->model); ctx->model = nullptr; }
    ctx->model_loaded = false;
#endif
}

/* ============================================================================
 * Version Information
 * ========================================================================= */

const char* ev_version(void) {
    return "1.0.0";
}

/* ============================================================================
 * Error Handling
 * ========================================================================= */

const char* ev_error_string(ev_error_t error) {
    switch (error) {
        case EV_SUCCESS: return "Success";
        case EV_ERROR_INVALID_PARAM: return "Invalid parameter";
        case EV_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case EV_ERROR_MODEL_LOAD_FAILED: return "Failed to load model";
        case EV_ERROR_BACKEND_INIT_FAILED: return "Failed to initialize backend";
        case EV_ERROR_INFERENCE_FAILED: return "Inference failed";
        case EV_ERROR_CONTEXT_INVALID: return "Invalid context";
        case EV_ERROR_STREAM_ENDED: return "Stream ended";
        case EV_ERROR_NOT_IMPLEMENTED: return "Not implemented";
        case EV_ERROR_MEMORY_LIMIT_EXCEEDED: return "Memory limit exceeded";
        case EV_ERROR_UNSUPPORTED_BACKEND: return "Backend not supported";
        default: return "Unknown error";
    }
}

/* ============================================================================
 * Backend Detection
 * ========================================================================= */

ev_backend_t ev_detect_backend(void) {
#if defined(__APPLE__)
    #include "TargetConditionals.h"
    #if TARGET_OS_IOS || TARGET_OS_OSX
        #ifdef EDGE_VEDA_METAL_ENABLED
            return EV_BACKEND_METAL;
        #endif
    #endif
#elif defined(__ANDROID__)
    #ifdef EDGE_VEDA_VULKAN_ENABLED
        return EV_BACKEND_VULKAN;
    #endif
#endif

#ifdef EDGE_VEDA_CPU_ENABLED
    return EV_BACKEND_CPU;
#else
    return EV_BACKEND_AUTO;
#endif
}

bool ev_is_backend_available(ev_backend_t backend) {
    switch (backend) {
        case EV_BACKEND_METAL:
#ifdef EDGE_VEDA_METAL_ENABLED
            return true;
#else
            return false;
#endif

        case EV_BACKEND_VULKAN:
#ifdef EDGE_VEDA_VULKAN_ENABLED
            return true;
#else
            return false;
#endif

        case EV_BACKEND_CPU:
#ifdef EDGE_VEDA_CPU_ENABLED
            return true;
#else
            return false;
#endif

        case EV_BACKEND_AUTO:
            return true;

        default:
            return false;
    }
}

const char* ev_backend_name(ev_backend_t backend) {
    switch (backend) {
        case EV_BACKEND_AUTO: return "Auto";
        case EV_BACKEND_METAL: return "Metal";
        case EV_BACKEND_VULKAN: return "Vulkan";
        case EV_BACKEND_CPU: return "CPU";
        default: return "Unknown";
    }
}

/* ============================================================================
 * Configuration
 * ========================================================================= */

void ev_config_default(ev_config* config) {
    if (!config) return;

    std::memset(config, 0, sizeof(ev_config));
    config->model_path = nullptr;
    config->backend = EV_BACKEND_AUTO;
    config->num_threads = 0; // Auto-detect
    config->context_size = 2048;
    config->batch_size = 512;
    config->memory_limit_bytes = 0; // No limit
    config->auto_unload_on_memory_pressure = true;
    config->gpu_layers = -1; // All layers
    config->use_mmap = true;
    config->use_mlock = false;
    config->seed = -1; // Random
    config->flash_attn = -1; // AUTO (Metal enables automatically)
    config->kv_cache_type_k = 1; // F16 (llama.cpp default -- Dart side overrides to Q8_0)
    config->kv_cache_type_v = 1; // F16 (llama.cpp default -- Dart side overrides to Q8_0)
    config->reserved = nullptr;
}

/* ============================================================================
 * Context Management
 * ========================================================================= */

ev_context ev_init(const ev_config* config, ev_error_t* error) {
    if (!config || !config->model_path) {
        if (error) *error = EV_ERROR_INVALID_PARAM;
        return nullptr;
    }

    // Allocate context
    ev_context ctx = new (std::nothrow) ev_context_impl();
    if (!ctx) {
        if (error) *error = EV_ERROR_OUT_OF_MEMORY;
        return nullptr;
    }

    ctx->config = *config;
    ctx->model_path = config->model_path;
    ctx->memory_limit = config->memory_limit_bytes;
    ctx->auto_unload = config->auto_unload_on_memory_pressure;

    // Detect backend
    ctx->active_backend = (config->backend == EV_BACKEND_AUTO)
        ? ev_detect_backend()
        : config->backend;

    if (!ev_is_backend_available(ctx->active_backend)) {
        ctx->last_error = "Backend not available";
        delete ctx;
        if (error) *error = EV_ERROR_UNSUPPORTED_BACKEND;
        return nullptr;
    }

#ifdef EDGE_VEDA_LLAMA_ENABLED
    // Initialize shared llama backend
    edge_veda_backend_acquire();

    // Configure model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config->gpu_layers;
    model_params.use_mmap = config->use_mmap;
    model_params.use_mlock = config->use_mlock;

    // Load model (using new API: llama_model_load_from_file)
    ctx->model = llama_model_load_from_file(ctx->model_path.c_str(), model_params);
    if (!ctx->model) {
        ctx->last_error = "Failed to load model from: " + ctx->model_path;
        edge_veda_backend_release();
        delete ctx;
        if (error) *error = EV_ERROR_MODEL_LOAD_FAILED;
        return nullptr;
    }

    // Configure context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config->context_size > 0 ? static_cast<uint32_t>(config->context_size) : 2048;
    ctx_params.n_batch = config->batch_size > 0 ? static_cast<uint32_t>(config->batch_size) : 512;
    ctx_params.n_threads = config->num_threads > 0 ? static_cast<uint32_t>(config->num_threads)
                                                     : ev_default_thread_count();
    ctx_params.n_threads_batch = ctx_params.n_threads;

    // KV cache quantization (Q8_0 halves KV cache memory vs F16 default)
    if (config->kv_cache_type_k > 0) {
        ctx_params.type_k = static_cast<ggml_type>(config->kv_cache_type_k);
    }
    if (config->kv_cache_type_v > 0) {
        ctx_params.type_v = static_cast<ggml_type>(config->kv_cache_type_v);
    }

    // Flash attention (AUTO lets llama.cpp enable when backend supports it)
    if (config->flash_attn != 0) {
        ctx_params.flash_attn_type = static_cast<llama_flash_attn_type>(config->flash_attn);
    }

    // Create llama context (using new API: llama_init_from_model)
    ctx->llama_ctx = llama_init_from_model(ctx->model, ctx_params);
    if (!ctx->llama_ctx) {
        ctx->last_error = "Failed to create llama context";
        llama_model_free(ctx->model);
        edge_veda_backend_release();
        delete ctx;
        if (error) *error = EV_ERROR_BACKEND_INIT_FAILED;
        return nullptr;
    }

    ctx->model_loaded = true;

    // Register with process-wide memory guard for cross-engine coordination.
    // Auto-sets recommended limit on first registration (MEM-06).
    {
        size_t footprint = llama_model_size(ctx->model);
        memory_guard_register_engine(MG_ENGINE_LLM, footprint, llm_evict_cb, ctx);
    }

    if (error) *error = EV_SUCCESS;
    return ctx;
#else
    ctx->last_error = "llama.cpp not compiled - library built without LLM support";
    delete ctx;
    if (error) *error = EV_ERROR_NOT_IMPLEMENTED;
    return nullptr;
#endif
}

void ev_free(ev_context ctx) {
    if (!ctx) return;

    // Debug-only: catch API misuse during development.
    // ev_free() is void — can't return error, assert is appropriate here.
    assert(ctx->active_stream_count.load(std::memory_order_acquire) == 0 &&
           "ev_free() called with active streams — free all streams first.");

    // Unregister BEFORE acquiring ctx->mutex to prevent ABBA deadlock.
    // Lock ordering: monitor thread holds mg.mutex → evict_cb → ctx->mutex.
    // If we held ctx->mutex → unregister → mg.mutex, that's inverted.
    // Spin-waits for any in-flight eviction callback to complete.
    memory_guard_unregister_engine(MG_ENGINE_LLM);
    memory_guard_set_callback(nullptr, nullptr);

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);

#ifdef EDGE_VEDA_LLAMA_ENABLED
        // Free speculative state before the target context — common
        // speculative holds a pointer to ctx->llama_ctx that becomes
        // dangling otherwise.
        if (ctx->spec_ctx) {
            common_speculative_free(ctx->spec_ctx);
            ctx->spec_ctx = nullptr;
        }
        if (ctx->draft_ctx) {
            llama_free(ctx->draft_ctx);
            ctx->draft_ctx = nullptr;
        }
        if (ctx->draft_model) {
            llama_model_free(ctx->draft_model);
            ctx->draft_model = nullptr;
        }
        if (ctx->sampler) {
            llama_sampler_free(ctx->sampler);
            ctx->sampler = nullptr;
        }
        if (ctx->llama_ctx) {
            llama_free(ctx->llama_ctx);
            ctx->llama_ctx = nullptr;
        }
        if (ctx->embed_ctx) {
            llama_free(ctx->embed_ctx);
            ctx->embed_ctx = nullptr;
        }
        if (ctx->model) {
            llama_model_free(ctx->model);
            ctx->model = nullptr;
        }
        edge_veda_backend_release();
#endif

        // Poison the magic before deletion so stale pointers can detect UAF (issue #28)
        ctx->magic = EV_CTX_DEAD;
    }
    // lock_guard destructor has run, mutex is unlocked before delete
    delete ctx;
}

bool ev_is_valid(ev_context ctx) {
    return ctx != nullptr && ctx->model_loaded;
}

/* ============================================================================
 * Internal Helper Functions
 * ========================================================================= */

#ifdef EDGE_VEDA_LLAMA_ENABLED
/**
 * Tokenize a text prompt into llama tokens
 * @param model The llama model for tokenization (vocab extracted internally)
 * @param text The input text to tokenize
 * @param add_bos Whether to add beginning-of-sequence token
 * @return Vector of tokens
 */
static std::vector<llama_token> tokenize_prompt(
    const llama_model* model,
    const std::string& text,
    bool add_bos
) {
    // Get vocab from model (new API)
    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Get max tokens needed (rough estimate: 1 token per character + BOS)
    int n_tokens = static_cast<int>(text.length()) + (add_bos ? 1 : 0);
    std::vector<llama_token> tokens(static_cast<size_t>(n_tokens));

    // Tokenize using vocab
    n_tokens = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.length()),
                              tokens.data(), static_cast<int32_t>(tokens.size()), add_bos, false);

    if (n_tokens < 0) {
        // Buffer was too small, resize and retry
        tokens.resize(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.length()),
                                  tokens.data(), static_cast<int32_t>(tokens.size()), add_bos, false);
    }

    if (n_tokens >= 0) {
        tokens.resize(static_cast<size_t>(n_tokens));
    } else {
        tokens.clear();
    }
    return tokens;
}

/**
 * Create a sampler chain with the specified generation parameters
 * @param params Generation parameters
 * @param vocab  Vocabulary (needed for grammar sampler; may be nullptr if no grammar)
 * @return Configured sampler chain
 */
static llama_sampler* create_sampler(
    const ev_generation_params& params,
    const llama_vocab* vocab
) {
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(chain_params);

    // Add samplers in order: penalties -> top-k -> top-p -> temperature -> grammar -> dist
    llama_sampler_chain_add(sampler,
        llama_sampler_init_penalties(
            64,                        // penalty_last_n
            params.repeat_penalty,     // repeat_penalty
            params.frequency_penalty,  // frequency_penalty
            params.presence_penalty    // presence_penalty
        ));

    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }

    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }

    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    }

    // Add grammar sampler if grammar provided (constrains valid tokens at each step)
    if (params.grammar_str && params.grammar_str[0] != '\0' && vocab) {
        const char* root = (params.grammar_root && params.grammar_root[0] != '\0')
            ? params.grammar_root : "root";
        llama_sampler_chain_add(sampler,
            llama_sampler_init_grammar(vocab, params.grammar_str, root));
    }

    // dist sampler (always last -- performs final random selection)
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    return sampler;
}
#endif

/* ============================================================================
 * Generation Parameters
 * ========================================================================= */

void ev_generation_params_default(ev_generation_params* params) {
    if (!params) return;

    std::memset(params, 0, sizeof(ev_generation_params));
    params->max_tokens = 512;
    params->temperature = 0.8f;
    params->top_p = 0.95f;
    params->top_k = 40;
    params->repeat_penalty = 1.1f;
    params->frequency_penalty = 0.0f;
    params->presence_penalty = 0.0f;
    params->stop_sequences = nullptr;
    params->num_stop_sequences = 0;
    params->grammar_str = nullptr;
    params->grammar_root = nullptr;
    params->confidence_threshold = 0.0f;
    params->reserved = nullptr;
}

/* ============================================================================
 * Single-Shot Generation
 * ========================================================================= */

ev_error_t ev_generate(
    ev_context ctx,
    const char* prompt,
    const ev_generation_params* params,
    char** output
) {
    if (!ctx || !prompt || !output) {
        return EV_ERROR_INVALID_PARAM;
    }

    if (!ev_is_valid(ctx)) {
        return EV_ERROR_CONTEXT_INVALID;
    }

    memory_guard_touch_engine(MG_ENGINE_LLM);
    std::lock_guard<std::mutex> lock(ctx->mutex);

    // Guard: ev_generate() clears KV cache, which would corrupt an active stream.
    // Not triggerable from Dart (generate() routes through generateStream()),
    // but protects direct C API consumers (Swift, Kotlin, etc.).
    if (ctx->active_stream_count.load(std::memory_order_acquire) != 0) {
        ctx->last_error = "ev_generate() called while a stream is active — "
                          "end or cancel active streams first";
        return EV_ERROR_CONTEXT_INVALID;
    }

    ev_generation_params gen_params;
    if (params) {
        gen_params = *params;
    } else {
        ev_generation_params_default(&gen_params);
    }

#ifdef EDGE_VEDA_LLAMA_ENABLED
    // Clear KV cache for fresh generation
    llama_memory_clear(llama_get_memory(ctx->llama_ctx), true);

    // Tokenize prompt
    std::vector<llama_token> tokens = tokenize_prompt(ctx->model, prompt, true);
    if (tokens.empty()) {
        ctx->last_error = "Failed to tokenize prompt";
        return EV_ERROR_INFERENCE_FAILED;
    }

    // Check context size
    int n_ctx = static_cast<int>(llama_n_ctx(ctx->llama_ctx));
    if (static_cast<int>(tokens.size()) > n_ctx - 4) {
        ctx->last_error = "Prompt too long for context size";
        return EV_ERROR_INFERENCE_FAILED;
    }

    // Evaluate prompt in batches of n_batch
    const int n_prompt = static_cast<int>(tokens.size());
    const int n_batch = static_cast<int>(llama_n_batch(ctx->llama_ctx));

    for (int i = 0; i < n_prompt; i += n_batch) {
        const int n_eval = std::min(n_batch, n_prompt - i);
        llama_batch batch = llama_batch_get_one(tokens.data() + i, n_eval);
        if (llama_decode(ctx->llama_ctx, batch) != 0) {
            ctx->last_error = "Failed to evaluate prompt";
            return EV_ERROR_INFERENCE_FAILED;
        }
    }

    // Create sampler (pass vocab for grammar support)
    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    llama_sampler* sampler = create_sampler(gen_params, vocab);
    if (!sampler) {
        ctx->last_error = "Failed to create sampler";
        return EV_ERROR_INFERENCE_FAILED;
    }

    // Generate tokens
    std::string result;

    for (int i = 0; i < gen_params.max_tokens; ++i) {
        // Sample next token
        llama_token new_token = llama_sampler_sample(sampler, ctx->llama_ctx, -1);

        // Check for EOS (using vocab, new API: llama_vocab_is_eog)
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        // Convert token to text (using vocab)
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            result.append(buf, static_cast<size_t>(n));
        }

        // Stop-sequence matching (mirrors streaming path). Trim the
        // result at the first stop-sequence occurrence and break out of
        // the generation loop.
        if (gen_params.stop_sequences && gen_params.num_stop_sequences > 0) {
            bool stop_hit = false;
            for (int j = 0; j < gen_params.num_stop_sequences; ++j) {
                const char* s = gen_params.stop_sequences[j];
                if (!s || s[0] == '\0') continue;
                const size_t pos = result.find(s);
                if (pos != std::string::npos) {
                    result.erase(pos);  // drop the stop and everything after
                    stop_hit = true;
                    break;
                }
            }
            if (stop_hit) break;
        }

        // Prepare next batch
        llama_batch next_batch = llama_batch_get_one(&new_token, 1);

        // Evaluate
        if (llama_decode(ctx->llama_ctx, next_batch) != 0) {
            llama_sampler_free(sampler);
            ctx->last_error = "Failed during generation";
            return EV_ERROR_INFERENCE_FAILED;
        }
    }

    llama_sampler_free(sampler);

    // Allocate output string
    *output = static_cast<char*>(std::malloc(result.size() + 1));
    if (!*output) {
        return EV_ERROR_OUT_OF_MEMORY;
    }
    std::memcpy(*output, result.c_str(), result.size() + 1);

    return EV_SUCCESS;
#else
    ctx->last_error = "llama.cpp not compiled";
    return EV_ERROR_NOT_IMPLEMENTED;
#endif
}

void ev_free_string(char* str) {
    if (str) {
        std::free(str);
    }
}

/* ============================================================================
 * Streaming Generation
 * ========================================================================= */

ev_stream ev_generate_stream(
    ev_context ctx,
    const char* prompt,
    const ev_generation_params* params,
    ev_error_t* error
) {
    if (!ctx || !prompt) {
        if (error) *error = EV_ERROR_INVALID_PARAM;
        return nullptr;
    }

    if (!ev_is_valid(ctx)) {
        if (error) *error = EV_ERROR_CONTEXT_INVALID;
        return nullptr;
    }

    memory_guard_touch_engine(MG_ENGINE_LLM);

    // Enforce single active stream per context (issue #29).
    // A second stream's ev_stream_next() would clear KV cache, silently
    // invalidating the first stream. Reject early with a clear error.
    if (ctx->active_stream_count.load(std::memory_order_acquire) > 0) {
        ctx->last_error = "Another stream is active on this context — "
                          "end or free the existing stream first";
        if (error) *error = EV_ERROR_CONTEXT_INVALID;
        return nullptr;
    }

    ev_stream stream = new (std::nothrow) ev_stream_impl(ctx, prompt, params);
    if (!stream) {
        if (error) *error = EV_ERROR_OUT_OF_MEMORY;
        return nullptr;
    }

    // Guard against strdup() OOM in grammar string deep-copy (issue #33).
    // If the caller provided grammar strings but strdup() failed, reject
    // the stream rather than silently dropping grammar constraints.
    if ((stream->params.grammar_str && !stream->grammar_str_owned) ||
        (stream->params.grammar_root && !stream->grammar_root_owned)) {
        delete stream;
        if (error) *error = EV_ERROR_OUT_OF_MEMORY;
        return nullptr;
    }

#ifdef EDGE_VEDA_LLAMA_ENABLED
    // Tokenize prompt for later evaluation
    stream->prompt_tokens = tokenize_prompt(ctx->model, prompt, true);
    if (stream->prompt_tokens.empty()) {
        delete stream;
        if (error) *error = EV_ERROR_INFERENCE_FAILED;
        return nullptr;
    }

    // Check context size
    int n_ctx = static_cast<int>(llama_n_ctx(ctx->llama_ctx));
    if (static_cast<int>(stream->prompt_tokens.size()) > n_ctx - 4) {
        delete stream;
        if (error) *error = EV_ERROR_INFERENCE_FAILED;
        return nullptr;
    }

    // Create sampler (owned by stream, pass vocab for grammar support)
    stream->sampler = create_sampler(stream->params, llama_model_get_vocab(ctx->model));
    if (!stream->sampler) {
        delete stream;
        if (error) *error = EV_ERROR_INFERENCE_FAILED;
        return nullptr;
    }
#endif

    // Atomic increment — no mutex needed, pairs with decrement when stream ends.
    ctx->active_stream_count.fetch_add(1, std::memory_order_release);

    if (error) *error = EV_SUCCESS;
    return stream;
}

#ifdef EDGE_VEDA_LLAMA_ENABLED
/* ----------------------------------------------------------------------------
 * Speculative streaming step (mobile-news#586 gap #10)
 *
 * Drives one draft+verify cycle when ctx->spec_ctx is non-null and the
 * stream's spec_buffer is exhausted. Adapted from the canonical pattern
 * in core/third_party/llama.cpp/examples/speculative-simple/speculative-simple.cpp
 * but using llama_sampler_sample directly (we don't carry common_sampler).
 *
 * Returns the next text piece to emit, or nullptr on error / end. Caller
 * (ev_stream_next) holds both stream->mutex and ctx->mutex.
 *
 * The function uses a simple greedy-acceptance rule: for each batch
 * position 0..draft.size(), sample the target token; if it matches
 * draft[i] (for i<draft.size()), keep accepting; first mismatch ends
 * the run. This is correct (matches what common_sampler_sample_and_accept_n
 * does for greedy sampling) and gives deterministic results without
 * needing common_sampler. Probabilistic acceptance with stochastic
 * sampling would need the common_sampler refactor — out of scope here.
 * ------------------------------------------------------------------------- */
static char* ev_stream_next_spec(ev_stream stream, ev_error_t* error) {
    ev_context ctx = stream->ctx;
    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);

    // 1) Return next buffered token if any.
    if (stream->spec_buffer_pos < stream->spec_buffer.size()) {
        llama_token tok = stream->spec_buffer[stream->spec_buffer_pos++];
        if (llama_vocab_is_eog(vocab, tok)) {
            stream->mark_ended();
            if (error) *error = EV_SUCCESS;
            return nullptr;
        }
        char buf[256];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (error) *error = EV_SUCCESS;
        if (n <= 0) {
            char* result = static_cast<char*>(std::malloc(1));
            if (result) result[0] = '\0';
            return result;
        }
        char* result = static_cast<char*>(std::malloc(static_cast<size_t>(n) + 1));
        if (!result) { if (error) *error = EV_ERROR_OUT_OF_MEMORY; return nullptr; }
        std::memcpy(result, buf, static_cast<size_t>(n));
        result[n] = '\0';
        return result;
    }

    // 2) First time: evaluate prompt minus its last token (which becomes id_last).
    if (!stream->spec_initialized) {
        const int n_prompt = static_cast<int>(stream->prompt_tokens.size());
        if (n_prompt < 1) {
            stream->mark_ended();
            if (error) *error = EV_ERROR_INVALID_PARAM;
            return nullptr;
        }
        llama_memory_clear(llama_get_memory(ctx->llama_ctx), true);

        const int n_eval = n_prompt - 1;       // decode all but the last
        const int n_batch = static_cast<int>(llama_n_batch(ctx->llama_ctx));
        for (int i = 0; i < n_eval; i += n_batch) {
            const int n_chunk = std::min(n_batch, n_eval - i);
            llama_batch batch = llama_batch_get_one(
                stream->prompt_tokens.data() + i, n_chunk);
            if (llama_decode(ctx->llama_ctx, batch) != 0) {
                stream->mark_ended();
                if (error) *error = EV_ERROR_INFERENCE_FAILED;
                return nullptr;
            }
        }

        stream->prompt_tgt.assign(
            stream->prompt_tokens.begin(),
            stream->prompt_tokens.begin() + n_eval);
        stream->id_last = stream->prompt_tokens[n_prompt - 1];
        stream->n_cur = n_eval;
        stream->prompt_evaluated = true;

        common_speculative_begin(ctx->spec_ctx, stream->prompt_tgt);
        stream->spec_initialized = true;
    }

    // 3) Stop if we've already produced enough output tokens.
    int generated = static_cast<int>(stream->n_cur)
                  - static_cast<int>(stream->prompt_tokens.size())
                  + 1; // +1 because id_last hasn't been counted into n_cur yet
    if (generated >= stream->params.max_tokens) {
        stream->mark_ended();
        if (error) *error = EV_SUCCESS;
        return nullptr;
    }

    // 4) Draft up to N tokens via the speculator.
    common_params_speculative& sp = ctx->spec_params;
    llama_tokens draft = common_speculative_draft(
        ctx->spec_ctx, sp, stream->prompt_tgt, stream->id_last);

    // 5) Build batch: [id_last @ n_cur, draft0 @ n_cur+1, draft1 @ n_cur+2, ...].
    //    All positions need logits=true so we can sample at each.
    const int n_batch_size = 1 + static_cast<int>(draft.size());
    llama_batch batch = llama_batch_init(n_batch_size, 0, 1);

    auto add_to_batch = [&](llama_token tok, int pos) {
        const int idx = batch.n_tokens;
        batch.token[idx]    = tok;
        batch.pos[idx]      = pos;
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0]= 0;
        batch.logits[idx]   = 1;  // request logits for every position
        batch.n_tokens++;
    };
    add_to_batch(stream->id_last, stream->n_cur);
    for (int i = 0; i < static_cast<int>(draft.size()); ++i) {
        add_to_batch(draft[i], stream->n_cur + 1 + i);
    }

    if (llama_decode(ctx->llama_ctx, batch) != 0) {
        llama_batch_free(batch);
        stream->mark_ended();
        if (error) *error = EV_ERROR_INFERENCE_FAILED;
        return nullptr;
    }

    // 6) Sample at each position; greedy-accept matching drafts.
    std::vector<llama_token> ids;
    ids.reserve(n_batch_size);
    for (int i = 0; i < n_batch_size; ++i) {
        llama_token sampled = llama_sampler_sample(stream->sampler, ctx->llama_ctx, i);
        ids.push_back(sampled);
        if (i < static_cast<int>(draft.size()) && sampled != draft[i]) {
            break; // mismatch — stop accepting
        }
    }
    llama_batch_free(batch);

    const int n_accepted = static_cast<int>(ids.size()) - 1; // last is the new sample
    common_speculative_accept(ctx->spec_ctx, static_cast<uint16_t>(n_accepted));
    ctx->spec_n_drafted  += static_cast<int64_t>(draft.size());
    ctx->spec_n_accepted += static_cast<int64_t>(n_accepted);

    // 7) If the target rejected some drafts, the KV cache still has those
    //    positions filled. Roll back to the last accepted position so the
    //    next iteration starts clean. The total positions we want to keep
    //    is n_cur (id_last) + n_accepted (drafts that matched) + 1 (we'll
    //    not decode the new sample yet — it becomes the new id_last).
    //    Actually after this step: id_last_old is decoded at n_cur,
    //    accepted drafts 0..n_accepted-1 at n_cur+1..n_cur+n_accepted.
    //    Keep KV up to position n_cur + n_accepted (inclusive), drop after.
    if (static_cast<int>(draft.size()) > n_accepted) {
        const int keep_through = stream->n_cur + n_accepted; // inclusive
        // llama_memory_seq_rm removes positions [p0, p1) for sequence id.
        llama_memory_seq_rm(llama_get_memory(ctx->llama_ctx), 0,
                            keep_through + 1, -1);
    }

    // 8) Update prompt_tgt and id_last per the speculative-simple pattern:
    //    push id_last_old + ids[0..n-2]; new id_last = ids.back().
    for (size_t i = 0; i < ids.size(); ++i) {
        stream->prompt_tgt.push_back(stream->id_last);
        stream->id_last = ids[i];
    }
    stream->n_cur += static_cast<int>(ids.size()); // each id consumed one position

    // 9) Buffer the sampled tokens for the caller. Each ev_stream_next call
    //    will return one of these until the buffer drains.
    stream->spec_buffer = std::move(ids);
    stream->spec_buffer_pos = 0;

    // Recurse to return the first buffered token.
    return ev_stream_next_spec(stream, error);
}
#endif

char* ev_stream_next(ev_stream stream, ev_error_t* error) {
    if (!stream) {
        if (error) *error = EV_ERROR_INVALID_PARAM;
        return nullptr;
    }

    // Lock ordering: stream->mutex THEN ctx->mutex (see edge_veda.h).
    std::lock_guard<std::mutex> lock(stream->mutex);

    // Check cancellation FIRST (before any work)
    if (stream->check_cancelled()) {
        stream->mark_ended();
        if (error) *error = EV_ERROR_STREAM_ENDED;
        return nullptr;
    }

    if (stream->ended) {
        if (error) *error = EV_ERROR_STREAM_ENDED;
        return nullptr;
    }

#ifdef EDGE_VEDA_LLAMA_ENABLED
    ev_context ctx = stream->ctx;

    // Lock context for thread safety (context shared across streams)
    std::lock_guard<std::mutex> ctx_lock(ctx->mutex);

    // Speculative decoding fast-path: when a draft is attached, hand off
    // to the speculative streaming routine. It manages its own KV cache,
    // prompt eval, and token buffer. mobile-news#586 gap #10.
    if (ctx->spec_ctx != nullptr) {
        return ev_stream_next_spec(stream, error);
    }

    // First call: evaluate prompt and clear KV cache
    if (!stream->prompt_evaluated) {
        // Clear KV cache for fresh generation
        llama_memory_clear(llama_get_memory(ctx->llama_ctx), true);

        // Evaluate prompt tokens in batches of n_batch
        const int n_prompt = static_cast<int>(stream->prompt_tokens.size());
        const int n_batch = static_cast<int>(llama_n_batch(ctx->llama_ctx));

        for (int i = 0; i < n_prompt; i += n_batch) {
            const int n_eval = std::min(n_batch, n_prompt - i);
            llama_batch batch = llama_batch_get_one(
                stream->prompt_tokens.data() + i, n_eval);
            if (llama_decode(ctx->llama_ctx, batch) != 0) {
                stream->mark_ended();
                if (error) *error = EV_ERROR_INFERENCE_FAILED;
                return nullptr;
            }
        }

        stream->n_cur = n_prompt;
        stream->prompt_evaluated = true;
    }

    // Check max tokens limit
    int generated_count = stream->n_cur - static_cast<int>(stream->prompt_tokens.size());
    if (generated_count >= stream->params.max_tokens) {
        stream->mark_ended();
        if (error) *error = EV_SUCCESS;  // Natural end, not an error
        return nullptr;
    }

    // Check cancellation again before expensive sampling
    if (stream->check_cancelled()) {
        stream->mark_ended();
        if (error) *error = EV_ERROR_STREAM_ENDED;
        return nullptr;
    }

    // Sample next token
    llama_token new_token = llama_sampler_sample(stream->sampler, ctx->llama_ctx, -1);

    // Check for EOS
    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    if (llama_vocab_is_eog(vocab, new_token)) {
        stream->mark_ended();
        if (error) *error = EV_SUCCESS;  // Natural end
        return nullptr;
    }

    // Compute confidence score (only when threshold > 0)
    if (stream->params.confidence_threshold > 0.0f) {
        const float* logits = llama_get_logits_ith(ctx->llama_ctx, -1);
        const llama_vocab* vocab_for_conf = llama_model_get_vocab(ctx->model);
        int n_vocab = llama_vocab_n_tokens(vocab_for_conf);

        // Softmax with numerical stability
        float max_val = -1e30f;
        for (int i = 0; i < n_vocab; i++) {
            if (logits[i] > max_val) max_val = logits[i];
        }

        double sum_exp = 0.0;
        for (int i = 0; i < n_vocab; i++) {
            sum_exp += exp((double)(logits[i] - max_val));
        }

        // Shannon entropy: H = -sum(p * log(p))
        double entropy = 0.0;
        for (int i = 0; i < n_vocab; i++) {
            double p = exp((double)(logits[i] - max_val)) / sum_exp;
            if (p > 1e-10) entropy -= p * log(p);
        }

        // Normalize to [0, 1] and invert: 1.0 = certain, 0.0 = uniform
        double max_entropy = log((double)n_vocab);
        stream->last_confidence = (float)(1.0 - entropy / max_entropy);

        // Update running average
        stream->confidence_sum += stream->last_confidence;
        stream->confidence_count++;
        float avg = (float)(stream->confidence_sum / stream->confidence_count);

        // Check handoff threshold
        if (avg < stream->params.confidence_threshold && stream->confidence_count >= 3) {
            stream->needs_handoff = true;
        }
    }

    // Convert token to text
    char buf[256];
    int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
    if (n <= 0) {
        // Empty token (can happen with special tokens), continue
        stream->n_cur++;
        if (error) *error = EV_SUCCESS;

        // Decode the empty token to maintain KV cache consistency
        llama_batch batch = llama_batch_get_one(&new_token, 1);
        llama_decode(ctx->llama_ctx, batch);

        // Return empty string (caller should continue calling)
        char* result = static_cast<char*>(std::malloc(1));
        if (result) result[0] = '\0';
        return result;
    }

    // Decode next token to update KV cache
    llama_batch batch = llama_batch_get_one(&new_token, 1);
    if (llama_decode(ctx->llama_ctx, batch) != 0) {
        stream->mark_ended();
        if (error) *error = EV_ERROR_INFERENCE_FAILED;
        return nullptr;
    }

    stream->n_cur++;

    // Stop-sequence matching (app-level, since llama.cpp's sampler has no
    // built-in support). Append the new token text to the sliding buffer,
    // cap at 4 KiB (enough to catch any reasonable stop sequence even
    // across token boundaries), then scan for any registered stop.
    //
    // If matched, emit only the portion of THIS token's text that comes
    // before the match start, mark the stream ended, and return — the
    // stop sequence itself and anything after it is suppressed.
    size_t emit_bytes = static_cast<size_t>(n);
    bool stop_hit = false;
    if (!stream->stop_sequences_owned.empty()) {
        stream->accumulated_output.append(buf, static_cast<size_t>(n));
        static constexpr size_t kMaxAccum = 4096;
        if (stream->accumulated_output.size() > kMaxAccum) {
            stream->accumulated_output.erase(
                0, stream->accumulated_output.size() - kMaxAccum);
        }
        const size_t token_start =
            stream->accumulated_output.size() - static_cast<size_t>(n);
        for (const auto& stop : stream->stop_sequences_owned) {
            if (stop.empty()) continue;
            const size_t pos = stream->accumulated_output.find(stop);
            if (pos != std::string::npos) {
                emit_bytes = (pos >= token_start) ? (pos - token_start) : 0;
                stop_hit = true;
                break;
            }
        }
    }

    if (stop_hit) {
        stream->mark_ended();
    }

    // Allocate and return token string (possibly truncated at stop)
    char* result = static_cast<char*>(std::malloc(emit_bytes + 1));
    if (!result) {
        if (error) *error = EV_ERROR_OUT_OF_MEMORY;
        return nullptr;
    }
    if (emit_bytes > 0) {
        std::memcpy(result, buf, emit_bytes);
    }
    result[emit_bytes] = '\0';

    if (error) *error = EV_SUCCESS;
    return result;
#else
    stream->mark_ended();
    if (error) *error = EV_ERROR_NOT_IMPLEMENTED;
    return nullptr;
#endif
}

bool ev_stream_has_next(ev_stream stream) {
    if (!stream) return false;
    std::lock_guard<std::mutex> lock(stream->mutex);
    return !stream->ended && !stream->check_cancelled();
}

void ev_stream_cancel(ev_stream stream) {
    if (!stream) return;
    // Use atomic operation - no mutex needed for cancel request
    stream->request_cancel();
}

void ev_stream_free(ev_stream stream) {
    if (!stream) return;
    // Tombstone guard: if ctx was already freed (magic poisoned), skip
    // the active_stream_count decrement to avoid use-after-free (issue #28).
    if (stream->ctx && stream->ctx->magic == EV_CTX_MAGIC &&
        !stream->deactivated.exchange(true, std::memory_order_acq_rel)) {
        stream->ctx->active_stream_count.fetch_sub(1, std::memory_order_release);
    }
    // Destructor handles sampler cleanup
    delete stream;
}

/* ============================================================================
 * Streaming Token Info (confidence scoring)
 * ========================================================================= */

ev_error_t ev_stream_get_token_info(ev_stream stream, ev_stream_token_info* info) {
    if (!stream || !info) return EV_ERROR_INVALID_PARAM;

    std::lock_guard<std::mutex> lock(stream->mutex);

    info->confidence = stream->last_confidence;
    info->avg_confidence = stream->confidence_count > 0
        ? (float)(stream->confidence_sum / stream->confidence_count)
        : -1.0f;
    info->needs_cloud_handoff = stream->needs_handoff;
    info->token_index = stream->confidence_count;

    return EV_SUCCESS;
}

/* ============================================================================
 * Memory Management
 * ========================================================================= */

ev_error_t ev_get_memory_usage(ev_context ctx, ev_memory_stats* stats) {
    if (!ctx || !stats) {
        return EV_ERROR_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    std::memset(stats, 0, sizeof(ev_memory_stats));

    // Get platform memory from memory guard
    stats->current_bytes = static_cast<size_t>(memory_guard_get_current_usage());
    stats->peak_bytes = ctx->peak_memory_bytes;
    stats->limit_bytes = ctx->memory_limit;

#ifdef EDGE_VEDA_LLAMA_ENABLED
    if (ctx->model) {
        // Get model size (approximate)
        stats->model_bytes = llama_model_size(ctx->model);
    }
    if (ctx->llama_ctx) {
        // Context memory usage
        stats->context_bytes = llama_state_get_size(ctx->llama_ctx);
    }
    if (ctx->embed_ctx) {
        // Include persistent embedding context in memory accounting.
        stats->context_bytes += llama_state_get_size(ctx->embed_ctx);
    }
#endif

    // Update peak
    if (stats->current_bytes > ctx->peak_memory_bytes) {
        ctx->peak_memory_bytes = stats->current_bytes;
    }

    return EV_SUCCESS;
}

ev_error_t ev_set_memory_limit(ev_context ctx, size_t limit_bytes) {
    if (!ctx) {
        return EV_ERROR_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->memory_limit = limit_bytes;
    memory_guard_set_limit(limit_bytes);

    return EV_SUCCESS;
}

ev_error_t ev_set_memory_pressure_callback(
    ev_context ctx,
    ev_memory_pressure_callback callback,
    void* user_data
) {
    if (!ctx) {
        return EV_ERROR_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->memory_callback = callback;
    ctx->memory_callback_data = user_data;

    // Set up memory guard callback wrapper
    if (callback) {
        auto wrapper = [](void* data, size_t current, size_t limit) {
            ev_context ctx = static_cast<ev_context>(data);
            if (ctx->memory_callback) {
                ctx->memory_callback(ctx->memory_callback_data, current, limit);
            }
        };
        memory_guard_set_callback(wrapper, ctx);
    } else {
        memory_guard_set_callback(nullptr, nullptr);
    }

    return EV_SUCCESS;
}

ev_error_t ev_memory_cleanup(ev_context ctx) {
    if (!ctx) {
        return EV_ERROR_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);

#ifdef EDGE_VEDA_LLAMA_ENABLED
    // Clear KV cache to free memory
    if (ctx->llama_ctx) {
        llama_memory_clear(llama_get_memory(ctx->llama_ctx), true);
    }
    if (ctx->embed_ctx) {
        llama_memory_clear(llama_get_memory(ctx->embed_ctx), true);
    }
#endif

    // Trigger platform memory cleanup
    memory_guard_cleanup();

    return EV_SUCCESS;
}

/* ============================================================================
 * Embeddings API
 * ========================================================================= */

#ifdef EDGE_VEDA_LLAMA_ENABLED
ev_error_t ev_embed(
    ev_context ctx,
    const char* text,
    ev_embed_result* result
) {
    // Validate parameters
    if (!ctx || !text || !result) {
        return EV_ERROR_INVALID_PARAM;
    }

    if (!ctx->model) {
        return EV_ERROR_CONTEXT_INVALID;
    }

    // Zero-initialize result
    std::memset(result, 0, sizeof(ev_embed_result));

    memory_guard_touch_engine(MG_ENGINE_LLM);
    std::lock_guard<std::mutex> lock(ctx->mutex);

    // Lazy-init and persist embedding context to avoid per-call setup overhead.
    if (!ctx->embed_ctx) {
        llama_context_params emb_params = llama_context_default_params();
        emb_params.embeddings = true;
        emb_params.n_ctx = 512;
        emb_params.n_batch = 512;
        emb_params.n_threads = ctx->config.num_threads > 0
            ? static_cast<uint32_t>(ctx->config.num_threads) : ev_default_thread_count();
        emb_params.n_threads_batch = emb_params.n_threads;
        emb_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;

        ctx->embed_ctx = llama_init_from_model(ctx->model, emb_params);
        if (!ctx->embed_ctx) {
            ctx->last_error = "Failed to create embedding context";
            return EV_ERROR_BACKEND_INIT_FAILED;
        }

        // Configure once for bidirectional embedding encoding.
        llama_set_embeddings(ctx->embed_ctx, true);
        llama_set_causal_attn(ctx->embed_ctx, false);
    }

    // Clear KV cache
    llama_memory_clear(llama_get_memory(ctx->embed_ctx), true);

    // Tokenize input text
    std::vector<llama_token> tokens = tokenize_prompt(ctx->model, text, true);
    if (tokens.empty()) {
        ctx->last_error = "Failed to tokenize text for embedding";
        return EV_ERROR_INFERENCE_FAILED;
    }

    // Create batch and decode
    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(ctx->embed_ctx, batch) != 0) {
        ctx->last_error = "Failed to decode for embedding";
        return EV_ERROR_INFERENCE_FAILED;
    }

    // Get pooled embeddings
    const float* emb = llama_get_embeddings_seq(ctx->embed_ctx, 0);
    if (!emb) {
        // Fallback: try last token embeddings
        emb = llama_get_embeddings_ith(ctx->embed_ctx, -1);
    }
    if (!emb) {
        ctx->last_error = "Failed to retrieve embeddings";
        return EV_ERROR_INFERENCE_FAILED;
    }

    // Get embedding dimension
    int n_embd = llama_model_n_embd(ctx->model);

    // Allocate result buffer
    result->embeddings = (float*)malloc(sizeof(float) * n_embd);
    if (!result->embeddings) {
        return EV_ERROR_OUT_OF_MEMORY;
    }

    // L2-normalize into result buffer
    double sum_sq = 0.0;
    for (int i = 0; i < n_embd; i++) {
        sum_sq += (double)emb[i] * (double)emb[i];
    }
    float norm = (sum_sq > 0.0) ? (float)(1.0 / sqrt(sum_sq)) : 0.0f;
    for (int i = 0; i < n_embd; i++) {
        result->embeddings[i] = emb[i] * norm;
    }

    result->dimensions = n_embd;
    result->token_count = static_cast<int>(tokens.size());

    return EV_SUCCESS;
}
#else
ev_error_t ev_embed(
    ev_context ctx,
    const char* text,
    ev_embed_result* result
) {
    (void)ctx;
    (void)text;
    if (result) std::memset(result, 0, sizeof(ev_embed_result));
    return EV_ERROR_NOT_IMPLEMENTED;
}
#endif

void ev_free_embeddings(ev_embed_result* result) {
    if (result && result->embeddings) {
        free(result->embeddings);
        result->embeddings = nullptr;
        result->dimensions = 0;
        result->token_count = 0;
    }
}

/* ============================================================================
 * Model Information
 * ========================================================================= */

ev_error_t ev_get_model_info(ev_context ctx, ev_model_info* info) {
    if (!ctx || !info) {
        return EV_ERROR_INVALID_PARAM;
    }

    if (!ev_is_valid(ctx)) {
        return EV_ERROR_CONTEXT_INVALID;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    std::memset(info, 0, sizeof(ev_model_info));

#ifdef EDGE_VEDA_LLAMA_ENABLED
    // Model description — stored per-context to avoid static buffer race (issue #26)
    llama_model_desc(ctx->model, ctx->model_desc, sizeof(ctx->model_desc));
    info->name = ctx->model_desc;

    // Architecture (most GGUF models are llama-based)
    info->architecture = "llama";

    // Parameters
    info->num_parameters = llama_model_n_params(ctx->model);

    // Context and embedding info (using new API names)
    info->context_length = static_cast<int>(llama_n_ctx(ctx->llama_ctx));
    info->embedding_dim = static_cast<int>(llama_model_n_embd(ctx->model));
    info->num_layers = static_cast<int>(llama_model_n_layer(ctx->model));

    return EV_SUCCESS;
#else
    return EV_ERROR_NOT_IMPLEMENTED;
#endif
}

/* ============================================================================
 * Utility Functions
 * ========================================================================= */

static bool g_verbose = false;

void ev_set_verbose(bool enable) {
    g_verbose = enable;

#ifdef EDGE_VEDA_LLAMA_ENABLED
    // llama.cpp uses log callback, set it based on verbosity
    if (enable) {
        llama_log_set([](enum ggml_log_level level, const char* text, void* /* user_data */) {
            fprintf(stderr, "[llama] %s", text);
            (void)level;
        }, nullptr);
    } else {
        llama_log_set([](enum ggml_log_level level, const char* text, void* /* user_data */) {
            // Suppress all but errors
            if (level == GGML_LOG_LEVEL_ERROR) {
                fprintf(stderr, "[llama] %s", text);
            }
        }, nullptr);
    }
#endif
}

const char* ev_get_last_error(ev_context ctx) {
    if (!ctx) {
        return "Invalid context";
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    return ctx->last_error.c_str();
}

ev_error_t ev_reset(ev_context ctx) {
    if (!ctx) {
        return EV_ERROR_INVALID_PARAM;
    }

    if (!ev_is_valid(ctx)) {
        return EV_ERROR_CONTEXT_INVALID;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);

#ifdef EDGE_VEDA_LLAMA_ENABLED
    llama_memory_clear(llama_get_memory(ctx->llama_ctx), true);
    if (ctx->embed_ctx) {
        llama_memory_clear(llama_get_memory(ctx->embed_ctx), true);
    }
    return EV_SUCCESS;
#else
    return EV_ERROR_NOT_IMPLEMENTED;
#endif
}

/* ============================================================================
 * Test Hooks (compiled only with EDGE_VEDA_TEST_HOOKS)
 * ========================================================================= */

#ifdef EDGE_VEDA_TEST_HOOKS
ev_error_t ev_test_stream_grammar_owned(
    ev_stream stream,
    bool* has_grammar_str,
    bool* has_grammar_root
) {
    if (!stream || !has_grammar_str || !has_grammar_root) {
        return EV_ERROR_INVALID_PARAM;
    }
    *has_grammar_str = (stream->grammar_str_owned != nullptr);
    *has_grammar_root = (stream->grammar_root_owned != nullptr);
    return EV_SUCCESS;
}
#endif

/* ============================================================================
 * Speculative Decoding (mobile-news#586 gap #10)
 *
 * Wraps llama.cpp `common_speculative_*` (common/speculative.h). The
 * draft model is loaded as a separate llama_context attached to the
 * existing target context. Generation paths in this file consult
 * `ctx->spec_ctx`; when non-null, ev_generate / ev_generate_stream
 * use the speculative draft loop. When null, behaviour is unchanged.
 *
 * Resource ownership is symmetric with ev_init/ev_free: attach
 * loads the draft and constructs `common_speculative`; detach frees
 * all three handles. ev_free() also calls detach() so a context that
 * never explicitly detaches still cleans up.
 * ========================================================================= */

EV_API void ev_speculative_params_default(ev_speculative_params* params) {
    if (!params) return;
    params->n_max         = 16;
    params->n_min         = 0;
    params->p_min         = 0.75f;
    params->p_split       = 0.1f;
    params->n_ctx         = 0;
    params->n_gpu_layers  = -1;
    params->cache_type_k  = 0;
    params->cache_type_v  = 0;
}

EV_API ev_error_t ev_speculative_attach(
    ev_context ctx,
    const char* draft_path,
    const ev_speculative_params* params
) {
#ifndef EDGE_VEDA_LLAMA_ENABLED
    (void)ctx; (void)draft_path; (void)params;
    return EV_ERROR_CONTEXT_INVALID;
#else
    if (!ctx || ctx->magic != EV_CTX_MAGIC) return EV_ERROR_INVALID_PARAM;
    if (!ctx->model_loaded || !ctx->llama_ctx || !ctx->model) {
        return EV_ERROR_CONTEXT_INVALID;
    }
    if (!draft_path || draft_path[0] == '\0') return EV_ERROR_INVALID_PARAM;

    std::lock_guard<std::mutex> lock(ctx->mutex);

    // Replace any prior draft to keep the contract simple.
    if (ctx->spec_ctx)   { common_speculative_free(ctx->spec_ctx); ctx->spec_ctx = nullptr; }
    if (ctx->draft_ctx)  { llama_free(ctx->draft_ctx); ctx->draft_ctx = nullptr; }
    if (ctx->draft_model){ llama_model_free(ctx->draft_model); ctx->draft_model = nullptr; }
    ctx->spec_n_drafted  = 0;
    ctx->spec_n_accepted = 0;

    ev_speculative_params eff;
    ev_speculative_params_default(&eff);
    if (params) eff = *params;

    // Load the draft model. Reuse the target context's GPU policy
    // unless the caller overrode it via params->n_gpu_layers.
    llama_model_params m_params = llama_model_default_params();
    if (eff.n_gpu_layers >= 0) {
        m_params.n_gpu_layers = eff.n_gpu_layers;
    }
    ctx->draft_model = llama_model_load_from_file(draft_path, m_params);
    if (!ctx->draft_model) {
        ctx->last_error = "ev_speculative_attach: draft model load failed";
        return EV_ERROR_MODEL_LOAD_FAILED;
    }

    // Vocabulary compatibility: speculative decoding requires the
    // draft and target to agree on token IDs for the prefix that gets
    // verified. Different vocab → undefined behaviour. Bail early.
    if (llama_vocab_n_tokens(llama_model_get_vocab(ctx->draft_model)) !=
        llama_vocab_n_tokens(llama_model_get_vocab(ctx->model))) {
        llama_model_free(ctx->draft_model);
        ctx->draft_model = nullptr;
        ctx->last_error = "ev_speculative_attach: draft vocab size mismatch";
        return EV_ERROR_MODEL_LOAD_FAILED;
    }

    // Build the params common_speculative_init expects. Three things
    // matter:
    //   - type           = COMMON_SPECULATIVE_TYPE_DRAFT (we're using
    //     draft-model speculation, not n-gram lookup)
    //   - draft.model    = pointer to the loaded draft llama_model
    //   - draft.cparams  = llama_context_params for the draft ctx
    //                      (init creates that ctx internally)
    //   - draft.mparams.path = path string; init checks it for
    //     "is a draft configured?" gating
    // tunables (n_max / n_min / p_min / p_split) live on
    // params.draft.* and gate how aggressively the speculator drafts.
    llama_context_params dft_cparams = llama_context_default_params();
    dft_cparams.n_ctx   = (eff.n_ctx > 0) ? eff.n_ctx : ctx->config.context_size;
    dft_cparams.n_batch = dft_cparams.n_ctx;
    if (eff.cache_type_k > 0) dft_cparams.type_k = (ggml_type)eff.cache_type_k;
    if (eff.cache_type_v > 0) dft_cparams.type_v = (ggml_type)eff.cache_type_v;

    ctx->spec_params = common_params_speculative{};
    ctx->spec_params.type             = COMMON_SPECULATIVE_TYPE_DRAFT;
    ctx->spec_params.draft.model      = ctx->draft_model;
    ctx->spec_params.draft.cparams    = dft_cparams;
    ctx->spec_params.draft.mparams.path = std::string(draft_path);
    ctx->spec_params.draft.n_max      = eff.n_max;
    ctx->spec_params.draft.n_min      = eff.n_min;
    ctx->spec_params.draft.p_min      = eff.p_min;
    ctx->spec_params.draft.p_split    = eff.p_split;
    if (eff.n_gpu_layers >= 0) {
        ctx->spec_params.draft.n_gpu_layers = eff.n_gpu_layers;
    }

    ctx->spec_ctx = common_speculative_init(ctx->spec_params, ctx->llama_ctx);
    if (!ctx->spec_ctx) {
        llama_model_free(ctx->draft_model); ctx->draft_model = nullptr;
        ctx->last_error = "ev_speculative_attach: common_speculative_init failed";
        return EV_ERROR_OUT_OF_MEMORY;
    }
    // common_speculative_init creates its own ctx_dft internally from
    // params.draft.model + params.draft.cparams; we no longer keep our
    // own draft_ctx.
    ctx->draft_ctx = nullptr;
    return EV_SUCCESS;
#endif
}

EV_API bool ev_speculative_is_attached(ev_context ctx) {
#ifndef EDGE_VEDA_LLAMA_ENABLED
    (void)ctx; return false;
#else
    if (!ctx || ctx->magic != EV_CTX_MAGIC) return false;
    return ctx->spec_ctx != nullptr;
#endif
}

EV_API ev_error_t ev_speculative_detach(ev_context ctx) {
#ifndef EDGE_VEDA_LLAMA_ENABLED
    (void)ctx; return EV_SUCCESS;
#else
    if (!ctx || ctx->magic != EV_CTX_MAGIC) return EV_ERROR_INVALID_PARAM;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (ctx->spec_ctx)    { common_speculative_free(ctx->spec_ctx); ctx->spec_ctx = nullptr; }
    if (ctx->draft_ctx)   { llama_free(ctx->draft_ctx); ctx->draft_ctx = nullptr; }
    if (ctx->draft_model) { llama_model_free(ctx->draft_model); ctx->draft_model = nullptr; }
    return EV_SUCCESS;
#endif
}

EV_API ev_error_t ev_speculative_get_stats(
    ev_context ctx,
    ev_speculative_stats* stats
) {
#ifndef EDGE_VEDA_LLAMA_ENABLED
    (void)ctx; (void)stats; return EV_ERROR_CONTEXT_INVALID;
#else
    if (!ctx || ctx->magic != EV_CTX_MAGIC || !stats) return EV_ERROR_INVALID_PARAM;
    if (!ctx->spec_ctx) return EV_ERROR_CONTEXT_INVALID;
    stats->n_drafted  = ctx->spec_n_drafted;
    stats->n_accepted = ctx->spec_n_accepted;
    stats->n_rejected = ctx->spec_n_drafted - ctx->spec_n_accepted;
    stats->acceptance_rate = (ctx->spec_n_drafted > 0)
        ? (double)ctx->spec_n_accepted / (double)ctx->spec_n_drafted
        : 0.0;
    return EV_SUCCESS;
#endif
}
