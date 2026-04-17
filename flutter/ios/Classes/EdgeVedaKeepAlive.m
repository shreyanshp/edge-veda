// EdgeVedaKeepAlive.m
//
// The Bitcoin News macOS Release link command uses `-Xlinker -dead_strip`.
// `-force_load` on libedge_veda_full.a pulls all .o files in, but
// dead_strip then walks reachability from main() and removes functions
// that nothing references at compile time. Every `ev_*` symbol is only
// called via Dart FFI `dlsym(RTLD_DEFAULT, "ev_X")` at runtime, so the
// linker sees them as unreachable and strips them — `dlsym` then returns
// NULL and EdgeVeda.init fails with
//   "Failed to lookup symbol 'ev_version': symbol not found"
// (Sentry MOBILE-NEWS-87, build 1035).
//
// Fix: this TU references every public FFI symbol. The references are
// compile-time-reachable from the plugin's init path (touchKeepAlive is
// declared __attribute__((used)) AND called from a +load method), so
// dead_strip keeps the .o files containing these symbols in the final
// binary, which keeps them resolvable via dlsym at runtime.
//
// This file intentionally lives at the plugin source level (Classes/)
// rather than inside the static archive — the Runner's link phase is
// what applies -dead_strip, and Classes/*.m is compiled into the
// edge_veda.framework module that Runner directly depends on.

#import <Foundation/Foundation.h>

// Forward declare all public ev_* entry points. We don't include edge_veda.h
// because it lives inside the xcframework Headers dir and Classes/ builds
// in a separate module path. Prototypes are fine — the linker only needs
// symbol names.
extern void ev_version(void);
extern void ev_init(void);
extern void ev_free(void);
extern void ev_is_valid(void);
extern void ev_get_model_info(void);
extern void ev_generate(void);
extern void ev_generate_stream(void);
extern void ev_detect_backend(void);
extern void ev_is_backend_available(void);
extern void ev_embed(void);
extern void ev_free_embeddings(void);
extern void ev_free_string(void);
extern void ev_error_string(void);
extern void ev_get_last_error(void);
extern void ev_set_verbose(void);
extern void ev_get_memory_usage(void);
extern void ev_set_memory_limit(void);
extern void ev_set_memory_pressure_callback(void);
extern void ev_memory_cleanup(void);
extern void ev_stream_next(void);
extern void ev_stream_has_next(void);
extern void ev_stream_cancel(void);
extern void ev_stream_free(void);
extern void ev_stream_get_token_info(void);
extern void ev_vision_init(void);
extern void ev_vision_is_valid(void);
extern void ev_vision_free(void);
extern void ev_vision_describe(void);
extern void ev_vision_describe_stream(void);
extern void ev_vision_get_last_timings(void);
extern void ev_image_init(void);
extern void ev_image_is_valid(void);
extern void ev_image_free(void);
extern void ev_image_generate(void);
extern void ev_image_cancel(void);
extern void ev_image_free_result(void);
extern void ev_image_set_progress_callback(void);
extern void ev_image_config_default(void);
extern void ev_image_gen_params_default(void);

// Volatile sink so the optimiser cannot reason about unreachability and
// constant-fold the call list away before the linker sees it.
static void *volatile kEdgeVedaKeepAliveSink = (void *)0;

__attribute__((used))
static void edge_veda_keep_alive_touch(void) {
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_version;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_init;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_free;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_is_valid;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_get_model_info;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_generate;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_generate_stream;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_detect_backend;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_is_backend_available;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_embed;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_free_embeddings;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_free_string;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_error_string;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_get_last_error;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_set_verbose;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_get_memory_usage;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_set_memory_limit;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_set_memory_pressure_callback;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_memory_cleanup;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_stream_next;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_stream_has_next;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_stream_cancel;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_stream_free;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_stream_get_token_info;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_vision_init;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_vision_is_valid;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_vision_free;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_vision_describe;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_vision_describe_stream;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_vision_get_last_timings;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_image_init;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_image_is_valid;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_image_free;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_image_generate;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_image_cancel;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_image_free_result;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_image_set_progress_callback;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_image_config_default;
  kEdgeVedaKeepAliveSink = (void *)(uintptr_t)&ev_image_gen_params_default;
}

// ObjC +load runs during image load — guaranteed reachability from the
// bundle's launch path, which anchors the keepalive touch function
// (and through it, every ev_* symbol) as reachable for dead_strip.
@interface EdgeVedaKeepAlive : NSObject
@end

@implementation EdgeVedaKeepAlive
+ (void)load {
  edge_veda_keep_alive_touch();
}
@end
