#ifndef FLUTTER_PLUGIN_EDGE_VEDA_PLUGIN_H_
#define FLUTTER_PLUGIN_EDGE_VEDA_PLUGIN_H_

#include <flutter/event_channel.h>
#include <flutter/event_stream_handler_functions.h>
#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace edge_veda {

// Edge Veda Windows plugin.
//
// Mirrors the Apple platform feature set in flutter/macos/Classes/
// EdgeVedaPlugin.swift via Win32 + WinRT equivalents. The Dart side
// (lib/src/edge_veda_impl.dart, lib/src/whisper_session.dart,
// lib/src/tts_service.dart, lib/src/voice_pipeline.dart, ...) calls
// the same method channels — `com.edgeveda.edge_veda/{telemetry,
// thermal, audio_capture, tts_events}` — and gets the same wire shape
// back regardless of platform. Where a feature is fundamentally
// macOS-specific (photo library insights, calendar insights, detective
// permissions) the Windows handler returns `UNAVAILABLE` and the Dart
// caller treats the feature as off — the host app already gates these
// at higher levels.
class EdgeVedaPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(
      flutter::PluginRegistrarWindows* registrar);

  EdgeVedaPlugin();
  ~EdgeVedaPlugin() override;

  EdgeVedaPlugin(const EdgeVedaPlugin&) = delete;
  EdgeVedaPlugin& operator=(const EdgeVedaPlugin&) = delete;

 private:
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue>& method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>>
          result);

  // Windowing parent — used by share / save-file dialogs to anchor
  // their picker UI to the runner HWND.
  HWND host_window_ = nullptr;
};

}  // namespace edge_veda

#endif  // FLUTTER_PLUGIN_EDGE_VEDA_PLUGIN_H_
