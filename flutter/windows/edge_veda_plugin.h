#ifndef FLUTTER_PLUGIN_EDGE_VEDA_PLUGIN_H_
#define FLUTTER_PLUGIN_EDGE_VEDA_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace edge_veda {

// Edge Veda Windows plugin shell.
//
// This first commit registers the `com.edgeveda.edge_veda/main` method
// channel and returns `UNAVAILABLE` for every call so the host Flutter
// app builds and links cleanly on Windows. The full surface — chat
// streaming, vision describe, Whisper STT, model lifecycle, telemetry,
// audio capture EventChannel, thermal state EventChannel — needs to be
// ported from `flutter/macos/Classes/EdgeVedaPlugin.swift` and is
// tracked separately from the issue #574 (mobile-news) Phase 4 PR.
//
// The host app gates AI surfaces on
// `PlatformCapabilities.edgeVedaAvailable`, which currently returns
// false on Windows; flip that flag once the methods listed above
// produce real results on Windows.
class EdgeVedaPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows* registrar);

  EdgeVedaPlugin();
  virtual ~EdgeVedaPlugin();

  // Disallow copy and assign.
  EdgeVedaPlugin(const EdgeVedaPlugin&) = delete;
  EdgeVedaPlugin& operator=(const EdgeVedaPlugin&) = delete;

 private:
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue>& method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace edge_veda

#endif  // FLUTTER_PLUGIN_EDGE_VEDA_PLUGIN_H_
