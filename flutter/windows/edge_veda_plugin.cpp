#include "edge_veda_plugin.h"

#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>

#include <memory>

namespace edge_veda {

// static
void EdgeVedaPlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarWindows* registrar) {
  auto channel =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          registrar->messenger(), "com.edgeveda.edge_veda/main",
          &flutter::StandardMethodCodec::GetInstance());

  auto plugin = std::make_unique<EdgeVedaPlugin>();

  channel->SetMethodCallHandler(
      [plugin_pointer = plugin.get()](const auto& call, auto result) {
        plugin_pointer->HandleMethodCall(call, std::move(result));
      });

  registrar->AddPlugin(std::move(plugin));
}

EdgeVedaPlugin::EdgeVedaPlugin() {}

EdgeVedaPlugin::~EdgeVedaPlugin() {}

void EdgeVedaPlugin::HandleMethodCall(
    const flutter::MethodCall<flutter::EncodableValue>& method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  // Until the macOS Swift surface is ported, every call short-circuits
  // with UNAVAILABLE. The host app already gates AI features on
  // PlatformCapabilities.edgeVedaAvailable (false on Windows), so this
  // path should not be hit in normal flow — it's a safety net for any
  // direct EdgeVeda SDK calls that escape the gate.
  result->Error(
      "UNAVAILABLE",
      "edge_veda Windows port is in progress. The plugin DLL builds "
      "and registers, but model lifecycle, chat, vision, Whisper STT, "
      "and audio capture handlers are not yet wired up. Track "
      "https://github.com/bitcoin-portal/mobile-news/issues/574 Phase 4 "
      "for status.");
}

}  // namespace edge_veda
