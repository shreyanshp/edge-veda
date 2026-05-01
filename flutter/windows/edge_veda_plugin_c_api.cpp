#include "include/edge_veda/edge_veda_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "edge_veda_plugin.h"

void EdgeVedaPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  edge_veda::EdgeVedaPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
