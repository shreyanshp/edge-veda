# Windows plugin build for edge_veda.
#
# Builds the plugin DLL that registers the `com.edgeveda.edge_veda/main`
# method channel and the matching event channels with the Flutter
# engine. The C++ core is intentionally NOT linked in this initial
# commit — every method short-circuits with UNAVAILABLE — so the host
# Flutter app builds and ships on Windows without waiting for the full
# llama.cpp / whisper.cpp port.
#
# Future GPU strategy on Windows when we wire the core:
#   - Default: Vulkan (covers NVIDIA + AMD + Intel; one binary).
#   - Opt-in:  CUDA (NVIDIA-only, faster for LLMs). Toggle with
#     `-DEDGE_VEDA_ENABLE_CUDA=ON` at configure time.
#   - Skipped: DirectML (slower for LLMs than Vulkan).

cmake_minimum_required(VERSION 3.14)
set(PROJECT_NAME "edge_veda")
project(${PROJECT_NAME} LANGUAGES CXX)

# This value is used when generating builds using this plugin, so it must
# not be changed.
set(PLUGIN_NAME "edge_veda_plugin")

list(APPEND PLUGIN_SOURCES
  "edge_veda_plugin.cpp"
  "include/edge_veda/edge_veda_plugin_c_api.h"
  "edge_veda_plugin_c_api.cpp"
)

add_library(${PLUGIN_NAME} SHARED
  ${PLUGIN_SOURCES}
)

target_compile_features(${PLUGIN_NAME} PRIVATE cxx_std_17)
target_include_directories(${PLUGIN_NAME} INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(${PLUGIN_NAME} PRIVATE flutter flutter_wrapper_plugin)

# List of absolute paths to libraries that should be bundled with the
# plugin in the install step.
set(edge_veda_bundled_libraries
  ""
  PARENT_SCOPE
)
