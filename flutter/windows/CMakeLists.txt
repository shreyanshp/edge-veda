# Windows plugin build for edge_veda.
#
# Builds the plugin DLL that registers `com.edgeveda.edge_veda/{telemetry,
# thermal, audio_capture, tts_events}` with the Flutter engine. Compiles
# the C++ core in-tree when the third_party submodules (llama.cpp,
# whisper.cpp, stable-diffusion.cpp) are present alongside it; falls
# back to a no-core build (FFI calls into ev_* will fail at dlsym) when
# they aren't, so the plugin still loads from a pub git checkout where
# submodules weren't initialised.
#
# GPU strategy:
#   - Default: Vulkan (NVIDIA + AMD + Intel; one binary).
#   - Opt-in:  CUDA (NVIDIA only). Configure with
#     `-DEDGE_VEDA_ENABLE_VULKAN=OFF -DEDGE_VEDA_ENABLE_CUDA=ON`.
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

# Win32 + WinRT libs used by the method/event channel handlers.
target_link_libraries(${PLUGIN_NAME} PRIVATE
  "psapi.lib"        # GetProcessMemoryInfo (RSS)
  "shell32.lib"      # SHGetKnownFolderPath, SHOpenFolderAndSelectItems
  "ole32.lib"        # COM init for IFileSaveDialog / WASAPI
  "advapi32.lib"     # RegOpenKeyEx for device model / CPU brand
  "propsys.lib"      # IPropertyStore (audio device endpoint props)
  "runtimeobject.lib" # WinRT activation factory
)

# Suppress the MSVC C4996 deprecation noise from Windows SDK / WinRT
# headers. Allow exceptions globally — winrt + IFileSaveDialog throw.
target_compile_definitions(${PLUGIN_NAME} PRIVATE
  _CRT_SECURE_NO_WARNINGS
  _CRT_NONSTDC_NO_DEPRECATE
  NOMINMAX)
target_compile_options(${PLUGIN_NAME} PRIVATE "/EHsc")

# Compile the C++ core if the third_party submodules are present.
# Ignored when the package was pulled via pub git (submodules don't
# init); use a separate Windows DLL build (.github/workflows/windows.yml
# in this repo) to produce a vendored binary for those host builds.
set(EDGE_VEDA_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../core")
if(EXISTS "${EDGE_VEDA_CORE_DIR}/third_party/llama.cpp/CMakeLists.txt")
  message(STATUS "edge_veda: third_party submodules present; "
                  "compiling C++ core in-tree.")
  set(EDGE_VEDA_ENABLE_VULKAN ON CACHE BOOL "" FORCE)
  set(EDGE_VEDA_ENABLE_CPU ON CACHE BOOL "" FORCE)
  set(EDGE_VEDA_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(EDGE_VEDA_BUILD_STATIC ON CACHE BOOL "" FORCE)
  add_subdirectory("${EDGE_VEDA_CORE_DIR}" edge_veda_core_build)
  target_link_libraries(${PLUGIN_NAME} PRIVATE edge_veda_static)
  target_include_directories(${PLUGIN_NAME} PRIVATE
    "${EDGE_VEDA_CORE_DIR}/include")
  target_compile_definitions(${PLUGIN_NAME} PRIVATE
    "EDGE_VEDA_CORE_LINKED")
else()
  message(STATUS "edge_veda: third_party submodules not present; "
                  "building plugin without C++ core. ev_* FFI calls "
                  "will fail at dlsym — host app should keep "
                  "PlatformCapabilities.edgeVedaAvailable = false on "
                  "Windows until a vendored DLL is wired in.")
endif()

# List of absolute paths to libraries that should be bundled with the
# plugin in the install step.
set(edge_veda_bundled_libraries
  ""
  PARENT_SCOPE
)
