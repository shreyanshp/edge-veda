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
# When pulled via pub git (no submodule init), the core build is
# skipped — host apps should vendor a pre-built `edge_veda.dll` next
# to the exe (the windows-dll.yml workflow in this repo produces that
# artifact and attaches it to releases).
#
# The Dart FFI layer at `lib/src/ffi/bindings.dart:984` opens
# `edge_veda.dll` by name, so we build the core as SHARED and add
# the resulting DLL to `edge_veda_bundled_libraries` so Flutter's
# install step copies it next to the runner exe.
set(EDGE_VEDA_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../core")
set(EDGE_VEDA_DLL_PATHS "")
if(EXISTS "${EDGE_VEDA_CORE_DIR}/third_party/llama.cpp/CMakeLists.txt")
  message(STATUS "edge_veda: third_party submodules present; "
                  "compiling C++ core in-tree as edge_veda.dll.")
  set(EDGE_VEDA_ENABLE_VULKAN ON CACHE BOOL "" FORCE)
  set(EDGE_VEDA_ENABLE_CPU ON CACHE BOOL "" FORCE)
  set(EDGE_VEDA_BUILD_SHARED ON CACHE BOOL "" FORCE)
  set(EDGE_VEDA_BUILD_STATIC OFF CACHE BOOL "" FORCE)
  add_subdirectory("${EDGE_VEDA_CORE_DIR}" edge_veda_core_build)
  # Plugin doesn't link the core — FFI dlopen reaches it at runtime.
  # We just need to ensure the DLL ships next to the exe.
  list(APPEND EDGE_VEDA_DLL_PATHS "$<TARGET_FILE:edge_veda>")
  target_compile_definitions(${PLUGIN_NAME} PRIVATE
    "EDGE_VEDA_CORE_LINKED")
else()
  message(STATUS "edge_veda: third_party submodules not present. "
                  "Pre-built edge_veda.dll must be present in "
                  "flutter/windows/Frameworks/ and added to "
                  "edge_veda_bundled_libraries below for FFI to "
                  "resolve at runtime.")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Frameworks/edge_veda.dll")
    list(APPEND EDGE_VEDA_DLL_PATHS
      "${CMAKE_CURRENT_SOURCE_DIR}/Frameworks/edge_veda.dll")
  endif()
endif()

# List of absolute paths to libraries that should be bundled with the
# plugin in the install step. Flutter copies these next to the runner
# .exe so DynamicLibrary.open('edge_veda.dll') resolves at runtime.
set(edge_veda_bundled_libraries
  "${EDGE_VEDA_DLL_PATHS}"
  PARENT_SCOPE
)
