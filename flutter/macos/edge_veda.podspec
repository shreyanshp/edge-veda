Pod::Spec.new do |s|
  s.name             = 'edge_veda'
  s.version          = '2.5.0'
  s.summary          = 'Edge Veda SDK - On-device AI inference for Flutter'
  s.description      = 'On-device LLM, STT, and vision inference via llama.cpp'
  s.homepage         = 'https://github.com/shreyanshp/edge-veda'
  s.license          = { :type => 'Apache-2.0', :file => '../LICENSE' }
  s.author           = { 'Edge Veda' => 'contact@edgeveda.com' }
  s.source           = { :git => 'https://github.com/shreyanshp/edge-veda.git', :tag => s.version.to_s }
  s.platform         = :osx, '11.0'
  s.osx.deployment_target = '11.0'
  s.swift_version    = '5.0'
  s.source_files     = 'Classes/**/*'
  s.frameworks       = 'Metal', 'MetalPerformanceShaders', 'Accelerate', 'AVFoundation', 'Photos', 'EventKit', 'IOKit', 'AppKit'
  s.dependency 'FlutterMacOS'
  s.static_framework = true
  s.preserve_paths = 'Frameworks/EdgeVedaCore.xcframework'
  s.libraries = 'c++'

  # Vendored library — link the static archive directly
  s.vendored_libraries = 'Frameworks/EdgeVedaCore.xcframework/macos-arm64_x86_64/libedge_veda_full.a'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY' => 'libc++',
  }

  # Force-load the vendored static archive so ALL C/C++ symbols survive
  # dead stripping. Without this, dlsym(RTLD_DEFAULT, "ev_version") fails
  # because the linker strips symbols that no ObjC/Swift code references —
  # they are only called at runtime via Dart FFI.
  s.user_target_xcconfig = {
    'OTHER_LDFLAGS' => '$(inherited) -force_load "${PODS_ROOT}/../Flutter/ephemeral/.symlinks/plugins/edge_veda/macos/Frameworks/EdgeVedaCore.xcframework/macos-arm64_x86_64/libedge_veda_full.a"',
  }
end
