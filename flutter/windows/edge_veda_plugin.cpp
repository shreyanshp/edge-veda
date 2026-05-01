// Edge Veda Windows plugin — full method/event channel surface.
//
// Mirrors `flutter/macos/Classes/EdgeVedaPlugin.swift`. Each handler
// targets a Win32/WinRT API equivalent of the Apple primitive used on
// macOS; the wire format (Dart EncodableValue types) matches the macOS
// returns exactly so the Dart `lib/src/` layer is platform-agnostic.

#include "edge_veda_plugin.h"

#include <flutter/event_channel.h>
#include <flutter/event_sink.h>
#include <flutter/event_stream_handler_functions.h>
#include <flutter/method_channel.h>
#include <flutter/standard_method_codec.h>

#include <windows.h>
#include <psapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <combaseapi.h>
#include <propkey.h>
#include <propvarutil.h>
#include <atlbase.h>
#include <atlwin.h>

#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.System.Power.h>
#include <winrt/Windows.System.Profile.h>
#include <winrt/Windows.Media.SpeechSynthesis.h>
#include <winrt/Windows.Storage.Streams.h>
#include <winrt/Windows.Media.Playback.h>
#include <winrt/Windows.Media.Core.h>

#include <mmdeviceapi.h>
#include <audioclient.h>
#include <functiondiscoverykeys_devpkey.h>
#include <wrl/client.h>

#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "propsys.lib")

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace edge_veda {

using flutter::EncodableList;
using flutter::EncodableMap;
using flutter::EncodableValue;
using flutter::MethodCall;
using flutter::MethodResult;
using flutter::EventSink;

template <typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

namespace winfdn = winrt::Windows::Foundation;
namespace winpwr = winrt::Windows::System::Power;
namespace winprf = winrt::Windows::System::Profile;
namespace winsps = winrt::Windows::Media::SpeechSynthesis;
namespace winstr = winrt::Windows::Storage::Streams;
namespace winmpb = winrt::Windows::Media::Playback;
namespace winmcr = winrt::Windows::Media::Core;

// =====================================================================
// String helpers
// =====================================================================

std::wstring Utf8ToWide(const std::string& utf8) {
  if (utf8.empty()) return std::wstring();
  int len = ::MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(),
                                   static_cast<int>(utf8.size()),
                                   nullptr, 0);
  std::wstring wide(len, L'\0');
  ::MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(),
                         static_cast<int>(utf8.size()),
                         wide.data(), len);
  return wide;
}

std::string WideToUtf8(std::wstring_view wide) {
  if (wide.empty()) return std::string();
  int len = ::WideCharToMultiByte(CP_UTF8, 0, wide.data(),
                                    static_cast<int>(wide.size()),
                                    nullptr, 0, nullptr, nullptr);
  std::string utf8(len, '\0');
  ::WideCharToMultiByte(CP_UTF8, 0, wide.data(),
                         static_cast<int>(wide.size()),
                         utf8.data(), len, nullptr, nullptr);
  return utf8;
}

// =====================================================================
// Telemetry — memory / disk / device / battery / thermal / power
// =====================================================================

static int64_t GetProcessRSSBytes() {
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  pmc.cb = sizeof(pmc);
  if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                              sizeof(pmc))) {
    return 0;
  }
  return static_cast<int64_t>(pmc.WorkingSetSize);
}

static int64_t GetAvailableMemoryBytes() {
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof(ms);
  if (!GlobalMemoryStatusEx(&ms)) return 0;
  return static_cast<int64_t>(ms.ullAvailPhys);
}

static int64_t GetTotalMemoryBytes() {
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof(ms);
  if (!GlobalMemoryStatusEx(&ms)) return 0;
  return static_cast<int64_t>(ms.ullTotalPhys);
}

static int64_t GetFreeDiskSpaceBytes() {
  // Mirror macOS: report free space on the user's home volume. On
  // Windows we use SHGetKnownFolderPath(FOLDERID_Profile) — that's the
  // %USERPROFILE% root which is on the same volume as user data.
  PWSTR profile = nullptr;
  HRESULT hr = SHGetKnownFolderPath(FOLDERID_Profile, 0, nullptr, &profile);
  if (FAILED(hr) || !profile) return -1;
  ULARGE_INTEGER avail{};
  ULARGE_INTEGER total{};
  ULARGE_INTEGER total_free{};
  BOOL ok = GetDiskFreeSpaceExW(profile, &avail, &total, &total_free);
  CoTaskMemFree(profile);
  if (!ok) return -1;
  return static_cast<int64_t>(avail.QuadPart);
}

static std::string GetDeviceModelString() {
  // SystemManufacturer + SystemFamily / SystemProductName from registry
  // give a reasonable approximation of macOS hw.model. Fall back to
  // PROCESSOR_ARCHITECTURE if registry read fails.
  HKEY key;
  if (RegOpenKeyExW(HKEY_LOCAL_MACHINE,
                      L"HARDWARE\\DESCRIPTION\\System\\BIOS",
                      0, KEY_READ, &key) != ERROR_SUCCESS) {
    return std::string();
  }
  wchar_t mfg[256] = {};
  DWORD mfg_size = sizeof(mfg);
  wchar_t prod[256] = {};
  DWORD prod_size = sizeof(prod);
  RegQueryValueExW(key, L"SystemManufacturer", nullptr, nullptr,
                    reinterpret_cast<LPBYTE>(mfg), &mfg_size);
  RegQueryValueExW(key, L"SystemProductName", nullptr, nullptr,
                    reinterpret_cast<LPBYTE>(prod), &prod_size);
  RegCloseKey(key);
  std::wstring combined;
  if (mfg[0] && prod[0]) {
    combined = std::wstring(mfg) + L" " + std::wstring(prod);
  } else if (prod[0]) {
    combined = std::wstring(prod);
  } else if (mfg[0]) {
    combined = std::wstring(mfg);
  }
  return WideToUtf8(combined);
}

static std::string GetCpuBrandString() {
  // Read CPU "ProcessorNameString" from registry — the same string
  // wmic / Task Manager show. Equivalent to macOS
  // machdep.cpu.brand_string.
  HKEY key;
  if (RegOpenKeyExW(HKEY_LOCAL_MACHINE,
                      L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &key) != ERROR_SUCCESS) {
    return std::string();
  }
  wchar_t name[512] = {};
  DWORD size = sizeof(name);
  RegQueryValueExW(key, L"ProcessorNameString", nullptr, nullptr,
                    reinterpret_cast<LPBYTE>(name), &size);
  RegCloseKey(key);
  return WideToUtf8(std::wstring(name));
}

static std::string GetGpuBackendLabel() {
  // The C++ core picks Vulkan by default on Windows (see
  // core/CMakeLists.txt). CUDA only appears when the build was
  // configured with -DEDGE_VEDA_ENABLE_CUDA=ON. We probe the linked
  // ggml backend at runtime via ev_get_gpu_backend() once that lands;
  // for now report the configured backend conservatively.
  //
  // The Dart side uses this label only for telemetry / diagnostics —
  // it does not branch behaviour on it.
#if defined(EDGE_VEDA_CUDA_LINKED)
  return "CUDA";
#elif defined(EDGE_VEDA_VULKAN_LINKED)
  return "Vulkan";
#else
  return "CPU";
#endif
}

// Battery + power via WinRT.
static double GetBatteryLevelFraction() {
  try {
    auto status = winpwr::PowerManager::RemainingChargePercent();
    if (status < 0) return -1.0;
    return static_cast<double>(status) / 100.0;
  } catch (...) {
    return -1.0;
  }
}

static int GetBatteryStateInt() {
  // 0=unknown, 1=unplugged, 2=charging, 3=full
  try {
    SYSTEM_POWER_STATUS sps{};
    if (!GetSystemPowerStatus(&sps)) return 0;
    if (sps.BatteryFlag == 128 /*no battery*/) return 0;
    if ((sps.BatteryFlag & 8) /*charging*/) return 2;
    if (sps.BatteryLifePercent == 100) return 3;
    if (sps.ACLineStatus == 0) return 1;
    if (sps.ACLineStatus == 1) return 2;  // plugged in (treat as charging-ish)
    return 0;
  } catch (...) {
    return 0;
  }
}

static bool IsLowPowerModeEnabled() {
  try {
    auto saver = winpwr::PowerManager::EnergySaverStatus();
    return saver == winpwr::EnergySaverStatus::On;
  } catch (...) {
    return false;
  }
}

static int GetThermalStateInt() {
  // Windows exposes no per-app thermal-state API as cleanly as
  // ProcessInfo.thermalState on Apple. Return 0 (.nominal). We can
  // refine via a perf-counter heuristic later if needed.
  return 0;
}

// =====================================================================
// File share / save-as
// =====================================================================

static bool ShareFileViaShell(HWND owner, const std::wstring& path) {
  // Windows 10+ has IDataTransferManager + SHCreateDataObject for
  // sharing. The simplest reliable path that opens the system share
  // sheet is `OpenAs_RunDLL` — but it shows "Open with" rather than
  // "Share". For a real Share UI we'd need IDataTransferManager via
  // the IDataTransferManagerInterop COM interface — that requires
  // running with package identity (MSIX) so it's a polish item.
  //
  // For now: open the containing folder with the file selected via
  // SHOpenFolderAndSelectItems. That gives the user an immediate
  // path to right-click → Share or copy elsewhere.
  ITEMIDLIST* pidl = ILCreateFromPathW(path.c_str());
  if (!pidl) return false;
  HRESULT hr = SHOpenFolderAndSelectItems(pidl, 0, nullptr, 0);
  ILFree(pidl);
  return SUCCEEDED(hr);
}

static std::wstring SaveFileDialog(HWND owner,
                                    const std::wstring& source_path) {
  ComPtr<IFileSaveDialog> dlg;
  HRESULT hr = CoCreateInstance(CLSID_FileSaveDialog, nullptr,
                                  CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(&dlg));
  if (FAILED(hr)) return L"";

  // Default to the Downloads folder.
  PWSTR downloads = nullptr;
  if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Downloads, 0, nullptr,
                                       &downloads))) {
    ComPtr<IShellItem> item;
    if (SUCCEEDED(SHCreateItemFromParsingName(downloads, nullptr,
                                                IID_PPV_ARGS(&item)))) {
      dlg->SetDefaultFolder(item.Get());
    }
    CoTaskMemFree(downloads);
  }

  // Default file name = basename of source.
  size_t slash = source_path.find_last_of(L"\\/");
  std::wstring base = (slash == std::wstring::npos)
      ? source_path
      : source_path.substr(slash + 1);
  dlg->SetFileName(base.c_str());

  hr = dlg->Show(owner);
  if (FAILED(hr)) return L"";  // user cancelled
  ComPtr<IShellItem> result_item;
  if (FAILED(dlg->GetResult(&result_item))) return L"";
  PWSTR result_path = nullptr;
  if (FAILED(result_item->GetDisplayName(SIGDN_FILESYSPATH,
                                          &result_path))) {
    return L"";
  }
  std::wstring path = result_path;
  CoTaskMemFree(result_path);
  return path;
}

// =====================================================================
// TTS — Windows.Media.SpeechSynthesis
// =====================================================================

class TtsHandler {
 public:
  TtsHandler() = default;

  void SetEventSink(std::unique_ptr<EventSink<EncodableValue>> sink) {
    std::lock_guard<std::mutex> lk(mu_);
    sink_ = std::move(sink);
  }
  void ClearEventSink() {
    std::lock_guard<std::mutex> lk(mu_);
    sink_.reset();
    Stop();
  }

  EncodableList AvailableVoices() {
    EncodableList out;
    try {
      auto voices = winsps::SpeechSynthesizer::AllVoices();
      for (const auto& v : voices) {
        EncodableMap entry;
        entry[EncodableValue("id")] =
            EncodableValue(WideToUtf8(std::wstring_view(v.Id())));
        entry[EncodableValue("name")] =
            EncodableValue(WideToUtf8(std::wstring_view(v.DisplayName())));
        entry[EncodableValue("language")] =
            EncodableValue(WideToUtf8(std::wstring_view(v.Language())));
        out.push_back(EncodableValue(entry));
      }
    } catch (...) {
    }
    return out;
  }

  void Speak(const std::string& text,
              const std::string& voice_id_utf8,
              double rate, double pitch, double volume) {
    Stop();
    try {
      synth_ = winsps::SpeechSynthesizer();
      if (!voice_id_utf8.empty()) {
        for (auto& v : winsps::SpeechSynthesizer::AllVoices()) {
          auto vid = WideToUtf8(std::wstring_view(v.Id()));
          if (vid == voice_id_utf8) {
            synth_.Voice(v);
            break;
          }
        }
      }
      auto opts = synth_.Options();
      // SpeakingRate range: 0.5 .. 6.0 (default 1.0). Map directly.
      opts.SpeakingRate(rate <= 0 ? 1.0 : rate);
      // AudioPitch range: 0.0 .. 2.0 (default 1.0).
      opts.AudioPitch(pitch <= 0 ? 1.0 : pitch);
      // AudioVolume range: 0.0 .. 1.0 (default 1.0).
      opts.AudioVolume(volume < 0 ? 1.0 : (volume > 1.0 ? 1.0 : volume));

      auto wide = Utf8ToWide(text);
      auto stream =
          synth_.SynthesizeTextToStreamAsync(wide).get();
      player_ = winmpb::MediaPlayer();
      auto src = winmcr::MediaSource::CreateFromStream(
          stream, stream.ContentType());
      player_.Source(src);
      player_.MediaEnded([this](auto&&, auto&&) { Emit("done"); });
      player_.Play();
      Emit("start");
    } catch (...) {
      Emit("cancel");
    }
  }
  void Stop() {
    try {
      if (player_) {
        player_.Pause();
        player_.Source(nullptr);
        player_ = nullptr;
        Emit("cancel");
      }
    } catch (...) {
    }
  }
  void Pause() {
    try { if (player_) player_.Pause(); Emit("pause"); } catch (...) {}
  }
  void Resume() {
    try { if (player_) player_.Play(); Emit("resume"); } catch (...) {}
  }

 private:
  void Emit(const char* type) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!sink_) return;
    EncodableMap m;
    m[EncodableValue("type")] = EncodableValue(std::string(type));
    sink_->Success(EncodableValue(m));
  }

  std::mutex mu_;
  std::unique_ptr<EventSink<EncodableValue>> sink_;
  winsps::SpeechSynthesizer synth_{nullptr};
  winmpb::MediaPlayer player_{nullptr};
};

static TtsHandler& Tts() {
  static TtsHandler t;
  return t;
}

// =====================================================================
// Audio capture — WASAPI 16 kHz mono Float32 PCM EventChannel
// =====================================================================

class AudioCaptureStreamHandler
    : public flutter::StreamHandler<EncodableValue> {
 public:
  std::unique_ptr<flutter::StreamHandlerError<EncodableValue>>
  OnListenInternal(
      const EncodableValue* arguments,
      std::unique_ptr<EventSink<EncodableValue>>&& events) override {
    sink_ = std::move(events);
    stop_.store(false);
    capture_thread_ = std::thread([this]() { CaptureLoop(); });
    return nullptr;
  }

  std::unique_ptr<flutter::StreamHandlerError<EncodableValue>>
  OnCancelInternal(const EncodableValue* arguments) override {
    stop_.store(true);
    if (capture_thread_.joinable()) capture_thread_.join();
    sink_.reset();
    return nullptr;
  }

 private:
  void CaptureLoop() {
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) return;

    ComPtr<IMMDeviceEnumerator> enumerator;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                                  CLSCTX_ALL,
                                  __uuidof(IMMDeviceEnumerator),
                                  (void**)enumerator.GetAddressOf()))) {
      EmitError("MIC_ENUM_FAILED",
                  "Could not enumerate audio capture devices.");
      CoUninitialize();
      return;
    }

    ComPtr<IMMDevice> device;
    if (FAILED(enumerator->GetDefaultAudioEndpoint(
            eCapture, eCommunications, device.GetAddressOf()))) {
      EmitError("MIC_NO_DEVICE",
                  "No default audio capture device found.");
      CoUninitialize();
      return;
    }

    ComPtr<IAudioClient> client;
    if (FAILED(device->Activate(__uuidof(IAudioClient), CLSCTX_ALL,
                                 nullptr,
                                 (void**)client.GetAddressOf()))) {
      EmitError("MIC_ACTIVATE_FAILED",
                  "Could not activate audio client.");
      CoUninitialize();
      return;
    }

    // Request 16 kHz mono Float32 PCM directly. WASAPI's mix-format
    // negotiation will resample if necessary.
    WAVEFORMATEX wfx{};
    wfx.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
    wfx.nChannels = 1;
    wfx.nSamplesPerSec = 16000;
    wfx.wBitsPerSample = 32;
    wfx.nBlockAlign = wfx.nChannels * wfx.wBitsPerSample / 8;
    wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;
    wfx.cbSize = 0;

    REFERENCE_TIME buffer_duration = 200 * 10000;  // 200 ms in 100ns
    hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED,
                              0,
                              buffer_duration, 0, &wfx, nullptr);
    if (FAILED(hr)) {
      EmitError("MIC_INIT_FAILED",
                  "Audio client init failed (HRESULT 0x" +
                      HexFromHResult(hr) + ")");
      CoUninitialize();
      return;
    }

    ComPtr<IAudioCaptureClient> cap;
    if (FAILED(client->GetService(__uuidof(IAudioCaptureClient),
                                    (void**)cap.GetAddressOf()))) {
      EmitError("MIC_SERVICE_FAILED", "Capture client unavailable.");
      CoUninitialize();
      return;
    }

    if (FAILED(client->Start())) {
      EmitError("MIC_START_FAILED", "Could not start capture.");
      CoUninitialize();
      return;
    }

    while (!stop_.load()) {
      UINT32 pkt = 0;
      cap->GetNextPacketSize(&pkt);
      while (pkt > 0 && !stop_.load()) {
        BYTE* data = nullptr;
        UINT32 frames = 0;
        DWORD flags = 0;
        if (FAILED(cap->GetBuffer(&data, &frames, &flags,
                                    nullptr, nullptr))) {
          break;
        }
        size_t bytes = static_cast<size_t>(frames) * wfx.nBlockAlign;
        if (data && bytes > 0 &&
            !(flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
          std::vector<uint8_t> buf(data, data + bytes);
          EmitFrame(std::move(buf));
        } else if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
          // Emit zeroed buffer so timestamp continuity is preserved
          // for the Whisper pipeline.
          std::vector<uint8_t> buf(bytes, 0);
          EmitFrame(std::move(buf));
        }
        cap->ReleaseBuffer(frames);
        cap->GetNextPacketSize(&pkt);
      }
      Sleep(10);
    }

    client->Stop();
    CoUninitialize();
  }

  void EmitFrame(std::vector<uint8_t>&& buf) {
    if (!sink_) return;
    sink_->Success(EncodableValue(std::move(buf)));
  }
  void EmitError(const char* code, const std::string& msg) {
    if (!sink_) return;
    sink_->Error(code, msg);
  }
  std::string HexFromHResult(HRESULT hr) {
    char hex[16];
    _snprintf_s(hex, sizeof(hex), _TRUNCATE, "%08X", hr);
    return std::string(hex);
  }

  std::unique_ptr<EventSink<EncodableValue>> sink_;
  std::atomic<bool> stop_{false};
  std::thread capture_thread_;
};

// =====================================================================
// Thermal + TTS event channels
// =====================================================================

class ThermalStreamHandler
    : public flutter::StreamHandler<EncodableValue> {
 public:
  std::unique_ptr<flutter::StreamHandlerError<EncodableValue>>
  OnListenInternal(
      const EncodableValue* arguments,
      std::unique_ptr<EventSink<EncodableValue>>&& events) override {
    sink_ = std::move(events);
    Send();  // initial value
    return nullptr;
  }
  std::unique_ptr<flutter::StreamHandlerError<EncodableValue>>
  OnCancelInternal(const EncodableValue* arguments) override {
    sink_.reset();
    return nullptr;
  }

 private:
  void Send() {
    if (!sink_) return;
    SYSTEMTIME st;
    GetSystemTime(&st);
    EncodableMap m;
    m[EncodableValue("thermalState")] =
        EncodableValue(GetThermalStateInt());
    // Match macOS payload: timestamp in ms since epoch.
    FILETIME ft;
    SystemTimeToFileTime(&st, &ft);
    ULARGE_INTEGER u;
    u.LowPart = ft.dwLowDateTime;
    u.HighPart = ft.dwHighDateTime;
    int64_t ms = (u.QuadPart - 116444736000000000LL) / 10000LL;
    m[EncodableValue("timestamp")] =
        EncodableValue(static_cast<double>(ms));
    sink_->Success(EncodableValue(m));
  }

  std::unique_ptr<EventSink<EncodableValue>> sink_;
};

class TtsEventStreamHandler
    : public flutter::StreamHandler<EncodableValue> {
 public:
  std::unique_ptr<flutter::StreamHandlerError<EncodableValue>>
  OnListenInternal(
      const EncodableValue* arguments,
      std::unique_ptr<EventSink<EncodableValue>>&& events) override {
    Tts().SetEventSink(std::move(events));
    return nullptr;
  }
  std::unique_ptr<flutter::StreamHandlerError<EncodableValue>>
  OnCancelInternal(const EncodableValue* arguments) override {
    Tts().ClearEventSink();
    return nullptr;
  }
};

// =====================================================================
// Plugin registration / dispatch
// =====================================================================

// static
void EdgeVedaPlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarWindows* registrar) {
  // Telemetry method channel.
  auto telemetry =
      std::make_unique<flutter::MethodChannel<EncodableValue>>(
          registrar->messenger(),
          "com.edgeveda.edge_veda/telemetry",
          &flutter::StandardMethodCodec::GetInstance());

  auto plugin = std::make_unique<EdgeVedaPlugin>();
  plugin->host_window_ = registrar->GetView()->GetNativeWindow();

  telemetry->SetMethodCallHandler(
      [plug = plugin.get()](const auto& call, auto result) {
        plug->HandleMethodCall(call, std::move(result));
      });

  // Thermal event channel.
  auto thermal_channel =
      std::make_unique<flutter::EventChannel<EncodableValue>>(
          registrar->messenger(),
          "com.edgeveda.edge_veda/thermal",
          &flutter::StandardMethodCodec::GetInstance());
  thermal_channel->SetStreamHandler(
      std::make_unique<ThermalStreamHandler>());

  // Audio capture event channel.
  auto audio_channel =
      std::make_unique<flutter::EventChannel<EncodableValue>>(
          registrar->messenger(),
          "com.edgeveda.edge_veda/audio_capture",
          &flutter::StandardMethodCodec::GetInstance());
  audio_channel->SetStreamHandler(
      std::make_unique<AudioCaptureStreamHandler>());

  // TTS event channel.
  auto tts_event_channel =
      std::make_unique<flutter::EventChannel<EncodableValue>>(
          registrar->messenger(),
          "com.edgeveda.edge_veda/tts_events",
          &flutter::StandardMethodCodec::GetInstance());
  tts_event_channel->SetStreamHandler(
      std::make_unique<TtsEventStreamHandler>());

  // Keep references alive for the lifetime of the plugin.
  registrar->AddPlugin(std::move(plugin));
  // Channels themselves capture shared state internally; leak the
  // unique_ptrs into static storage so they live as long as the
  // engine. (Flutter's EventChannel doesn't have an "AddPlugin"
  // sibling.)
  static std::vector<
      std::unique_ptr<flutter::EventChannel<EncodableValue>>>
      kept;
  kept.push_back(std::move(thermal_channel));
  kept.push_back(std::move(audio_channel));
  kept.push_back(std::move(tts_event_channel));
  static std::vector<
      std::unique_ptr<flutter::MethodChannel<EncodableValue>>>
      kept_methods;
  kept_methods.push_back(std::move(telemetry));
}

EdgeVedaPlugin::EdgeVedaPlugin() {}
EdgeVedaPlugin::~EdgeVedaPlugin() {}

void EdgeVedaPlugin::HandleMethodCall(
    const MethodCall<EncodableValue>& call,
    std::unique_ptr<MethodResult<EncodableValue>> result) {
  const std::string& method = call.method_name();

  // ── Pure-numeric telemetry ────────────────────────────────────────
  if (method == "getMemoryRSS") {
    result->Success(EncodableValue(GetProcessRSSBytes()));
    return;
  }
  if (method == "getAvailableMemory") {
    result->Success(EncodableValue(GetAvailableMemoryBytes()));
    return;
  }
  if (method == "getTotalMemory") {
    result->Success(EncodableValue(GetTotalMemoryBytes()));
    return;
  }
  if (method == "getFreeDiskSpace") {
    result->Success(EncodableValue(GetFreeDiskSpaceBytes()));
    return;
  }
  if (method == "getThermalState") {
    result->Success(EncodableValue(GetThermalStateInt()));
    return;
  }
  if (method == "getBatteryLevel") {
    result->Success(EncodableValue(GetBatteryLevelFraction()));
    return;
  }
  if (method == "getBatteryState") {
    result->Success(EncodableValue(GetBatteryStateInt()));
    return;
  }
  if (method == "isLowPowerMode") {
    result->Success(EncodableValue(IsLowPowerModeEnabled()));
    return;
  }
  if (method == "getDeviceModel") {
    result->Success(EncodableValue(GetDeviceModelString()));
    return;
  }
  if (method == "getChipName") {
    result->Success(EncodableValue(GetCpuBrandString()));
    return;
  }
  if (method == "hasNeuralEngine") {
    // Windows desktops have no general-purpose Neural Engine analogue.
    // NPU detection is vendor-specific (Intel NPU via OpenVINO, AMD
    // Ryzen AI, Qualcomm Hexagon) and not exposed via a single Win32
    // API. Conservative default: false.
    result->Success(EncodableValue(false));
    return;
  }
  if (method == "getGpuBackend") {
    result->Success(EncodableValue(GetGpuBackendLabel()));
    return;
  }

  // ── Voice pipeline / mic permission ───────────────────────────────
  if (method == "configureVoicePipelineAudio") {
    // Windows has no equivalent of AVAudioRoutingArbiter — WASAPI
    // share-mode capture coexists with playback by default. Verify
    // a default capture endpoint exists; report ready.
    ComPtr<IMMDeviceEnumerator> e;
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                                    CLSCTX_ALL,
                                    __uuidof(IMMDeviceEnumerator),
                                    (void**)e.GetAddressOf());
    if (FAILED(hr)) {
      result->Error("AUDIO_INIT_FAILED",
                    "Could not enumerate audio endpoints.");
      return;
    }
    ComPtr<IMMDevice> dev;
    hr = e->GetDefaultAudioEndpoint(eCapture, eCommunications,
                                       dev.GetAddressOf());
    if (FAILED(hr) || !dev) {
      result->Error("AUDIO_INPUT_UNAVAILABLE",
                    "No default audio input device found.");
      return;
    }
    result->Success(EncodableValue(true));
    return;
  }
  if (method == "resetAudioSession") {
    // No persistent session state on Windows WASAPI share-mode — the
    // capture client is torn down per session by the EventChannel
    // handler. Always succeed.
    result->Success(EncodableValue(true));
    return;
  }
  if (method == "requestMicrophonePermission") {
    // Windows mic privacy gate is set in Settings → Privacy → Microphone.
    // There's no programmatic "request access" — the system surface
    // is shown the first time WASAPI is opened. Report true so the
    // Dart side proceeds; the EventChannel handler emits MIC_INIT_FAILED
    // if the user has denied access at the OS level.
    result->Success(EncodableValue(true));
    return;
  }

  // ── Share / save ──────────────────────────────────────────────────
  if (method == "shareFile") {
    const auto* args = std::get_if<EncodableMap>(call.arguments());
    if (!args) {
      result->Error("INVALID_ARG", "Missing 'path' argument.");
      return;
    }
    auto it = args->find(EncodableValue("path"));
    if (it == args->end()) {
      result->Error("INVALID_ARG", "Missing 'path' argument.");
      return;
    }
    auto path = std::get<std::string>(it->second);
    DWORD attr = GetFileAttributesW(Utf8ToWide(path).c_str());
    if (attr == INVALID_FILE_ATTRIBUTES) {
      result->Error("FILE_NOT_FOUND", "File not found", path);
      return;
    }
    bool ok = ShareFileViaShell(host_window_, Utf8ToWide(path));
    result->Success(EncodableValue(ok));
    return;
  }
  if (method == "saveFileToDownloads") {
    const auto* args = std::get_if<EncodableMap>(call.arguments());
    if (!args) {
      result->Error("INVALID_ARG", "Missing 'path' argument.");
      return;
    }
    auto it = args->find(EncodableValue("path"));
    if (it == args->end()) {
      result->Error("INVALID_ARG", "Missing 'path' argument.");
      return;
    }
    auto src_utf8 = std::get<std::string>(it->second);
    auto src = Utf8ToWide(src_utf8);
    if (GetFileAttributesW(src.c_str()) == INVALID_FILE_ATTRIBUTES) {
      result->Error("FILE_NOT_FOUND", "File not found", src_utf8);
      return;
    }
    auto dst = SaveFileDialog(host_window_, src);
    if (dst.empty()) {
      result->Success({});  // user cancelled
      return;
    }
    if (!CopyFileW(src.c_str(), dst.c_str(), FALSE)) {
      DWORD err = GetLastError();
      result->Error("COPY_FAILED",
                    "CopyFile failed",
                    std::to_string(static_cast<unsigned>(err)));
      return;
    }
    result->Success(EncodableValue(WideToUtf8(dst)));
    return;
  }

  // ── TTS ───────────────────────────────────────────────────────────
  if (method == "tts_speak") {
    const auto* args = std::get_if<EncodableMap>(call.arguments());
    if (!args) {
      result->Error("INVALID_ARG", "Missing 'text' argument.");
      return;
    }
    auto text_it = args->find(EncodableValue("text"));
    if (text_it == args->end()) {
      result->Error("INVALID_ARG", "Missing 'text' argument.");
      return;
    }
    auto text = std::get<std::string>(text_it->second);
    std::string voice_id;
    double rate = 1.0, pitch = 1.0, volume = 1.0;
    if (auto v = args->find(EncodableValue("voiceId"));
          v != args->end() &&
          std::holds_alternative<std::string>(v->second)) {
      voice_id = std::get<std::string>(v->second);
    }
    if (auto v = args->find(EncodableValue("rate"));
          v != args->end()) {
      if (auto* d = std::get_if<double>(&v->second)) rate = *d;
      else if (auto* i = std::get_if<int>(&v->second)) rate = *i;
    }
    if (auto v = args->find(EncodableValue("pitch"));
          v != args->end()) {
      if (auto* d = std::get_if<double>(&v->second)) pitch = *d;
      else if (auto* i = std::get_if<int>(&v->second)) pitch = *i;
    }
    if (auto v = args->find(EncodableValue("volume"));
          v != args->end()) {
      if (auto* d = std::get_if<double>(&v->second)) volume = *d;
      else if (auto* i = std::get_if<int>(&v->second)) volume = *i;
    }
    Tts().Speak(text, voice_id, rate, pitch, volume);
    result->Success(EncodableValue(true));
    return;
  }
  if (method == "tts_stop") {
    Tts().Stop();
    result->Success(EncodableValue(true));
    return;
  }
  if (method == "tts_pause") {
    Tts().Pause();
    result->Success(EncodableValue(true));
    return;
  }
  if (method == "tts_resume") {
    Tts().Resume();
    result->Success(EncodableValue(true));
    return;
  }
  if (method == "tts_voices") {
    result->Success(EncodableValue(Tts().AvailableVoices()));
    return;
  }

  // ── macOS-only (return UNAVAILABLE / sensible defaults) ──────────
  if (method == "checkDetectivePermissions") {
    // Photo + Calendar permissions on macOS only; no Windows analogue
    // for the Detective flow (calendar events + photo library
    // semantic search). Report all-denied so the Dart layer hides
    // the feature.
    EncodableMap m;
    m[EncodableValue("photos")] = EncodableValue(std::string("denied"));
    m[EncodableValue("calendar")] =
        EncodableValue(std::string("denied"));
    result->Success(EncodableValue(m));
    return;
  }
  if (method == "requestDetectivePermissions") {
    EncodableMap m;
    m[EncodableValue("photos")] = EncodableValue(std::string("denied"));
    m[EncodableValue("calendar")] =
        EncodableValue(std::string("denied"));
    result->Success(EncodableValue(m));
    return;
  }
  if (method == "getPhotoInsights" || method == "getCalendarInsights") {
    // Surface as empty list — the Dart side merges these with other
    // sources, so empty == feature off rather than error path.
    result->Success(EncodableValue(EncodableList{}));
    return;
  }

  result->NotImplemented();
}

}  // namespace edge_veda
