/// Device-aware model recommendation engine for Edge Veda SDK
///
/// Provides device profiling (via sysctlbyname FFI), memory estimation
/// with calibrated formulas, and 4-dimensional model scoring with
/// use-case weighted recommendations.
library;

import 'dart:ffi' as ffi;
import 'dart:io' show File, Platform;
import 'dart:math' show log, min, max;

import 'package:ffi/ffi.dart';

import 'types.dart';
import 'model_manager.dart';
import 'telemetry_service.dart';
import 'edge_veda_impl.dart';

// ── FFI Typedefs ──────────────────────────────────────────────────────────

typedef _SysctlByNameC =
    ffi.Int Function(
      ffi.Pointer<ffi.Char>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Size>,
      ffi.Pointer<ffi.Void>,
      ffi.Size,
    );
typedef _SysctlByNameDart =
    int Function(
      ffi.Pointer<ffi.Char>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Size>,
      ffi.Pointer<ffi.Void>,
      int,
    );

// ── Enums ─────────────────────────────────────────────────────────────────

/// Device capability tier based on RAM and chip generation
enum DeviceTier {
  minimum, // 4 GB, A14/A15
  low, // 6 GB, A14/A15
  medium, // 6 GB, A16
  high, // 8 GB, A17 Pro/A18/A18 Pro/A19
  ultra, // 12 GB, A19 Pro
}

/// Use case for model recommendation
enum UseCase { chat, reasoning, toolCalling, vision, stt, embedding, fast }

// ── DeviceProfile ─────────────────────────────────────────────────────────

/// Device hardware profile detected via sysctlbyname FFI
///
/// Provides sync, cached device identification for model recommendations.
/// Uses hw.machine identifier to look up chip, RAM, and tier from a
/// built-in database of 27 iPhone models (iPhone 12 through iPhone 17).
class DeviceProfile {
  /// Raw hw.machine identifier (e.g., "iPhone17,1")
  final String identifier;

  /// Human-readable device name (e.g., "iPhone 16 Pro")
  final String deviceName;

  /// Total device RAM in GB
  final double totalRamGB;

  /// Apple chip name (e.g., "A18 Pro")
  final String chipName;

  /// Device capability tier
  final DeviceTier tier;

  /// True if a GPU backend is available for inference acceleration
  /// (Metal on Apple, Vulkan on Android with `Vulkan 1.2+` support,
  /// CUDA / Vulkan on Windows / Linux desktops). Filled by the
  /// async `detectFull` path; the sync `detect` constructor leaves
  /// this `false` because it can't see the runtime probe.
  ///
  /// Use `model_advisor.recommend(...)` and
  /// `RuntimePolicy.evaluate(...)` consumers should *not* assume
  /// CPU-only when this is `false` — call `DeviceProfile.detectFull`
  /// once during app boot before relying on the value.
  final bool hasGpuBackend;

  const DeviceProfile({
    required this.identifier,
    required this.deviceName,
    required this.totalRamGB,
    required this.chipName,
    required this.tier,
    this.hasGpuBackend = false,
  });

  /// Safe memory budget in MB (Android: 50%, iOS: 60%, macOS: 80%)
  int get safeMemoryBudgetMB {
    if (Platform.isAndroid) {
      return (totalRamGB * 1024 * 0.50).round();
    }
    return (totalRamGB * 1024 * (Platform.isMacOS ? 0.80 : 0.60)).round();
  }

  // ── Detection (sync, cached) ──

  static DeviceProfile? _cached;

  // Lazily resolved — only accessed on Apple platforms (iOS/macOS).
  // On Android, sysctlbyname does not exist in the process symbol table.
  static _SysctlByNameDart? _sysctlbyname;

  static _SysctlByNameDart _getSysctlbyname() {
    _sysctlbyname ??= ffi.DynamicLibrary.process()
        .lookupFunction<_SysctlByNameC, _SysctlByNameDart>('sysctlbyname');
    return _sysctlbyname!;
  }

  /// Async refresh of the cached profile that fills in the
  /// information `detect()` can't see synchronously:
  ///
  /// - **Android `Build.SOC_MODEL`** (Android 12+) — reliably names
  ///   the actual SoC ("SM8650", "Tensor G3", "Exynos 2400") so
  ///   `ThreadAdvisor.recommend(...)` can pick the right thread
  ///   count. `/proc/cpuinfo`'s `Hardware:` line on most flagships
  ///   reports a useless "Qualcomm Technologies, Inc TARO" string.
  /// - **GPU backend availability** — Metal (always on Apple),
  ///   Vulkan (probed via PackageManager.FEATURE_VULKAN_HARDWARE_VERSION
  ///   ≥ 1.2 on Android), CUDA / Vulkan on desktops.
  ///
  /// Call this once during app boot (after `EdgeVeda.init` if the
  /// host wants the most-precise picture) and the result becomes
  /// the new value of `DeviceProfile.detect()`. Falls back to the
  /// sync profile silently if the channel call fails.
  static Future<DeviceProfile> detectFull([TelemetryService? telemetry]) async {
    final base = detect();
    final ts = telemetry ?? TelemetryService();
    String chipName = base.chipName;
    String deviceName = base.deviceName;
    bool hasGpu = base.hasGpuBackend;

    // Apple platforms: Metal is universal on macOS 11+ / iOS 13+,
    // already implied by being on the platform at all.
    if (Platform.isIOS || Platform.isMacOS) {
      hasGpu = true;
    }

    if (Platform.isAndroid) {
      // Build.SOC_MODEL is the only reliable SoC name source on
      // Android 12+; chipName from /proc/cpuinfo is flagship-noise.
      try {
        final socModel = await ts.getChipName();
        if (socModel.isNotEmpty && socModel != 'unknown') {
          chipName = socModel;
        }
      } catch (_) {}
      try {
        final dm = await ts.getDeviceModel();
        if (dm.isNotEmpty) deviceName = dm;
      } catch (_) {}
      try {
        final backend = await ts.getGpuBackend();
        hasGpu = backend != 'CPU' && backend.isNotEmpty;
      } catch (_) {}
    }

    final refined = DeviceProfile(
      identifier: base.identifier,
      deviceName: deviceName,
      totalRamGB: base.totalRamGB,
      chipName: chipName,
      tier: base.tier,
      hasGpuBackend: hasGpu,
    );
    _cached = refined;
    return refined;
  }

  /// Detect current device hardware profile.
  ///
  /// Sync call, cached after first invocation. On Android, returns a
  /// conservative default profile (async detection via MethodChannel is
  /// handled separately by TelemetryService). On simulator or unknown
  /// Apple hardware, falls back to RAM-based tier detection via hw.memsize.
  ///
  /// **For best results call [detectFull] once at boot** — it fills in
  /// SOC_MODEL and GPU-backend availability that the sync path can't see.
  static DeviceProfile detect() {
    if (_cached != null) return _cached!;

    // Android: read /proc/meminfo for total RAM and /proc/cpuinfo for chip.
    // No MethodChannel needed — these files are always readable.
    if (Platform.isAndroid) {
      double ramGB = 4.0;
      String chipName = 'ARM64';
      String deviceName = 'Android Device';
      try {
        final memInfo = File('/proc/meminfo').readAsStringSync();
        final match = RegExp(r'MemTotal:\s+(\d+)\s+kB').firstMatch(memInfo);
        if (match != null) {
          ramGB = int.parse(match.group(1)!) / (1024 * 1024);
        }
      } catch (_) {}
      try {
        final cpuInfo = File('/proc/cpuinfo').readAsStringSync();
        final hwMatch = RegExp(r'Hardware\s*:\s*(.+)').firstMatch(cpuInfo);
        if (hwMatch != null) {
          chipName = hwMatch.group(1)!.trim();
          deviceName = chipName;
        }
      } catch (_) {}
      final tier = ramGB < 3
          ? DeviceTier.minimum
          : ramGB < 6
              ? DeviceTier.low
              : ramGB < 10
                  ? DeviceTier.medium
                  : ramGB < 16
                      ? DeviceTier.high
                      : DeviceTier.ultra;
      _cached = DeviceProfile(
        identifier: 'android',
        deviceName: deviceName,
        totalRamGB: ramGB,
        chipName: chipName,
        tier: tier,
      );
      return _cached!;
    }

    String identifier;
    try {
      identifier = _readString('hw.machine');
    } catch (_) {
      identifier = Platform.operatingSystem;
    }

    final entry = _deviceDb[identifier];
    if (entry != null) {
      // For Macs, RAM is customizable; read exact hw.memsize instead of DB fallback
      double ramGB = entry.$2;
      if (identifier.startsWith('Mac') || identifier.startsWith('iMac')) {
        try {
          ramGB = _readInt64('hw.memsize') / (1024 * 1024 * 1024);
        } catch (_) {}
      }

      // Compute accurate tier for Macs since base RAM varies
      var tier = entry.$4;
      if (identifier.startsWith('Mac') || identifier.startsWith('iMac')) {
        tier =
            ramGB < 6
                ? DeviceTier.minimum
                : ramGB < 8
                ? DeviceTier.low
                : ramGB < 10
                ? DeviceTier.medium
                : ramGB < 16
                ? DeviceTier.high
                : DeviceTier.ultra;
      }

      _cached = DeviceProfile(
        identifier: identifier,
        deviceName: entry.$1,
        totalRamGB: ramGB,
        chipName: entry.$3,
        tier: tier,
      );
      return _cached!;
    }

    // Simulator or unknown device -- fall back to RAM-based tier
    double ramGB;
    try {
      ramGB = _readInt64('hw.memsize') / (1024 * 1024 * 1024);
    } catch (_) {
      ramGB = 4;
    }

    final tier =
        ramGB < 6
            ? DeviceTier.minimum
            : ramGB < 8
            ? DeviceTier.low
            : ramGB < 10
            ? DeviceTier.medium
            : ramGB < 16
            ? DeviceTier.high
            : DeviceTier.ultra;

    final isMac = identifier.startsWith('Mac') || identifier.startsWith('iMac');
    final chipName = isMac ? 'Apple Silicon' : 'Unknown';

    _cached = DeviceProfile(
      identifier: identifier,
      deviceName: isMac ? 'Mac' : identifier,
      totalRamGB: ramGB,
      chipName: chipName,
      tier: tier,
    );
    return _cached!;
  }

  // ── FFI Helpers (Apple platforms only) ──

  static String _readString(String name) {
    final sysctl = _getSysctlbyname();
    final namePtr = name.toNativeUtf8();
    final sizePtr = calloc<ffi.Size>();
    try {
      sysctl(namePtr.cast(), ffi.nullptr, sizePtr, ffi.nullptr, 0);
      final bufLen = sizePtr.value;
      if (bufLen == 0) return 'Unknown';

      final buf = calloc<ffi.Uint8>(bufLen);
      try {
        sysctl(namePtr.cast(), buf.cast(), sizePtr, ffi.nullptr, 0);
        return buf.cast<Utf8>().toDartString();
      } finally {
        calloc.free(buf);
      }
    } finally {
      calloc.free(sizePtr);
      calloc.free(namePtr);
    }
  }

  static int _readInt64(String name) {
    final sysctl = _getSysctlbyname();
    final namePtr = name.toNativeUtf8();
    final sizePtr = calloc<ffi.Size>();
    final valPtr = calloc<ffi.Int64>();
    try {
      sizePtr.value = ffi.sizeOf<ffi.Int64>();
      sysctl(namePtr.cast(), valPtr.cast(), sizePtr, ffi.nullptr, 0);
      return valPtr.value;
    } finally {
      calloc.free(valPtr);
      calloc.free(sizePtr);
      calloc.free(namePtr);
    }
  }

  // ── Device Database (27 entries: iPhone 12 through iPhone 17 series) ──

  /// hw.machine -> (Device Name, RAM GB, Chip, DeviceTier)
  static const Map<String, (String, double, String, DeviceTier)> _deviceDb = {
    // Tier policy (calibrated against 2B-param Q4_K_M + Metal):
    //   minimum → pre-A11 / iPhone 7 era (no Metal 3)
    //   low     → A11-A14 with 4 GB (Metal OK, memory-constrained)
    //   medium  → A14-A15 with 6 GB OR A15 w/ 4 GB (plenty for 2-3 B)
    //   high    → A16-A18 with 6-8 GB (handles 7 B Q4 comfortably)
    //   ultra   → A18 Pro / A19+ / Apple Silicon Mac (13 B+ runs)
    //
    // Earlier table had iPhone 13 (A15 + 4 GB Metal) at minimum which
    // was wildly wrong — it hobbled context to 1024, shortened response
    // budgets, and triggered unnecessary CPU fallbacks. A15 with Metal
    // runs Gemma 4 E2B Q4 at ~25 tok/s. Promoting to low/medium.

    // iPhone 12 series (2020) - A14 Bionic
    'iPhone13,1': ('iPhone 12 mini', 4, 'A14 Bionic', DeviceTier.low),
    'iPhone13,2': ('iPhone 12', 4, 'A14 Bionic', DeviceTier.low),
    'iPhone13,3': ('iPhone 12 Pro', 6, 'A14 Bionic', DeviceTier.medium),
    'iPhone13,4': ('iPhone 12 Pro Max', 6, 'A14 Bionic', DeviceTier.medium),

    // iPhone 13 series (2021) - A15 Bionic
    'iPhone14,4': ('iPhone 13 mini', 4, 'A15 Bionic', DeviceTier.low),
    'iPhone14,5': ('iPhone 13', 4, 'A15 Bionic', DeviceTier.low),
    'iPhone14,2': ('iPhone 13 Pro', 6, 'A15 Bionic', DeviceTier.medium),
    'iPhone14,3': ('iPhone 13 Pro Max', 6, 'A15 Bionic', DeviceTier.medium),

    // iPhone SE 3rd gen (2022) - A15 Bionic 4 GB
    'iPhone14,6': ('iPhone SE (3rd gen)', 4, 'A15 Bionic', DeviceTier.low),

    // iPhone 14 series (2022) - A15/A16 Bionic
    'iPhone14,7': ('iPhone 14', 6, 'A15 Bionic', DeviceTier.medium),
    'iPhone14,8': ('iPhone 14 Plus', 6, 'A15 Bionic', DeviceTier.medium),
    'iPhone15,2': ('iPhone 14 Pro', 6, 'A16 Bionic', DeviceTier.high),
    'iPhone15,3': ('iPhone 14 Pro Max', 6, 'A16 Bionic', DeviceTier.high),

    // iPhone 15 series (2023) - A16/A17 Pro
    'iPhone15,4': ('iPhone 15', 6, 'A16 Bionic', DeviceTier.high),
    'iPhone15,5': ('iPhone 15 Plus', 6, 'A16 Bionic', DeviceTier.high),
    'iPhone16,1': ('iPhone 15 Pro', 8, 'A17 Pro', DeviceTier.high),
    'iPhone16,2': ('iPhone 15 Pro Max', 8, 'A17 Pro', DeviceTier.high),

    // iPhone 16 series (2024) - A18/A18 Pro
    'iPhone17,3': ('iPhone 16', 8, 'A18', DeviceTier.high),
    'iPhone17,4': ('iPhone 16 Plus', 8, 'A18', DeviceTier.high),
    'iPhone17,1': ('iPhone 16 Pro', 8, 'A18 Pro', DeviceTier.ultra),
    'iPhone17,2': ('iPhone 16 Pro Max', 8, 'A18 Pro', DeviceTier.ultra),

    // iPhone 16e / SE 4th gen (2025) - A18
    'iPhone17,5': ('iPhone 16e', 8, 'A18', DeviceTier.high),

    // iPhone 17 series (2025) - A19/A19 Pro
    'iPhone18,3': ('iPhone 17', 8, 'A19', DeviceTier.ultra),
    'iPhone18,4': ('iPhone Air', 8, 'A19', DeviceTier.ultra),
    'iPhone18,1': ('iPhone 17 Pro', 12, 'A19 Pro', DeviceTier.ultra),
    'iPhone18,2': ('iPhone 17 Pro Max', 12, 'A19 Pro', DeviceTier.ultra),

    // Mac M1 series (2020)
    'MacBookAir10,1': ('MacBook Air M1', 8, 'M1', DeviceTier.high),
    'MacBookPro17,1': ('MacBook Pro M1', 8, 'M1', DeviceTier.high),
    'Macmini9,1': ('Mac mini M1', 8, 'M1', DeviceTier.high),
    'iMac21,1': ('iMac M1', 8, 'M1', DeviceTier.high),
    'iMac21,2': ('iMac M1', 8, 'M1', DeviceTier.high),

    // Mac M1 Pro/Max (2021)
    'MacBookPro18,1': (
      'MacBook Pro 16" M1 Pro/Max',
      16,
      'M1 Pro/Max/Ultra',
      DeviceTier.ultra,
    ),
    'MacBookPro18,2': (
      'MacBook Pro 16" M1 Pro/Max',
      16,
      'M1 Pro/Max/Ultra',
      DeviceTier.ultra,
    ),
    'MacBookPro18,3': (
      'MacBook Pro 14" M1 Pro/Max',
      16,
      'M1 Pro/Max/Ultra',
      DeviceTier.ultra,
    ),
    'MacBookPro18,4': (
      'MacBook Pro 14" M1 Pro/Max',
      16,
      'M1 Pro/Max/Ultra',
      DeviceTier.ultra,
    ),

    // Mac M2 series (2022)
    'Mac14,2': ('MacBook Air M2', 8, 'M2', DeviceTier.high),
    'Mac14,3': ('Mac mini M2', 8, 'M2', DeviceTier.high),
    'Mac14,5': ('MacBook Pro M2', 8, 'M2', DeviceTier.high),
    'Mac14,6': ('MacBook Pro M2 Max', 32, 'M2 Pro/Max/Ultra', DeviceTier.ultra),
    'Mac14,7': ('MacBook Pro M2', 8, 'M2', DeviceTier.high),
    'Mac14,8': ('Mac Studio M2', 32, 'M2 Pro/Max/Ultra', DeviceTier.ultra),
    'Mac14,9': ('MacBook Pro M2 Pro', 16, 'M2 Pro/Max/Ultra', DeviceTier.ultra),
    'Mac14,10': (
      'MacBook Pro M2 Max',
      32,
      'M2 Pro/Max/Ultra',
      DeviceTier.ultra,
    ),

    // Mac M3 series (2023)
    'Mac15,3': ('MacBook Pro M3', 8, 'M3', DeviceTier.high),
    'Mac15,4': ('iMac M3', 8, 'M3', DeviceTier.high),
    'Mac15,5': ('iMac M3', 8, 'M3', DeviceTier.high),
    'Mac15,6': ('MacBook Pro M3 Pro', 18, 'M3 Pro/Max/Ultra', DeviceTier.ultra),
    'Mac15,7': ('MacBook Pro M3 Max', 36, 'M3 Pro/Max/Ultra', DeviceTier.ultra),
    'Mac15,8': ('MacBook Pro M3 Max', 36, 'M3 Pro/Max/Ultra', DeviceTier.ultra),
    'Mac15,10': (
      'MacBook Pro M3 Max',
      36,
      'M3 Pro/Max/Ultra',
      DeviceTier.ultra,
    ),
    'Mac15,12': ('MacBook Air M3', 8, 'M3', DeviceTier.high),
    'Mac15,13': ('MacBook Air M3', 8, 'M3', DeviceTier.high),

    // Mac M4 series (2024)
    'Mac16,1': ('MacBook Pro M4', 16, 'M4', DeviceTier.ultra),
    'Mac16,2': ('MacBook Pro M4', 16, 'M4', DeviceTier.ultra),
    'Mac16,3': ('MacBook Pro M4', 16, 'M4', DeviceTier.ultra),
    'Mac16,5': ('MacBook Pro M4', 16, 'M4', DeviceTier.ultra),
    'Mac16,6': ('MacBook Pro M4 Pro', 24, 'M4 Pro/Max/Ultra', DeviceTier.ultra),
    'Mac16,7': ('MacBook Pro M4 Max', 36, 'M4 Pro/Max/Ultra', DeviceTier.ultra),
    'Mac16,8': ('Mac mini M4', 16, 'M4', DeviceTier.ultra),
  };

  /// Returns true on Android devices with <4 GB total RAM or <500 MB available.
  ///
  /// Used to apply aggressive inference caps (smaller context, fewer threads,
  /// shorter timeout) that keep vision inference within RAM on budget tablets.
  static Future<bool> isLowEndAndroid(TelemetryService telemetry) async {
    if (!Platform.isAndroid) return false;
    final totalMem = await telemetry.getTotalMemory();
    final availMem = await telemetry.getAvailableMemory();
    if (totalMem > 0 && totalMem < 4 * 1024 * 1024 * 1024) return true;
    if (availMem > 0 && availMem < 500 * 1024 * 1024) return true;
    return false;
  }

  @override
  String toString() =>
      'DeviceProfile($deviceName, ${totalRamGB}GB, $chipName, $tier)';
}

// ── MemoryEstimate ────────────────────────────────────────────────────────

/// Detailed memory usage estimate for a model on a specific device
class MemoryEstimate {
  final int totalMB;
  final int modelWeightsMB;
  final int kvCacheMB;
  final int metalBuffersMB;
  final int runtimeOverheadMB;

  /// Ratio of estimated memory to device safe budget (>1.0 means won't fit)
  final double memoryRatio;

  /// Whether the model fits within the device's safe memory budget
  final bool fits;

  const MemoryEstimate({
    required this.totalMB,
    required this.modelWeightsMB,
    required this.kvCacheMB,
    required this.metalBuffersMB,
    required this.runtimeOverheadMB,
    required this.memoryRatio,
    required this.fits,
  });

  @override
  String toString() =>
      'MemoryEstimate(${totalMB}MB, ratio: ${memoryRatio.toStringAsFixed(2)}, fits: $fits)';
}

// ── MemoryEstimator ───────────────────────────────────────────────────────

/// Estimates model memory usage using calibrated formulas
///
/// Calibrated against real-world Phase 19 measurements: Llama 3.2 1B Q4_K_M
/// uses 400-550MB RSS on iPhone with KV cache Q8_0 at context 2048.
class MemoryEstimator {
  /// Estimate total memory usage for a model on a device
  ///
  /// For whisper/minilm families (non-LLM): simpler formula (file size + 100MB).
  /// For LLM/VLM models: calibrated formula with 1.3x safety multiplier.
  static MemoryEstimate estimate({
    required ModelInfo model,
    required DeviceProfile device,
    int contextLength = 2048,
  }) {
    final family = model.family ?? '';

    // Non-LLM models: simpler formula
    if (family == 'whisper' || family == 'minilm') {
      final totalMB = (model.sizeBytes / (1024 * 1024) + 100).round();
      final ratio = totalMB / device.safeMemoryBudgetMB;
      return MemoryEstimate(
        totalMB: totalMB,
        modelWeightsMB: (model.sizeBytes / (1024 * 1024)).round(),
        kvCacheMB: 0,
        metalBuffersMB: 0,
        runtimeOverheadMB: 100,
        memoryRatio: ratio,
        fits: ratio <= 1.0,
      );
    }

    // LLM/VLM models.
    //
    // Previous formula used `sizeBytes * 0.15` for modelWeightsMB — off
    // by ~6x. A Q4_K_M gguf's resident memory on iOS is close to the
    // full file size (mmap pages stay hot during generation; unified
    // memory means Metal doesn't get a "copy", the pages are shared).
    // The 0.15 factor let a 1.8GB Gemma 4 E2B pass the fits check on a
    // 4GB iPhone 13 (2.4GB safe budget) — then iOS watchdog killed the
    // app for RAM overuse (Sentry 7415826426 / WatchdogTermination).
    //
    // Use 0.90 as a conservative estimate: accounts for some pages
    // being paged out under pressure, but treats the weights as
    // overwhelmingly resident. The 1.3x global multiplier on top still
    // applies.
    final modelWeightsMB = (model.sizeBytes * 0.90 / (1024 * 1024)).round();

    // KV cache quantization factor — F16/BF16/F32 weights default to
    // an unquantised KV cache, doubling its footprint vs the Q8_0 KV
    // cache used by every other quant.
    final kvQuantFactor =
        ModelAdvisor._highPrecisionKvQuants.contains(model.quantization)
            ? 2.0
            : 1.0;
    final kvCacheMB =
        ((model.parametersB ?? 1.0) *
                4.0 *
                (contextLength / 2048) *
                kvQuantFactor)
            .round();

    final metalBuffersMB = ((model.parametersB ?? 1.0) * 80).round();
    const runtimeOverheadMB = 150;

    final rawTotal =
        modelWeightsMB + kvCacheMB + metalBuffersMB + runtimeOverheadMB;
    final totalMB = (rawTotal * 1.3).round(); // 1.3x safety multiplier

    final ratio = totalMB / device.safeMemoryBudgetMB;
    return MemoryEstimate(
      totalMB: totalMB,
      modelWeightsMB: modelWeightsMB,
      kvCacheMB: kvCacheMB,
      metalBuffersMB: metalBuffersMB,
      runtimeOverheadMB: runtimeOverheadMB,
      memoryRatio: ratio,
      fits: ratio <= 1.0,
    );
  }
}

// ── ModelScore ────────────────────────────────────────────────────────────

/// Composite score for a model on a specific device and use case
class ModelScore {
  final ModelInfo model;
  final int fitScore;
  final int qualityScore;
  final int speedScore;
  final int contextScore;
  final int finalScore;
  final MemoryEstimate memoryEstimate;
  final bool fits;
  final EdgeVedaConfig recommendedConfig;
  final String? warning;

  const ModelScore({
    required this.model,
    required this.fitScore,
    required this.qualityScore,
    required this.speedScore,
    required this.contextScore,
    required this.finalScore,
    required this.memoryEstimate,
    required this.fits,
    required this.recommendedConfig,
    this.warning,
  });

  @override
  String toString() =>
      'ModelScore(${model.name}, score: $finalScore, fits: $fits)';
}

// ── ModelRecommendation ───────────────────────────────────────────────────

/// Ranked list of model recommendations for a device and use case
class ModelRecommendation {
  /// All models ranked by final score (descending), including non-fitting
  final List<ModelScore> ranked;

  /// Best fitting model (first entry where fits == true), or null
  final ModelScore? bestMatch;

  /// Device the recommendation was generated for
  final DeviceProfile device;

  /// Use case the recommendation was generated for
  final UseCase useCase;

  const ModelRecommendation({
    required this.ranked,
    this.bestMatch,
    required this.device,
    required this.useCase,
  });
}

// ── StorageCheck ──────────────────────────────────────────────────────────

/// Result of a storage availability check for a model download
class StorageCheck {
  /// Free disk space in bytes (-1 if unavailable)
  final int freeDiskBytes;

  /// Model file size in bytes
  final int requiredBytes;

  /// Whether there is sufficient free space
  final bool hasSufficientSpace;

  /// Human-readable warning message, or null if OK
  final String? warning;

  const StorageCheck({
    required this.freeDiskBytes,
    required this.requiredBytes,
    required this.hasSufficientSpace,
    this.warning,
  });

  @override
  String toString() =>
      'StorageCheck(free: ${(freeDiskBytes / (1024 * 1024)).round()}MB, '
      'required: ${(requiredBytes / (1024 * 1024)).round()}MB, ok: $hasSufficientSpace)';
}

// ── MemoryValidation ─────────────────────────────────────────────────────

/// Result of real-time memory validation after model load
class MemoryValidation {
  /// Current memory usage percentage (0.0 - 1.0)
  final double usagePercent;

  /// Whether memory pressure is detected (>80%)
  final bool isHighPressure;

  /// Whether memory usage is critical (>90%)
  final bool isCritical;

  /// Human-readable status message
  final String status;

  /// Human-readable warning, or null if healthy
  final String? warning;

  const MemoryValidation({
    required this.usagePercent,
    required this.isHighPressure,
    required this.isCritical,
    required this.status,
    this.warning,
  });

  @override
  String toString() =>
      'MemoryValidation(${(usagePercent * 100).toStringAsFixed(1)}%, '
      'status: $status)';
}

// ── ModelAdvisor ──────────────────────────────────────────────────────────

/// Device-aware model recommendation engine with 4D scoring
///
/// Scores models across four dimensions (fit, quality, speed, context)
/// weighted by use case, and generates optimal EdgeVedaConfig for each.
class ModelAdvisor {
  /// Look up the recommended draft model path for speculative decoding
  /// given a target model GGUF path (mobile-news#586 gap #10).
  ///
  /// Returns the absolute path of an on-disk draft GGUF when:
  ///   - the target's family has a known-good smaller pair (Llama →
  ///     Llama-3.2-1B, Qwen → Qwen-2.5-0.5B), AND
  ///   - the paired draft GGUF is already cached locally (the host's
  ///     ModelManager downloaded it).
  /// Returns null otherwise — the caller (typically EdgeVeda.init's
  /// auto-attach hook) should silently skip speculation.
  ///
  /// Vocabulary compatibility is enforced at attach time by the FFI
  /// layer, so the pairing rule here is purely heuristic-by-family.
  static String? recommendDraftPath(String targetGgufPath) {
    final lower = targetGgufPath.toLowerCase();

    // Family detection by filename — GGUF naming conventions are
    // stable enough across HuggingFace mirrors that this works in
    // practice for our shipping registry. False negatives just
    // disable speculation, never crash.
    final dir = targetGgufPath.contains('/')
        ? targetGgufPath.substring(0, targetGgufPath.lastIndexOf('/'))
        : targetGgufPath.contains('\\')
            ? targetGgufPath.substring(0, targetGgufPath.lastIndexOf('\\'))
            : '';
    String? draftBasename;
    if (lower.contains('llama-3.1-8b') ||
        lower.contains('llama-3.2-3b') ||
        lower.contains('llama_3.1_8b')) {
      draftBasename = 'Llama-3.2-1B-Instruct-Q4_K_M.gguf';
    } else if (lower.contains('qwen2.5-7b') ||
        lower.contains('qwen-2.5-7b') ||
        lower.contains('qwen2.5-3b')) {
      draftBasename = 'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf';
    } else if (lower.contains('mistral-nemo') ||
        lower.contains('mistral-7b')) {
      // No widely-distributed Mistral draft exists today; reuse
      // Llama-3.2-1B which has a compatible BPE in many quants —
      // attach will validate vocab and bail on mismatch, costing
      // only the model load.
      draftBasename = 'Llama-3.2-1B-Instruct-Q4_K_M.gguf';
    }

    if (draftBasename == null) return null;
    final draftPath = dir.isEmpty ? draftBasename : '$dir/$draftBasename';
    final f = File(draftPath);
    if (!f.existsSync()) return null;
    return draftPath;
  }

  // ── Scoring Constants ──

  static const _familyBaseScores = <String, int>{
    'llama3': 78,
    'mistral': 80,
    'phi3': 82,
    'gemma2': 72,
    'qwen3': 70,
    'tinyllama': 55,
    'smolvlm': 65,
    'whisper': 70,
    'minilm': 70,
  };

  static const _chipMultipliers = <String, double>{
    'A14 Bionic': 0.6,
    'A15 Bionic': 0.7,
    'A16 Bionic': 0.8,
    'A17 Pro': 1.0,
    'A18': 1.1,
    'A18 Pro': 1.2,
    'A19': 1.2,
    'A19 Pro': 1.4,
    'M1': 1.8,
    'M1 Pro/Max/Ultra': 2.2,
    'M2': 2.2,
    'M2 Pro/Max/Ultra': 2.7,
    'M3': 2.7,
    'M3 Pro/Max/Ultra': 3.2,
    'M4': 3.2,
    'M4 Pro/Max/Ultra': 3.8,
    'Apple Silicon': 2.0,
  };

  /// Fallback chip multipliers by device tier — used for Android devices
  /// where /proc/cpuinfo chip names are too varied for exact matching.
  static const _tierChipMultipliers = <DeviceTier, double>{
    DeviceTier.minimum: 0.4,
    DeviceTier.low: 0.6,
    DeviceTier.medium: 0.8,
    DeviceTier.high: 1.0,
    DeviceTier.ultra: 1.2,
  };

  /// Speed multiplier per quantization, calibrated to Q4_K_M = 1.15 on
  /// Apple Silicon. Inference on this engine is memory-bandwidth-bound,
  /// so lower-bit quants are *faster* — fewer bytes per weight to read.
  /// Quality is paid for separately via [_quantQualityPenalty].
  ///
  /// "UD-" = Unsloth Dynamic — same K-quant kernels but a per-tensor
  /// mixing recipe that promotes precision-sensitive tensors (attention
  /// output, FFN down, embeddings) to a higher bit width than the
  /// suffix implies. Slightly slower than the matching plain K-quant.
  ///
  /// Suffixes after the bit width:
  ///   _S/_M/_L     mixing tier (Small/Medium/Large) — how aggressively
  ///                sensitive tensors get bumped to a higher bit width.
  ///   _XL          Unsloth Dynamic's wide-mix tier (UD- only).
  ///   _0/_1        legacy non-K quants (no super-blocks).
  /// Unknown quants fall through to 1.0.
  static const _quantSpeedMultipliers = <String, double>{
    // Full precision
    'F32': 0.30,
    'F16': 0.60,
    'BF16': 0.60,
    // 8-bit
    'Q8_0': 0.85,
    'Q8_1': 0.85,
    'Q8_K': 0.85,
    // 6-bit
    'Q6_K': 0.95,
    // 5-bit
    'Q5_0': 1.05,
    'Q5_1': 1.05,
    'Q5_K_S': 1.10,
    'Q5_K_M': 1.05,
    // 4-bit (the everyday sweet spot)
    'Q4_0': 1.20,
    'Q4_1': 1.20,
    'Q4_K_S': 1.20,
    'Q4_K_M': 1.15,
    // 3-bit (smaller / faster; quality starts dropping)
    'Q3_K_S': 1.40,
    'Q3_K_M': 1.30,
    'Q3_K_L': 1.20,
    // 2-bit (quality cliff)
    'Q2_K': 1.50,
    // i-quants — codebook-based, similar bpw but lookup overhead
    'IQ4_NL': 1.10,
    'IQ4_XS': 1.15,
    'IQ3_M': 1.20,
    'IQ3_S': 1.25,
    'IQ3_XS': 1.30,
    'IQ3_XXS': 1.35,
    'IQ2_M': 1.30,
    'IQ2_S': 1.35,
    'IQ2_XS': 1.40,
    'IQ2_XXS': 1.45,
    'IQ1_M': 1.55,
    'IQ1_S': 1.60,
    // Unsloth Dynamic — sensitive tensors held at higher precision
    'UD-Q2_K_XL': 1.30,
    'UD-Q3_K_XL': 1.18,
    'UD-Q4_K_XL': 1.10,
    'UD-Q5_K_XL': 1.00,
    'UD-Q6_K_XL': 0.92,
    'UD-Q8_K_XL': 0.82,
    'UD-IQ1_M': 1.45,
    'UD-IQ1_S': 1.50,
    'UD-IQ2_M': 1.20,
    'UD-IQ2_XXS': 1.35,
    'UD-IQ3_XXS': 1.25,
  };

  /// Quality penalty per quantization, relative to full precision (= 0).
  /// Calibration anchors: Q4_K_M = -3 (the long-standing "sweet spot"
  /// default), Q8_0 = -1 (essentially indistinguishable from F16),
  /// Q3_K_S = -10 (real quality cliff). UD- variants land roughly two
  /// tiers above the matching plain K-quant because Unsloth's recipe
  /// keeps sensitive tensors at higher precision. Unknown quants fall
  /// through to 0 — optimistic but never penalises an unrecognised file.
  static const _quantQualityPenalty = <String, int>{
    'F32': 0,
    'F16': 0,
    'BF16': 0,
    'Q8_0': -1,
    'Q8_1': -1,
    'Q8_K': -1,
    'Q6_K': -2,
    'Q5_0': -3,
    'Q5_1': -3,
    'Q5_K_M': -2,
    'Q5_K_S': -3,
    'Q4_0': -5,
    'Q4_1': -5,
    'Q4_K_M': -3,
    'Q4_K_S': -4,
    'Q3_K_L': -5,
    'Q3_K_M': -7,
    'Q3_K_S': -10,
    'Q2_K': -15,
    'IQ4_NL': -3,
    'IQ4_XS': -4,
    'IQ3_M': -7,
    'IQ3_S': -8,
    'IQ3_XS': -9,
    'IQ3_XXS': -10,
    'IQ2_M': -12,
    'IQ2_S': -13,
    'IQ2_XS': -15,
    'IQ2_XXS': -17,
    'IQ1_M': -20,
    'IQ1_S': -22,
    'UD-Q2_K_XL': -10,
    'UD-Q3_K_XL': -5,
    'UD-Q4_K_XL': -1,
    'UD-Q5_K_XL': 0,
    'UD-Q6_K_XL': 0,
    'UD-Q8_K_XL': 0,
    'UD-IQ1_M': -15,
    'UD-IQ1_S': -17,
    'UD-IQ2_M': -8,
    'UD-IQ2_XXS': -12,
    'UD-IQ3_XXS': -7,
  };

  /// Quants whose KV cache defaults to a higher-precision format and
  /// therefore consume ~2× the cache memory. Everything else assumes
  /// llama.cpp's default Q8_0 KV cache.
  static const _highPrecisionKvQuants = <String>{'F32', 'F16', 'BF16'};

  static const _useCaseTargetContext = <UseCase, int>{
    UseCase.chat: 4096,
    UseCase.reasoning: 8192,
    UseCase.toolCalling: 4096,
    UseCase.vision: 2048,
    UseCase.stt: 512,
    UseCase.embedding: 512,
    UseCase.fast: 2048,
  };

  static const _weights = <UseCase, ({double q, double s, double f, double c})>{
    UseCase.chat: (q: 0.35, s: 0.30, f: 0.25, c: 0.10),
    UseCase.reasoning: (q: 0.50, s: 0.15, f: 0.25, c: 0.10),
    UseCase.toolCalling: (q: 0.40, s: 0.25, f: 0.25, c: 0.10),
    UseCase.vision: (q: 0.35, s: 0.25, f: 0.30, c: 0.10),
    UseCase.stt: (q: 0.30, s: 0.40, f: 0.25, c: 0.05),
    UseCase.embedding: (q: 0.25, s: 0.40, f: 0.30, c: 0.05),
    UseCase.fast: (q: 0.20, s: 0.50, f: 0.25, c: 0.05),
  };

  /// Score a single model for a device and use case
  ///
  /// Returns a [ModelScore] with per-dimension breakdown, composite score,
  /// memory estimate, and an optimal [EdgeVedaConfig].
  static ModelScore score({
    required ModelInfo model,
    required DeviceProfile device,
    required UseCase useCase,
  }) {
    final targetCtx = _useCaseTargetContext[useCase] ?? 4096;

    // ── Fit Score ──
    final estimate = MemoryEstimator.estimate(
      model: model,
      device: device,
      contextLength: targetCtx,
    );
    final memRatio = estimate.memoryRatio;
    final fitScore =
        memRatio <= 0.50
            ? 100
            : memRatio <= 0.70
            ? 85
            : memRatio <= 0.85
            ? 60
            : memRatio <= 1.00
            ? 30
            : 0;

    // ── Quality Score ──
    final baseQuality = _familyBaseScores[model.family] ?? 50;
    final paramBonus =
        min(15, (log(max(model.parametersB ?? 0.1, 0.1)) / log(2)) * 5).round();
    final quantPenalty =
        _quantQualityPenalty[model.quantization] ?? 0;
    final taskBonus =
        (model.capabilities ?? []).contains(_requiredCapability(useCase))
            ? 10
            : 0;
    final qualityScore = (baseQuality + paramBonus + quantPenalty + taskBonus)
        .clamp(0, 100);

    // ── Speed Score ──
    final baseTokPerSec = 160.0 / max(model.parametersB ?? 1.0, 0.1);
    final chipMult = _chipMultipliers[device.chipName] ??
        _tierChipMultipliers[device.tier] ??
        0.8;
    final quantMult = _quantSpeedMultipliers[model.quantization] ?? 1.0;
    final estimatedTps = baseTokPerSec * chipMult * quantMult;
    final speedScore = (estimatedTps * 2.0).round().clamp(0, 100);

    // ── Context Score ──
    final maxCtx = model.maxContextLength ?? 2048;
    final contextScore =
        maxCtx >= targetCtx * 2
            ? 100
            : maxCtx >= targetCtx
            ? 80
            : maxCtx >= targetCtx ~/ 2
            ? 50
            : 20;

    // ── Final Score ──
    final w = _weights[useCase]!;
    final finalScore =
        (qualityScore * w.q +
                speedScore * w.s +
                fitScore * w.f +
                contextScore * w.c)
            .round();

    // ── Warning ──
    String? warning;
    if (memRatio > 1.0) {
      warning = 'Exceeds device memory budget';
    } else if (memRatio > 0.85) {
      warning = 'Tight fit, may cause jetsam on heavy usage';
    }

    // ── Recommended Config ──
    final config = _recommendedConfig(model, device, useCase);

    return ModelScore(
      model: model,
      fitScore: fitScore,
      qualityScore: qualityScore,
      speedScore: speedScore,
      contextScore: contextScore,
      finalScore: finalScore,
      memoryEstimate: estimate,
      fits: estimate.fits,
      recommendedConfig: config,
      warning: warning,
    );
  }

  /// Generate ranked model recommendations for a device and use case
  ///
  /// Returns ALL models (including those that don't fit) sorted by
  /// final score descending. Non-fitting models have fits: false.
  static ModelRecommendation recommend({
    required DeviceProfile device,
    required UseCase useCase,
    List<ModelInfo>? models,
  }) {
    // Collect all unique models (exclude mmproj)
    final allModels =
        models ??
        <ModelInfo>{
          ...ModelRegistry.getAllModels(),
          ...ModelRegistry.getVisionModels(),
          ...ModelRegistry.getWhisperModels(),
          ...ModelRegistry.getEmbeddingModels(),
        }.toList();

    final scores = <ModelScore>[];
    for (final model in allModels) {
      scores.add(score(model: model, device: device, useCase: useCase));
    }

    scores.sort((a, b) => b.finalScore.compareTo(a.finalScore));

    ModelScore? bestMatch;
    for (final s in scores) {
      if (s.fits) {
        bestMatch = s;
        break;
      }
    }

    return ModelRecommendation(
      ranked: scores,
      bestMatch: bestMatch,
      device: device,
      useCase: useCase,
    );
  }

  /// Quick boolean check: can this device run a given model for a use case?
  ///
  /// Wraps [score()] and returns true if the model fits within the device's
  /// memory budget. No network or async calls -- purely local estimation.
  static bool canRun({
    required ModelInfo model,
    UseCase useCase = UseCase.chat,
    DeviceProfile? device,
  }) {
    final d = device ?? DeviceProfile.detect();
    final result = score(model: model, device: d, useCase: useCase);
    return result.fits;
  }

  /// Check if there is sufficient free disk space to download a model.
  ///
  /// Uses NSFileManager (iOS) via MethodChannel to query real free space.
  /// Includes a 100MB buffer beyond model size to account for extraction
  /// and temp files. Returns [StorageCheck] with warning if insufficient.
  ///
  /// On non-iOS platforms or when disk space query fails, returns
  /// hasSufficientSpace: true with freeDiskBytes: -1 (optimistic fallback).
  static Future<StorageCheck> checkStorageAvailability({
    required ModelInfo model,
  }) async {
    final telemetry = TelemetryService();
    final freeBytes = await telemetry.getFreeDiskSpace();

    if (freeBytes < 0) {
      // Can't determine free space -- optimistic fallback
      return StorageCheck(
        freeDiskBytes: -1,
        requiredBytes: model.sizeBytes,
        hasSufficientSpace: true,
        warning: null,
      );
    }

    // Require model size + 100MB buffer for temp files
    const bufferBytes = 100 * 1024 * 1024;
    final requiredWithBuffer = model.sizeBytes + bufferBytes;
    final sufficient = freeBytes >= requiredWithBuffer;

    String? warning;
    if (!sufficient) {
      final freeMB = (freeBytes / (1024 * 1024)).round();
      final requiredMB = (model.sizeBytes / (1024 * 1024)).round();
      warning =
          'Insufficient storage: ${freeMB}MB free, '
          '${requiredMB}MB required for ${model.name}';
    } else if (freeBytes < requiredWithBuffer * 2) {
      // Warn if space is tight (less than 2x required)
      final freeMB = (freeBytes / (1024 * 1024)).round();
      warning = 'Low storage: ${freeMB}MB free after download';
    }

    return StorageCheck(
      freeDiskBytes: freeBytes,
      requiredBytes: model.sizeBytes,
      hasSufficientSpace: sufficient,
      warning: warning,
    );
  }

  /// Validate actual memory pressure after a model has been loaded.
  ///
  /// Uses [EdgeVeda.getMemoryStats()] (routed through StreamingWorker)
  /// to read real memory usage. Call this after [EdgeVeda.init()] to
  /// verify the model loaded within safe memory bounds.
  ///
  /// Requires an initialized [EdgeVeda] instance. If no worker is active,
  /// returns healthy status (model not yet loaded = no pressure).
  static Future<MemoryValidation> validateMemoryAfterLoad(
    EdgeVeda edgeVeda,
  ) async {
    final stats = await edgeVeda.getMemoryStats();

    if (stats.currentBytes == 0) {
      return const MemoryValidation(
        usagePercent: 0,
        isHighPressure: false,
        isCritical: false,
        status: 'No model loaded',
      );
    }

    final percent = stats.usagePercent;
    final currentMB = (stats.currentBytes / (1024 * 1024)).round();
    final limitMB = (stats.limitBytes / (1024 * 1024)).round();

    String status;
    String? warning;

    if (stats.isCritical) {
      status = 'Critical';
      warning =
          'Memory usage critical: ${currentMB}MB / ${limitMB}MB '
          '(${(percent * 100).toStringAsFixed(0)}%). '
          'Risk of jetsam termination. Consider reducing context size or using a smaller model.';
    } else if (stats.isHighPressure) {
      status = 'Warning';
      warning =
          'High memory pressure: ${currentMB}MB / ${limitMB}MB '
          '(${(percent * 100).toStringAsFixed(0)}%). '
          'May experience jetsam under heavy usage.';
    } else if (percent > 0.6) {
      status = 'Moderate';
      warning =
          'Memory usage moderate: ${currentMB}MB / ${limitMB}MB. Stable for normal use.';
    } else {
      status = 'Healthy';
    }

    return MemoryValidation(
      usagePercent: percent,
      isHighPressure: stats.isHighPressure,
      isCritical: stats.isCritical,
      status: status,
      warning: warning,
    );
  }

  // ── Helpers ──

  static String _requiredCapability(UseCase useCase) {
    switch (useCase) {
      case UseCase.chat:
        return 'chat';
      case UseCase.reasoning:
        return 'reasoning';
      case UseCase.toolCalling:
        return 'tool-calling';
      case UseCase.vision:
        return 'vision';
      case UseCase.stt:
        return 'stt';
      case UseCase.embedding:
        return 'embedding';
      case UseCase.fast:
        return 'chat';
    }
  }

  static EdgeVedaConfig _recommendedConfig(
    ModelInfo model,
    DeviceProfile device,
    UseCase useCase,
  ) {
    final targetCtx = _useCaseTargetContext[useCase] ?? 4096;
    var contextLength = min(targetCtx, model.maxContextLength ?? 2048);
    if (device.tier == DeviceTier.minimum) {
      contextLength = min(contextLength, 1024);
    }
    if (device.tier == DeviceTier.low) {
      contextLength = min(contextLength, 2048);
    }
    // Adaptive thread count with thermal-safe cap.
    // Android CPU-only: big.LITTLE SoCs throttle hard under sustained load,
    // so cap at 4 threads max. Low/minimum-tier devices get 2 threads to
    // preserve headroom for OS and UI.
    // iOS/macOS: Metal GPU handles most inference; threads for prompt eval.
    final isAndroid = device.identifier == 'android';
    int threads;
    if (isAndroid) {
      threads = device.tier.index >= DeviceTier.medium.index ? 4 : 2;
    } else {
      threads = device.tier.index >= DeviceTier.high.index ? 6 : 4;
    }
    // Android is CPU-only (no Metal); use 50% memory budget.
    // iOS/macOS have Metal GPU; use 60% memory budget.
    final memoryRatio = isAndroid ? 0.50 : 0.60;
    final maxMemoryMb = (device.totalRamGB * 1024 * memoryRatio).round();

    return EdgeVedaConfig(
      modelPath: '',
      numThreads: threads,
      contextLength: contextLength,
      useGpu: !isAndroid,
      maxMemoryMb: maxMemoryMb,
      kvCacheTypeK: 8,
      kvCacheTypeV: 8,
      flashAttn: -1,
    );
  }
}
