/// DeviceTier-driven inference configuration
///
/// Centralizes timeout, resolution, and degradation decisions
/// that were previously scattered as hardcoded platform checks.
library;

import 'dart:io' show Platform;

import 'model_advisor.dart' show DeviceProfile, DeviceTier;
import 'telemetry_service.dart' show TelemetryService;

/// Maps [DeviceTier] to inference parameters (timeouts, resolution).
///
/// Replaces scattered `Platform.isAndroid ? X : Y` checks with
/// tier-aware decisions so a Pixel 9 Pro (high) and a budget
/// tablet (minimum) get appropriately different limits.
class InferenceConfig {
  InferenceConfig._();

  /// LLM first-token timeout (covers prefill + first decode).
  ///
  /// Bumped high/ultra tiers after observing TimeoutException on
  /// iPhone 17 Pro sim running Qwen3 0.6B — `<think>` prefill plus
  /// cold-worker startup routinely exceeds 20–30s even on fast
  /// hardware without the generation actually being stuck.
  static Duration llmTokenTimeout(DeviceTier tier) {
    return switch (tier) {
      DeviceTier.minimum => const Duration(seconds: 180),
      DeviceTier.low => const Duration(seconds: 120),
      DeviceTier.medium => const Duration(seconds: 90),
      DeviceTier.high => const Duration(seconds: 60),
      DeviceTier.ultra => const Duration(seconds: 45),
    };
  }

  /// Vision inference timeout (covers image encoding + decode).
  static Duration visionTimeout(DeviceTier tier) {
    return switch (tier) {
      DeviceTier.minimum => const Duration(seconds: 90),
      DeviceTier.low => const Duration(seconds: 120),
      DeviceTier.medium => const Duration(seconds: 180),
      DeviceTier.high => const Duration(seconds: 60),
      DeviceTier.ultra => const Duration(seconds: 45),
    };
  }

  /// Whether GPU acceleration is available on the current device.
  ///
  /// iOS/macOS always have Metal. Android checks for Vulkan via
  /// TelemetryService MethodChannel. Falls back to false on error.
  static Future<bool> useGpu([TelemetryService? telemetry]) async {
    if (Platform.isIOS || Platform.isMacOS) return true;
    if (Platform.isAndroid) {
      final ts = telemetry ?? TelemetryService();
      final backend = await ts.getGpuBackend();
      return backend != 'CPU';
    }
    return false;
  }

  /// Max pixels on the longest edge before downscaling for vision inference.
  static int maxInferenceDimension(DeviceTier tier) {
    return switch (tier) {
      DeviceTier.minimum => 384,
      DeviceTier.low => 512,
      DeviceTier.medium => 640,
      DeviceTier.high => 768,
      DeviceTier.ultra => 1024,
    };
  }

  /// Recommended `n_batch` (logical batch size) for `EdgeVedaConfig`.
  ///
  /// Smaller batches → less peak memory but more kernel launches.
  /// Larger batches → faster prefill, amortizes GPU/CPU dispatch
  /// overhead, but bigger working set. The defaults below are tuned
  /// for the resident-RAM ranges in [DeviceTier]:
  ///
  /// - minimum (<3 GB): 128 — safe even with browser/social apps
  ///   open in background. Prefill latency is what the user feels.
  /// - low (3–6 GB):    256
  /// - medium (6–10 GB): 512 (matches llama.cpp's default)
  /// - high (10–16 GB):  1024
  /// - ultra (>=16 GB):  2048
  static int recommendedBatch(DeviceTier tier) {
    return switch (tier) {
      DeviceTier.minimum => 128,
      DeviceTier.low => 256,
      DeviceTier.medium => 512,
      DeviceTier.high => 1024,
      DeviceTier.ultra => 2048,
    };
  }

  /// Recommended `n_ubatch` (physical micro-batch). One quarter of
  /// `n_batch` is conservative — keeps peak working memory low
  /// without losing the prefill amortization benefit.
  static int recommendedUbatch(DeviceTier tier) =>
      recommendedBatch(tier) ~/ 4;
}

/// Tracks consecutive vision inference failures and degrades resolution.
///
/// Used by vision UIs (e.g. VisionScreen) to automatically lower resolution
/// when repeated timeouts indicate the device cannot keep up at the current
/// resolution. Resolution recovers after sustained successes.
class AdaptiveVisionConfig {
  int _consecutiveFailures = 0;
  int _consecutiveSuccesses = 0;
  int _currentMaxDimension;
  final int _initialMaxDimension;

  /// Create with a starting max dimension (typically from
  /// [InferenceConfig.maxInferenceDimension]).
  AdaptiveVisionConfig(int maxDimension)
    : _currentMaxDimension = maxDimension,
      _initialMaxDimension = maxDimension;

  /// Convenience: create from detected device tier.
  factory AdaptiveVisionConfig.fromTier(DeviceTier tier) {
    return AdaptiveVisionConfig(InferenceConfig.maxInferenceDimension(tier));
  }

  /// Convenience: create from auto-detected device profile.
  factory AdaptiveVisionConfig.auto() {
    return AdaptiveVisionConfig.fromTier(DeviceProfile.detect().tier);
  }

  /// Current max dimension (may have degraded from initial).
  int get maxDimension => _currentMaxDimension;

  /// Whether resolution has been degraded below initial.
  bool get isDegraded => _currentMaxDimension < _initialMaxDimension;

  /// Record a successful inference. Resets failure counter;
  /// after 5 consecutive successes, attempts to restore one resolution step.
  void recordSuccess() {
    _consecutiveFailures = 0;
    _consecutiveSuccesses++;
    if (_consecutiveSuccesses >= 5 &&
        _currentMaxDimension < _initialMaxDimension) {
      _currentMaxDimension = (_currentMaxDimension * 1.25).round().clamp(
        256,
        _initialMaxDimension,
      );
      _consecutiveSuccesses = 0;
    }
  }

  /// Record a timeout/failure. After 2 consecutive failures,
  /// degrades resolution by 25%.
  void recordTimeout() {
    _consecutiveFailures++;
    _consecutiveSuccesses = 0;
    if (_consecutiveFailures >= 2) {
      _currentMaxDimension = (_currentMaxDimension * 0.75).round().clamp(
        256,
        1024,
      );
      _consecutiveFailures = 0;
    }
  }
}
