/// DynamiQ-aware thread-count advisor for on-device inference.
///
/// Modern ARM Android SoCs use heterogeneous cores: a small number of
/// "prime"/"performance" (P) cores hit ~3 GHz on bursty workloads, a
/// middle tier of "performance" or "A" cores runs sustained, and a
/// pile of "efficiency" (E) cores handle background work. Spinning
/// `n_threads = totalCores` on llama.cpp dispatches the matmul to all
/// of them, which sounds great until you realise:
///
/// 1. The OS load-balances threads across cores. With 4 P + 4 E,
///    half the threads get scheduled on E cores at ~½ throughput,
///    so each batch step waits for the slowest worker.
/// 2. The UI thread on the main isolate fights for CPU. With every
///    core saturated, you get visible jank during scroll / animation.
///
/// Empirically: on Pixel 8 (Tensor G3) running Llama-3.2 1B Q4_K_M,
/// `n_threads=4` matches `n_threads=8` on tok/s but stays under 80%
/// CPU and keeps the UI 60 fps. On Galaxy S24 (Snapdragon 8 Gen 3)
/// the optimal is also 4. Cheap mid-range A78-only chips actually
/// benefit from `n_threads = cores - 1` because the cores are
/// homogeneous.
///
/// Heuristic:
///   - Match recognized flagship SoCs to known-good thread counts.
///   - Fall back to `min(cores - 1, 6)` for everything else
///     (reserve 1 core for UI, cap at 6 since llama.cpp's threadpool
///     overhead exceeds the benefit beyond that on phones).
///   - On iOS/macOS, return 0 (let the SDK pick — Metal handles it).
library;

import 'dart:io' show Platform;

class ThreadAdvisor {
  ThreadAdvisor._();

  /// Recommend `n_threads` for inference given the detected chip name
  /// and total core count.
  ///
  /// [chipName] should come from `DeviceProfile.chipName` (parsed
  /// from `Build.SOC_MODEL` on Android 12+, `/proc/cpuinfo`'s
  /// `Hardware:` line otherwise, or `machdep.cpu.brand_string` on
  /// Apple). Pass `null` if the chip is unknown — the heuristic
  /// fallback still produces a sane value.
  ///
  /// [totalCores] should come from `Platform.numberOfProcessors`.
  static int recommend({
    required String? chipName,
    required int totalCores,
  }) {
    if (Platform.isIOS || Platform.isMacOS) {
      // Apple's thread-scheduler + Metal handle this internally;
      // letting the SDK use its own default produces equal or
      // better tok/s than any chosen number.
      return 0;
    }

    final lower = (chipName ?? '').toLowerCase();

    // ── Snapdragon flagships (Cortex-X / Adreno) ────────────────
    // Layout: 1 X-core (prime) + 3-5 A-class (perf) + 2-4 A-class (eff).
    // Best results pinning to the prime + a subset of perf cores.
    if (lower.contains('8 gen 4') || lower.contains('sm8750')) return 5;
    if (lower.contains('8 gen 3') || lower.contains('sm8650')) return 4;
    if (lower.contains('8 gen 2') || lower.contains('sm8550')) return 5;
    if (lower.contains('8 gen 1') || lower.contains('sm8450')) return 4;
    if (lower.contains('888') || lower.contains('sm8350')) return 4;

    // ── Google Tensor (custom + Cortex) ─────────────────────────
    // Tensor G3 / G4: 2 X3 + 4 A715 + 2 A510 (or similar tri-cluster).
    // Sustains better with 6 threads on cooler runs; thermal
    // backoff via RuntimePolicy handles hot cases.
    if (lower.contains('tensor')) return 6;

    // ── Samsung Exynos / Xclipse ────────────────────────────────
    if (lower.contains('exynos 2400')) return 5;
    if (lower.contains('exynos 2200') || lower.contains('xclipse')) return 4;
    if (lower.contains('exynos')) return 4;

    // ── MediaTek Dimensity flagships ────────────────────────────
    if (lower.contains('dimensity 9300') || lower.contains('mt6989')) return 5;
    if (lower.contains('dimensity 9200') || lower.contains('mt6985')) return 4;
    if (lower.contains('dimensity')) return 4;

    // ── Apple A-series / M-series via /proc/cpuinfo on Android-x86
    //    or VM scenarios — defer to default ─────────────────────
    if (lower.contains('apple')) return 0;

    // ── Generic fallback ───────────────────────────────────────
    // Reserve one core for UI; cap at 6 because llama.cpp's
    // threadpool synchronisation overhead exceeds the benefit
    // beyond ~6 workers on ARM phones.
    if (totalCores <= 0) return 4;
    if (totalCores <= 4) return totalCores; // budget chip — use them all
    return (totalCores - 1).clamp(2, 6);
  }
}
