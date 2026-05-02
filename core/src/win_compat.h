// Tiny Windows compatibility shim for the POSIX timing calls used
// in vision_engine.cpp / whisper_engine.cpp.
//
// MSVC's UCRT (`<time.h>`) ships `struct timespec` since Windows 10
// SDK 10240 / VS 2015 Update 3, but does NOT provide
// `clock_gettime` or `CLOCK_MONOTONIC`. Rely on the system struct
// (so we don't trigger C2011 redefinition); just supply the missing
// macro and function. Maps to QueryPerformanceCounter — the
// documented monotonic-clock equivalent (sub-100ns resolution,
// monotonic across CPU sleep states).
//
// MinGW / Cygwin already provide clock_gettime, so they fall through
// to the system <time.h> only.

#ifndef EDGE_VEDA_WIN_COMPAT_H_
#define EDGE_VEDA_WIN_COMPAT_H_

#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__)

#include <time.h>      // struct timespec from UCRT
#include <windows.h>

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

static inline int clock_gettime(int /*clk_id*/, struct timespec* ts) {
  static LARGE_INTEGER s_freq = {0};
  if (s_freq.QuadPart == 0) {
    QueryPerformanceFrequency(&s_freq);
  }
  LARGE_INTEGER count;
  QueryPerformanceCounter(&count);
  ts->tv_sec  = (time_t)(count.QuadPart / s_freq.QuadPart);
  ts->tv_nsec = (long)(((count.QuadPart % s_freq.QuadPart) * 1000000000LL) /
                       s_freq.QuadPart);
  return 0;
}

#else

#include <time.h>

#endif

#endif  // EDGE_VEDA_WIN_COMPAT_H_
