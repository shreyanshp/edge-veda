// Tiny Windows compatibility shim for the POSIX timing calls used
// in vision_engine.cpp / whisper_engine.cpp / image_engine.cpp.
//
// Headers under MinGW/Cygwin already provide clock_gettime —
// MSVC + WinSDK do not. Map to QueryPerformanceCounter, which is
// the documented monotonic-clock equivalent on Windows (sub-100ns
// resolution, monotonic across CPU sleep states).

#ifndef EDGE_VEDA_WIN_COMPAT_H_
#define EDGE_VEDA_WIN_COMPAT_H_

#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__)

#include <windows.h>

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

#ifndef _STRUCT_TIMESPEC
#define _STRUCT_TIMESPEC
struct timespec {
  long long tv_sec;
  long long tv_nsec;
};
#endif

static inline int clock_gettime(int /*clk_id*/, struct timespec* ts) {
  static LARGE_INTEGER s_freq = {0};
  if (s_freq.QuadPart == 0) {
    QueryPerformanceFrequency(&s_freq);
  }
  LARGE_INTEGER count;
  QueryPerformanceCounter(&count);
  ts->tv_sec  = count.QuadPart / s_freq.QuadPart;
  ts->tv_nsec = ((count.QuadPart % s_freq.QuadPart) * 1000000000LL) /
                  s_freq.QuadPart;
  return 0;
}

#else

#include <time.h>

#endif

#endif  // EDGE_VEDA_WIN_COMPAT_H_
