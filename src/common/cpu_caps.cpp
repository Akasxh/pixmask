#include "common/cpu_caps.h"

#include <algorithm>
#include <mutex>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

#if defined(__linux__)
#include <sys/auxv.h>
#if defined(__arm__) || defined(__aarch64__)
#include <asm/hwcap.h>
#endif
#endif

namespace {

struct Caps {
    bool avx2 = false;
    bool neon = false;
    std::size_t threads = 1;
};

Caps detect_caps() {
    Caps caps{};
    const unsigned int hw = std::thread::hardware_concurrency();
    caps.threads = std::max<std::size_t>(1, static_cast<std::size_t>(hw));

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    bool avx2 = false;
#if defined(_MSC_VER)
    int regs[4] = {0};
    __cpuid(regs, 0);
    if (regs[0] >= 7) {
        __cpuidex(regs, 7, 0);
        avx2 = (regs[1] & (1 << 5)) != 0;
    }
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int max_leaf = __get_cpuid_max(0, nullptr);
    if (max_leaf >= 7) {
        unsigned int a = 0, b = 0, c = 0, d = 0;
        __cpuid_count(7, 0, a, b, c, d);
        avx2 = (b & (1u << 5)) != 0;
    }
#endif
    caps.avx2 = avx2;
#else
    caps.avx2 = false;
#endif

#if defined(__linux__) && (defined(__arm__) || defined(__aarch64__))
    unsigned long hwcap = getauxval(AT_HWCAP);
#if defined(__aarch64__)
    caps.neon = (hwcap & HWCAP_ASIMD) != 0;
#else
    caps.neon = (hwcap & HWCAP_NEON) != 0;
#endif
#elif defined(__APPLE__) && (defined(__arm__) || defined(__aarch64__))
    caps.neon = true;
#elif defined(_WIN32) && (defined(_M_ARM64) || defined(_M_ARM))
    caps.neon = true;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    caps.neon = true;
#else
    caps.neon = false;
#endif

    return caps;
}

Caps& cached_caps() {
    static Caps caps;
    static std::once_flag init_flag;
    std::call_once(init_flag, []() { caps = detect_caps(); });
    return caps;
}

} // namespace

namespace pixmask {

bool has_avx2() noexcept {
    return cached_caps().avx2;
}

bool has_neon() noexcept {
    return cached_caps().neon;
}

std::size_t hw_threads() noexcept {
    return cached_caps().threads;
}

} // namespace pixmask
