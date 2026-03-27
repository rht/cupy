#ifndef INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H
#define INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H

#ifdef __HIP_DEVICE_COMPILE__

// As per the comment below, use workaround conditionally:
// https://github.com/ROCm/clr/blob/68147fe9b20a72aa43e7898bdd9ba39bca4afd14/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h#L25
#if (HIP_VERSION < 60200000) || defined(HIP_DISABLE_WARP_SYNC_BUILTINS)

// ignore mask
#define __shfl_sync(mask, ...) __shfl(__VA_ARGS__)
#define __shfl_up_sync(mask, ...) __shfl_up(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...) __shfl_xor(__VA_ARGS__)

#else  // HIP >= 6.2 with warp sync builtins

// HIP's __shfl_*_sync require a 64-bit mask, but CUDA uses 32-bit.
// Define wrapper functions that cast the mask, then redirect via macros.
// The wrappers are defined BEFORE the macros so their bodies call the
// real HIP functions (the preprocessor is single-pass top-to-bottom).
template <typename T>
__device__ inline T cupy_shfl_sync(unsigned int mask, T var, int srcLane, int width = warpSize) {
    return __shfl_sync(static_cast<unsigned long long>(mask), var, srcLane, width);
}
template <typename T>
__device__ inline T cupy_shfl_up_sync(unsigned int mask, T var, unsigned int delta, int width = warpSize) {
    return __shfl_up_sync(static_cast<unsigned long long>(mask), var, delta, width);
}
template <typename T>
__device__ inline T cupy_shfl_down_sync(unsigned int mask, T var, unsigned int delta, int width = warpSize) {
    return __shfl_down_sync(static_cast<unsigned long long>(mask), var, delta, width);
}
template <typename T>
__device__ inline T cupy_shfl_xor_sync(unsigned int mask, T var, int laneMask, int width = warpSize) {
    return __shfl_xor_sync(static_cast<unsigned long long>(mask), var, laneMask, width);
}

#define __shfl_sync(mask, ...) cupy_shfl_sync(static_cast<unsigned int>(mask), __VA_ARGS__)
#define __shfl_up_sync(mask, ...) cupy_shfl_up_sync(static_cast<unsigned int>(mask), __VA_ARGS__)
#define __shfl_down_sync(mask, ...) cupy_shfl_down_sync(static_cast<unsigned int>(mask), __VA_ARGS__)
#define __shfl_xor_sync(mask, ...) cupy_shfl_xor_sync(static_cast<unsigned int>(mask), __VA_ARGS__)

#endif  // (HIP_VERSION < 60200000) || defined(HIP_DISABLE_WARP_SYNC_BUILTINS)

// In ROCm, threads in a warp march in lock-step, so we don't need to
// synchronize the threads. But it doesn't guarantee the memory order,
// which still make us use memory fences.
// https://rocmdocs.amd.com/en/latest/Programming_Guides/Kernel_language.html#warp-cross-lane-functions
#define __syncwarp() { __threadfence_block(); }

#endif  // __HIP_DEVICE_COMPILE__

#endif  // INCLUDE_GUARD_CUPY_HIP_WORKAROUND_H
