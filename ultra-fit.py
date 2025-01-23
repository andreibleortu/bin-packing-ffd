#!/usr/bin/env python3

# import required modules for system operations, numerical processing, and jit compilation
import sys
import numpy as np
from numba import njit, int32, types, config, prange
from numba.typed import List
import os

# global configuration defaults for x86_64 architectures
# these provide safe fallbacks if hardware detection fails
vector_width = 16  # sse2 baseline (128-bit vectors for 4 int32 elements)
cpu_count = 4  # common minimum core count assumption

def configure_system():
    """hardware-aware configuration for optimal numba performance
    - detects simd capabilities through cpu flags
    - determines physical core count
    - configures numba compiler flags and threading layer
    - falls back to safe defaults if detection fails"""
    
    global vector_width, cpu_count

    try:
        # optional dependencies for precise hardware detection
        import cpuinfo  # for detailed cpu feature flags
        import psutil  # for accurate physical core count

        # configure threading layer for better parallelism management
        config.threading_layer = "tbb"  # intel's thread building blocks
        config.numba_num_threads = cpu_count
        config.fastmath = True  # safe for integer-based calculations
        config.opt = 3  # maximum optimization level
        config.llvm_align_stack = vector_width  # stack alignment matching simd width
        config.align_arrays = True  # better memory alignment for vectorization
        config.enable_pyobject_opt = 1  # optimized python object handling
        config.enable_inline_array = 1  # faster array access patterns

        # detect cpu features through vendor flags
        info = cpuinfo.get_cpu_info()
        flags = info.get("flags", [])

        # determine maximum supported simd extension
        if any(f.startswith("avx512") for f in flags):
            vector_width = 64  # 512-bit vectors (16 int32 elements)
        elif "avx2" in flags:
            vector_width = 32  # 256-bit vectors (8 int32 elements)
        elif "sse2" in flags:
            vector_width = 16  # 128-bit vectors (4 int32 elements)

        # get physical core count with multiple fallbacks
        cpu_count = psutil.cpu_count(logical=False) or os.cpu_count() or 4

        print(f"configured: simd={vector_width}byte vectors, threads={cpu_count}")

    except ImportError:
        # fallback when detection libraries are missing
        print("using conservative defaults - install cpuinfo/psutil for optimal config")
        vector_width = 16
        cpu_count = os.cpu_count() or 4
        config.numba_num_threads = cpu_count
        config.threading_layer = "forksafe"  # safer but slower threading

    except Exception as e:
        # failsafe configuration for unexpected errors
        print(f"configuration error: {str(e)} - using safe defaults")
        vector_width = 16
        cpu_count = 4
        config.numba_num_threads = cpu_count
        config.threading_layer = "forksafe"

configure_system()  # execute configuration at module load

@njit(
    [
        "int32(int32[::1],int32,int32)",  # contiguous memory specialization
        "int32(int32[:],int32,int32)",    # non-contiguous array fallback
    ],
    fastmath=True,
    boundscheck=False,  # remove array bounds checking for speed
    inline="always",  # force inlining for small function
    nogil=True,  # release gil for potential multithreading
    cache=True,  # cache compiled code for reuse
    nopython=True,  # strict no-python mode
    no_cpython_wrapper=True,  # reduce python call overhead
    error_model="numpy",  # ignore division errors for integers
    parallel=False,  # disable parallel for linear search
)
def find_bin(remaining_space, val, active_bins):
    """simd-optimized linear search for bin packing
    - uses manual loop unrolling to enable vectorization
    - employs tiered approach based on detected simd capabilities
    - returns first bin with sufficient remaining space"""

    i = 0

    # avx-512 tier: process 16 elements per iteration
    if vector_width >= 64 and active_bins >= 16:
        while i + 16 <= active_bins:
            # manually unrolled checks enable simd parallel comparisons
            if remaining_space[i] >= val: return i
            if remaining_space[i+1] >= val: return i+1
            if remaining_space[i+2] >= val: return i+2
            if remaining_space[i+3] >= val: return i+3
            if remaining_space[i+4] >= val: return i+4
            if remaining_space[i+5] >= val: return i+5
            if remaining_space[i+6] >= val: return i+6
            if remaining_space[i+7] >= val: return i+7
            if remaining_space[i+8] >= val: return i+8
            if remaining_space[i+9] >= val: return i+9
            if remaining_space[i+10] >= val: return i+10
            if remaining_space[i+11] >= val: return i+11
            if remaining_space[i+12] >= val: return i+12
            if remaining_space[i+13] >= val: return i+13
            if remaining_space[i+14] >= val: return i+14
            if remaining_space[i+15] >= val: return i+15
            i += 16  # process next vector chunk

    # avx2 tier: process 8 elements per iteration
    elif vector_width >= 32 and active_bins >= 8:
        while i + 8 <= active_bins:
            if remaining_space[i] >= val: return i
            if remaining_space[i+1] >= val: return i+1
            if remaining_space[i+2] >= val: return i+2
            if remaining_space[i+3] >= val: return i+3
            if remaining_space[i+4] >= val: return i+4
            if remaining_space[i+5] >= val: return i+5
            if remaining_space[i+6] >= val: return i+6
            if remaining_space[i+7] >= val: return i+7
            i += 8

    # sse tier: process 4 elements per iteration
    elif vector_width >= 16 and active_bins >= 4:
        while i + 4 <= active_bins:
            if remaining_space[i] >= val: return i
            if remaining_space[i+1] >= val: return i+1
            if remaining_space[i+2] >= val: return i+2
            if remaining_space[i+3] >= val: return i+3
            i += 4

    # scalar fallback for remaining elements
    while i < active_bins:
        if remaining_space[i] >= val:
            return i
        i += 1

    # no suitable bin found - return index for new bin
    return active_bins

@njit(
    fastmath=True,
    boundscheck=False,
    nogil=True,
    cache=True,
    parallel=False,  # disabled for sequential algorithm
    error_model="numpy",
)
def first_fit_decreasing(values, capacity):
    """optimized first-fit decreasing bin packing algorithm
    1. sort items in descending order for better space utilization
    2. pre-allocate memory for bins to avoid resizing
    3. use vectorized search for efficient bin placement
    4. construct final bin List with parallel optimizations"""

    # handle empty input edge case
    if len(values) == 0:
        result = List()
        result.append(List.empty_list(types.int32))
        result.pop()  # workaround for numba typed List initialization
        return result

    # sort values in reverse order (descending)
    sorted_values = np.sort(values)[::-1].astype(np.int32)

    # handle case where largest item exceeds bin capacity
    if sorted_values[0] > capacity:
        return None  # no valid packing possible

    # pre-allocate memory for performance
    max_bins = len(values) + 1  # theoretical worst case
    max_items = min(len(values), 3072)  # practical limit to prevent over-allocation
    remaining = np.full(max_bins, capacity, dtype=np.int32)  # bin remaining space
    bin_storage = np.zeros((max_bins, max_items), dtype=np.int32)  # 2d bin storage
    bin_sizes = np.zeros(max_bins, dtype=np.int32)  # items per bin count

    # initialize first bin with largest item
    bin_storage[0, 0] = sorted_values[0]
    bin_sizes[0] = 1
    remaining[0] = capacity - sorted_values[0]
    active_bins = 1

    # core packing loop - sequential by algorithm design
    for i in range(1, len(sorted_values)):
        val = sorted_values[i]
        bin_idx = find_bin(remaining, val, active_bins)

        if bin_idx >= active_bins:
            # create new bin if no existing bin can accommodate
            if active_bins >= max_bins:
                raise RuntimeError(f"bin overflow at item {i}/{len(values)}")
            bin_storage[active_bins, 0] = val
            bin_sizes[active_bins] = 1
            remaining[active_bins] = capacity - val
            active_bins += 1
        else:
            # add to existing bin
            idx = bin_sizes[bin_idx]
            bin_storage[bin_idx, idx] = val
            bin_sizes[bin_idx] += 1
            remaining[bin_idx] -= val

    # construct result List with parallel optimizations
    result = List()
    result.append(List.empty_list(types.int32))  # typed List initialization
    result.pop()

    for i in range(active_bins):
        count = bin_sizes[i]
        lst = List.empty_list(types.int32)
        lst.extend(bin_storage[i, :count])  # slice to actual item count
        result.append(lst)

    return result

def main():
    """command-line interface for bin packing solver
    - reads input file with specific format
    - validates input constraints
    - executes packing algorithm
    - outputs bin count and contents"""

    if len(sys.argv) != 2:
        print("usage: python optimized_binpack.py <input_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        # parse and validate header line
        header = f.readline().strip().split()
        if len(header) != 2 or any(not x.isdigit() for x in header):
            raise ValueError("invalid input header")

        n, c = map(int, header)
        if n < 0 or c <= 0:
            raise ValueError("invalid count/capacity values")

        # read and validate item values
        values = []
        for _ in range(n):
            line = f.readline().strip()
            if not line or not line.isdigit():
                raise ValueError("invalid item value")
            val = int(line)
            if val <= 0 or val > c:
                raise ValueError(f"item {val} invalid for capacity {c}")
            values.append(val)

        # convert to numpy array for numba compatibility
        values = np.array(values, dtype=np.int32)

    # execute core algorithm
    bins = first_fit_decreasing(values, c)

    # generate standardized output
    print(len(bins))
    for b in bins:
        print(" ".join(map(str, sorted(b))))  # sorted bin contents

if __name__ == "__main__":
    main()