#!/usr/bin/env python3

import sys
import numpy as np
from numba import njit, int32, int64, types, config, prange
from numba.typed import List
import os

# Global configuration defaults
VECTOR_WIDTH = 16  # SSE2 baseline
CPU_COUNT = 4

def configure_system():
    """Enhanced hardware configuration with better error handling and thread validation"""
    global VECTOR_WIDTH, CPU_COUNT
    
    try:
        import cpuinfo
        import psutil
        
        config.THREADING_LAYER = 'tbb'
        config.FASTMATH = True
        config.OPT = 3

        # SIMD detection
        info = cpuinfo.get_cpu_info()
        flags = info.get('flags', [])
        
        if any(f.startswith('avx512') for f in flags):
            VECTOR_WIDTH = 64
        elif 'avx2' in flags:
            VECTOR_WIDTH = 32
        elif 'sse2' in flags:
            VECTOR_WIDTH = 16

        # Core count detection
        CPU_COUNT = psutil.cpu_count(logical=False) or os.cpu_count() or 4
        
        config.NUMBA_NUM_THREADS = CPU_COUNT

        # Validate threading layer
        from numba import vectorize
        @vectorize(['int32(int32,int32)'], target='parallel')
        def _test(x, y): return x + y
        _test(np.array([1], dtype=np.int32), np.array([1], dtype=np.int32))
        
        print(f"Configured: SIMD={VECTOR_WIDTH}byte vectors, Threads={CPU_COUNT}")
        
    except ImportError as e:
        print(f"Missing dependency: {str(e)} - using defaults")
        VECTOR_WIDTH = 16
        CPU_COUNT = os.cpu_count() or 4
        config.NUMBA_NUM_THREADS = CPU_COUNT
        config.THREADING_LAYER = 'sequential'

    except Exception as e:
        print(f"Config error: {str(e)} - using safe defaults")
        VECTOR_WIDTH = 16
        CPU_COUNT = 4
        config.NUMBA_NUM_THREADS = CPU_COUNT
        config.THREADING_LAYER = 'sequential'

@njit(['int64(int32[::1],int32,int64)'],
      fastmath=True, boundscheck=False, inline='always')
def find_bin(remaining_space, val, active_bins):
    """SIMD-optimized with explicit loop unrolling"""
    i = 0
    
    # AVX-512: 64-byte chunks (16 elements)
    if VECTOR_WIDTH >= 64:
        while i + 16 <= active_bins:
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
            i += 16

    # AVX2: 32-byte chunks (8 elements)
    elif VECTOR_WIDTH >= 32:
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

    # SSE: 16-byte chunks (4 elements)
    elif VECTOR_WIDTH >= 16:
        while i + 4 <= active_bins:
            if remaining_space[i] >= val: return i
            if remaining_space[i+1] >= val: return i+1
            if remaining_space[i+2] >= val: return i+2
            if remaining_space[i+3] >= val: return i+3
            i += 4

    # Scalar fallback with cache-line optimization
    while i < active_bins:
        limit = min(i + 4, active_bins)
        while i < limit:
            if remaining_space[i] >= val: return i
            i += 1

    return active_bins

@njit(
    fastmath=True,
    boundscheck=False,
    nogil=True,
    cache=True,
    parallel=True
)
def first_fit_decreasing(values, capacity):
    """Optimized with dynamic bin storage and parallel post-processing"""
    if len(values) == 0:
        empty = List()
        empty.append(List.empty_list(types.int32))
        empty.pop()
        return empty

    # Sort items in descending order (consider parallel sort for large datasets)
    sorted_values = np.sort(values)[::-1].astype(np.int32)

    if sorted_values[0] > capacity:
        outer = List()
        outer.append(List.empty_list(types.int32))
        outer.pop()
        return outer


    max_bins = len(values)
    remaining = np.full(max_bins, capacity, dtype=np.int32)
    bin_contents = List()
    bin_lengths = np.zeros(max_bins, dtype=np.int64)
    CACHE_LINE_ITEMS = 16

    # Initialize first bin
    first_bin = np.zeros(CACHE_LINE_ITEMS, dtype=np.int32)
    first_bin[0] = sorted_values[0]
    bin_contents.append(first_bin)
    remaining[0] = capacity - sorted_values[0]
    bin_lengths[0] = 1
    active_bins = 1

    # Main packing loop
    for i in range(1, len(sorted_values)):
        val = sorted_values[i]
        bin_idx = find_bin(remaining, val, np.int64(active_bins))

        if bin_idx >= active_bins:
            new_bin = np.zeros(CACHE_LINE_ITEMS, dtype=np.int32)
            new_bin[0] = val
            bin_contents.append(new_bin)
            remaining[active_bins] = capacity - val
            bin_lengths[active_bins] = 1
            active_bins += 1
        else:
            remaining[bin_idx] -= val
            current_bin = bin_contents[bin_idx]
            current_len = bin_lengths[bin_idx]

            if current_len == current_bin.size:
                new_size = current_bin.size * 2
                new_bin = np.zeros(new_size, dtype=np.int32)
                new_bin[:current_len] = current_bin[:current_len]
                bin_contents[bin_idx] = new_bin
                current_bin = new_bin

            current_bin[current_len] = val
            bin_lengths[bin_idx] += 1

    # Parallel result construction
    result = List()
    for _ in range(active_bins):
        result.append(List.empty_list(types.int32))

    for i in prange(active_bins):
        count = bin_lengths[i]
        bin_array = bin_contents[i]
        typed_list = List.empty_list(types.int32)
        for j in range(int(count)):  # Ensure count is treated as integer
            typed_list.append(bin_array[j])
        result[i] = typed_list

    return result

def main():
    if len(sys.argv) != 2:
        print("Usage: python optimized_binpack.py <input_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        header = f.readline().strip().split()
        n, c = map(int, header)
        
        values = []
        for _ in range(n):
            values.append(int(f.readline().strip()))
            
        values = np.array(values, dtype=np.int32)

    bins = first_fit_decreasing(values, c)
    
    print(len(bins))
    for b in bins:
        print(" ".join(map(str, sorted(b))))

if __name__ == "__main__":
    configure_system()
    main()