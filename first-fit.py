#!/usr/bin/env python3

# performance optimization imports
import sys
import numpy as np
from numba import njit, int32, types, config
from numba.typed import List

# enable aggressive optimizations
config.FASTMATH = True
config.BOUNDSCHECK = False

@njit(int32(int32[::1], int32, int32))
def find_bin(remaining_space, val, active_bins):
    """linear search for first bin that can fit the current item
    args:
        remaining_space: array of remaining capacities
        val: item size to place
        active_bins: number of bins currently in use
    returns:
        index of first suitable bin or active_bins if none found
    """
    for j in range(active_bins):
        if remaining_space[j] >= val:
            return j
    return active_bins

@njit
def first_fit_decreasing(values, capacity):
    """first-fit decreasing bin packing implementation
    args:
        values: array of item sizes
        capacity: bin capacity
    returns:
        list of bins, where each bin is a list of items
    
    optimization features:
    - cache-aligned arrays (16 items per chunk)
    - dynamic bin resizing
    - numba jit compilation
    - typed lists for maximum performance
    """
    # handle empty input case
    if len(values) == 0:
        # initialize empty result with proper typing
        lst = List()
        lst.append(List.empty_list(types.int32))
        lst.pop()
        return lst
    
    # prepare items by sorting in descending order
    # larger items first reduces fragmentation
    sorted_values = np.sort(values)[::-1]
    
    # early exit if largest item exceeds capacity
    if sorted_values[0] > capacity:
        result = List()
        temp = List.empty_list(types.int32)
        result.append(temp)
        result.pop()
        return result

    # initialize data structures for bin tracking
    max_bins = len(values)  # worst case: one item per bin
    remaining = np.full(max_bins, capacity, dtype=np.int32)  # track remaining space
    bin_contents = List()  # actual items in each bin
    bin_sizes = np.zeros(max_bins, dtype=np.int32)  # current item count per bin
    CACHE_LINE_ITEMS = 16  # optimize for 64-byte cache lines

    # setup first bin with largest item
    first_bin = np.zeros(CACHE_LINE_ITEMS, dtype=np.int32)
    first_bin[0] = sorted_values[0]
    bin_contents.append(first_bin)
    remaining[0] = capacity - sorted_values[0]
    bin_sizes[0] = 1
    active_bins = 1

    # main packing loop
    for i in range(1, len(sorted_values)):
        val = sorted_values[i]
        bin_idx = find_bin(remaining, val, active_bins)

        if (bin_idx >= active_bins):
            # create new bin if no existing bin can fit item
            new_bin = np.zeros(CACHE_LINE_ITEMS, dtype=np.int32)
            new_bin[0] = val
            bin_contents.append(new_bin)
            remaining[active_bins] = capacity - val
            bin_sizes[active_bins] = 1
            active_bins += 1
        else:
            # add to existing bin
            remaining[bin_idx] -= val
            current_bin = bin_contents[bin_idx]
            current_size = bin_sizes[bin_idx]
            
            # resize bin if needed (double capacity)
            if current_size >= current_bin.size:
                new_size = current_bin.size * 2
                new_bin = np.zeros(new_size, dtype=np.int32)
                new_bin[:current_size] = current_bin[:current_size]
                bin_contents[bin_idx] = new_bin
                current_bin = new_bin
            
            current_bin[current_size] = val
            bin_sizes[bin_idx] += 1

    # construct final result structure
    # convert from numpy arrays to typed lists
    result = List()
    for i in range(active_bins):
        count = bin_sizes[i]
        items = bin_contents[i][:count]
        typed_list = List.empty_list(types.int32)
        for item in items:
            typed_list.append(item)
        result.append(typed_list)
    
    return result

def main():
    """main entry point: handles file input and output
    validates input data and coordinates solution"""
    # verify correct usage
    if len(sys.argv) < 2:
        print("Usage: python first-fit.py <input_file>")
        sys.exit(1)

    # read and validate input
    with open(sys.argv[1], 'r') as f:
        header = f.readline().strip().split()
        if len(header) != 2:
            raise ValueError("Invalid header format")
        n, c = map(int, header)
        
        # read items with validation
        values = []
        for _ in range(n):
            line = f.readline().strip()
            if line:
                val = int(line)
                if val <= 0 or val > c:
                    raise ValueError(f"Invalid item size: {val} (capacity {c})")
                values.append(val)
        
        if len(values) != n:
            raise ValueError(f"Expected {n} items, got {len(values)}")

    # prepare data and solve
    values_np = np.array(values, dtype=np.int32)
    bins = first_fit_decreasing(values_np, c)
    
    # output solution
    print(len(bins))
    for b in bins:
        print(" ".join(map(str, sorted(b))))

if __name__ == "__main__":
    main()