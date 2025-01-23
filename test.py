#!/usr/bin/env python3

# performance benchmarking script for bin packing algorithms
# compares execution times between baseline and optimized implementations

import os
import subprocess  # for running algorithms in separate processes
import time  # precise timing measurements
import numpy as np  # efficient data generation

# directory configuration for test artifacts
test_dir = "perf_tests"
os.makedirs(test_dir, exist_ok=True)  # ensure test directory exists

def generate_perf_test(n, c, max_item_size):
    """generates valid test cases for bin packing algorithms
    - n: number of items to generate
    - c: bin capacity
    - max_item_size: maximum size for individual items
    ensures: 1 <= items <= min(max_item_size, c-1) for valid packing"""
    
    # safety clamp to prevent invalid items that exceed capacity
    max_item = min(max_item_size, c-1)
    
    # generate random integers using numpy for better performance
    values = np.random.randint(1, max_item+1, n).tolist()
    
    # create standardized filename with parameters
    filename = f"perf_{n}_{c}.txt"
    filepath = os.path.join(test_dir, filename)
    
    # write test file format compatible with algorithm inputs
    with open(filepath, 'w') as f:
        f.write(f"{n} {c}\n")  # header line
        for v in values:
            f.write(f"{v}\n")  # one item per line
            
    return filename

def time_algorithm(script_name, test_file):
    """measures execution time of a single algorithm run
    - uses subprocess to isolate execution environment
    - captures stderr for error reporting
    - returns timing in seconds or none for failed runs"""
    
    try:
        # build full path to test file
        test_path = os.path.join(test_dir, test_file)
        
        # time execution with high-resolution counter
        start = time.perf_counter()
        
        # run algorithm in separate process
        result = subprocess.run(
            ["python3", script_name, test_path],
            capture_output=True,  # capture output without printing
            text=True  # work with string streams
        )
        end = time.perf_counter()
        
        # handle non-zero exit codes
        if result.returncode != 0:
            print(f"  {script_name} failed: {result.stderr.strip()}")
            return None
            
        return end - start
        
    except Exception as e:
        print(f"  error timing {script_name}: {str(e)}")
        return None

def run_perf_comparison(test_config):
    """orchestrates full performance comparison for a test case
    - generates test data
    - runs both algorithm implementations
    - calculates and displays results"""
    
    print(f"\nbenchmarking {test_config['name']}...")
    
    # generate test data with validated parameters
    test_file = generate_perf_test(
        test_config["n"],
        test_config["c"],
        test_config["max_item"]
    )
    
    # display test parameters
    print(f"  dataset: {test_config['n']} items, capacity {test_config['c']}")
    print("  running tests...")
    
    # time both implementations
    base_time = time_algorithm("first-fit.py", test_file)  # baseline
    ultra_time = time_algorithm("ultra-fit.py", test_file)  # optimized
    
    # handle failed benchmarks
    if not base_time or not ultra_time:
        print("  benchmark failed")
        return
    
    # display formatted results
    print(f"\n  results:")
    print(f"  original ffd: {base_time:.4f} seconds")
    print(f"  ultra ffd:    {ultra_time:.4f} seconds")
    print(f"  speedup:      {base_time/ultra_time:.1f}x")
    print("  " + "-"*40)

# performance test configurations cover various scenarios
perf_tests = [
    {
        "name": "medium scale",
        "n": 10_000,
        "c": 1_000,
        "max_item": 750,
        "desc": "typical medium-sized problem"
    },
    {
        "name": "large uniform",
        "n": 100_000,
        "c": 10_000,
        "max_item": 8_000,
        "desc": "large dataset with uniform sizes"
    },
    {
        "name": "huge sparse",
        "n": 1_000_000,
        "c": 100_000,
        "max_item": 25_000,
        "desc": "million items with sparse distribution"
    },
    {
        "name": "worst-case pattern",
        "n": 50_000,
        "c": 100,
        "max_item": 66,
        "desc": "stress test with 2/3 capacity items"
    }
]

if __name__ == "__main__":
    print("starting performance benchmark suite")
    
    # execute all configured performance tests
    for test in perf_tests:
        run_perf_comparison(test)
        
    print("\nall benchmarks completed")