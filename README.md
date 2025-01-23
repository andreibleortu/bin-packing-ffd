# Bin Packing Optimization Challenge

An exploration of optimizing the First-Fit Decreasing (FFD) bin packing algorithm through hardware-aware implementations.

## Project Structure

```
.
├── first-fit.py    # Baseline FFD implementation with basic optimizations
├── ultra-fit.py    # Hardware-optimized FFD with SIMD/AVX support
├── test.py         # Performance benchmarking suite
└── README.md       # Technical analysis and implementation details
```

## Implementations

### first-fit.py
- Basic FFD implementation with Numba JIT compilation
- Cache-aligned data structures
- Dynamic bin resizing
- Type-specialized containers

### ultra-fit.py
- Hardware-aware configuration system
- SIMD-optimized bin search (SSE2/AVX2/AVX-512)
- Thread-optimized processing
- Enhanced memory layout
- Advanced compiler optimizations

### test.py
- Comprehensive benchmarking suite
- Generates test cases of varying sizes
- Compares performance between implementations
- Reports detailed timing statistics

## Requirements

```bash
pip install numpy numba cpuinfo psutil
```

## Usage

Running a single test:
```bash
python3 first-fit.py input.txt    # Basic implementation
python3 ultra-fit.py input.txt    # Optimized version
```

Running benchmarks:
```bash
python3 test.py                   # Executes full benchmark suite
```

Input format:
```
n c        # n = number of items, c = bin capacity
item_1     # One item size per line
item_2
...
item_n
```

Output format:
```
k          # k = number of bins used
bin_1      # Space-separated items in each bin
bin_2
...
bin_k
```

## Performance

Key improvements achieved:
- SIMD vectorization: Up to 4x speedup with AVX-512
- Cache optimization: Reduced memory latency
- Thread utilization: Improved parallel processing
- Memory layout: Better spatial locality

See ultra-fit.md for detailed technical analysis and benchmarks.

## Contributing

Feel free to experiment with:
- Alternative bin search algorithms
- New vectorization strategies
- Memory layout optimizations
- Additional hardware-specific tuning

## License

MIT License - See LICENSE file for details