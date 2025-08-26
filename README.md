# ThunderMittens
![ThunderMittens](https://github.com/user-attachments/assets/f539ee38-e4fa-4908-ac8a-3c1b0eb999bd)

## Overview

ThunderMittens is an Apple Metal Shading Language (MSL) port for the [ThunderKittens](github.com/HazyResearch/ThunderKittens/) framework.

## Project Structure

The repository supports two primary use cases:

### 1. MSL Kernel Development

For writing Metal Shading Language (MSL) kernels:
- Clone the ThunderKittens repository
- Open the project in Xcode
- Xcode will handle all build processes

### 2. MLX Kernel Integration with Python

For using ThunderKittens kernels within MLX in Python:

#### Prerequisites
- Python 3.8+
- CMake
- Xcode Command Line Tools

#### Installation Steps

1. Navigate to ThunderMittens/mlx directory
2. Install MLX with parallel build:
   ```bash
   CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e ".[dev]"
   ```

3. Navigate to ThunderMittens/kernels directory
4. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

5. Build kernels and bindings:
   ```bash
   python setup.py build_ext -j8 --inplace
   ```

## References

Please see [our blog post](https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx) to learn more about this work. 



