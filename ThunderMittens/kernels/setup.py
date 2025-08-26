# Copyright © 2023-2024 Apple Inc.

from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        name="tk",
        version="0.0.0",
        description="Sample C++ and Metal extensions for MLX primitives.",
        ext_modules=[extension.CMakeExtension("tk._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["tk"],
        package_data={"tk": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.8",
    )
