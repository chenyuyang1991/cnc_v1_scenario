from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os
import yaml

# Load files to compile from YAML
with open("cython_modules.yaml", "r") as f:
    config = yaml.safe_load(f)
    all_modules = config["python_files"]    # ex: model/agent/langchain/src/session.py

# Create Extension objects for each module
extensions = []
for module in all_modules:
    module_name = os.path.splitext(module)[0].replace("/", ".")
    extensions.append(
        Extension(
            module_name,
            [module],
            include_dirs=["."],
            extra_compile_args=["-O3"],
        )
    )

setup(
    name="cnc_genai",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": True,  # Enable bounds checking for array operations
            "wraparound": True,  # Enable wraparound checking for array operations
            "cdivision": True,  # Keep C-style division for performance
            "nonecheck": False,  # Keep none checking disabled for performance
            "c_string_type": "str",
            "c_string_encoding": "utf8",
            "embedsignature": True,  # Add function signatures to docstrings
            "always_allow_keywords": True,  # Allow keyword arguments
        },
    ),
    install_requires=[
        "cython>=3.0.0",
        "pyyaml>=6.0.0",
    ],
    python_requires=">=3.11",
    author="Your Name",
    author_email="your.email@example.com",
    description="GenAI 射出參數優化解決方案",
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
)
