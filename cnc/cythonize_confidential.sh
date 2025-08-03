#!/bin/bash

python setup.py build_ext --inplace
python clean_after_cythonize.py
rm build -rf
