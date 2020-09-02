import os
from setuptools import setup, Extension, Command, find_packages
import os

os.system("make build")

import numpy as np

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

if 'CUDA_PATH' in os.environ:
   CUDA_PATH = os.environ['CUDA_PATH']
else:
   print("Could not find CUDA_PATH in environment variables. Defaulting to /usr/local/cuda!")
   CUDA_PATH = "/usr/local/cuda"

if not os.path.isdir(CUDA_PATH):
   print("CUDA_PATH {} not found. Please update the CUDA_PATH variable and rerun".format(CUDA_PATH))
   exit(0)

if not os.path.isdir(os.path.join(CUDA_PATH, "include")):
    print("include directory not found in CUDA_PATH. Please update CUDA_PATH and try again")
    exit(0)

extra_compile_args = ["-fopenmp","-fno-wrapv"]
extra_link_args=['-fopenmp']


setup(name = 'parallelPermutationTest', version = '1.0.1',  \
   ext_modules = [
      Extension('permutationTest', ['permutationTest.cpp'],
      extra_compile_args=extra_compile_args,
      extra_link_args=extra_link_args,
      include_dirs=[np.get_include(), os.path.join(CUDA_PATH, "include")],
      libraries=["green", "cudart"],
        library_dirs = [".", os.path.join(CUDA_PATH, "lib64"), "./green"],
)],
    url="https://github.com/statisticalbiotechnology/parallelGreen",
    author="Markus Ekvall",
     author_email="marekv@kth.se",
    package_dir={'_parallelPermutationTest': '_parallelPermutationTest'},
    packages=['parallelPermutationTest'],
 )
