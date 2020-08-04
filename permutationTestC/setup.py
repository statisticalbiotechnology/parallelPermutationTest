import sysconfig

""" from distutils.core import Extension
from setuptools import setup """

from distutils.core import setup, Extension


""" extra_compile_args = sysconfig.get_config_var('CFLAGS').split() """
''' extra_compile_args = ["-fopenmp",'-fpic',"-O3", "-fno-wrapv"] '''
extra_compile_args = ["-fopenmp",'-fpic',"-O3", "-fno-wrapv"]
extra_link_args=['-fopenmp',"-shared"]

factorial_module = Extension('permutationTest',sources = ['permutationTest.cpp'],
 extra_compile_args=extra_compile_args,
 extra_link_args=extra_link_args)

setup(name='Exact Permuation Test',
version='1.0',
description='Exact Permuation Test',
ext_modules=[factorial_module],
)
