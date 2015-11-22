from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension("pyBlockSQP", ['src/IProblemspec.cpp', 'src/pyBlockSQP.pyx'],
              include_dirs = ['/opt/blockSQP/include',
                              '/opt/qpOASES-3.2.0/include'],
              libraries = ['blockSQP', 'qpOASES', 'openblas'],
              extra_link_args=["-L/opt/blockSQP/lib",
                               "-L/opt/qpOASES-3.2.0/bin",
                               "-L/opt/OpenBLAS/lib"],
              language='c++'),
]

setup(ext_modules = cythonize(extensions
      ))
