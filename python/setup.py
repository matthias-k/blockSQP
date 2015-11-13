from distutils.core import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension("pyBlockSQP", ['src/pyBlockSQP.pyx'],
              include_dirs = ['/opt/blockSQP/include'],
              libraries = ['blockSQP', 'qpOASES', 'openblas'],
              extra_link_args=["-L/opt/blockSQP/lib",
                               "-L/opt/qpOASES-3.2.0/bin",
                               "-L/opt/OpenBLAS/lib"],
              language='c++'),
]

setup(ext_modules = cythonize(extensions
      ))
