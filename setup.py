from  setuptools import setup, find_packages, Extension, dist
from Cython.Build import cythonize
import numpy as np


ext_modules=[Extension("proxcd",
                       sources=["regipca/proxcd.pyx"],
                       include_dirs=[np.get_include()]),
            Extension("proxcd_lasso",
                      sources=["regipca/proxcd_lasso.pyx"],
                      include_dirs=[np.get_include()]),
            Extension("lsgrad",
                      sources=["regipca/lsgrad.pyx"],
                      include_dirs=[np.get_include()]),
            Extension("kron",
                      sources=["regipca/kron.pyx"],
                      include_dirs=[np.get_include()]),
            Extension("partinner",
                      sources=["regipca/partinner.pyx"],
                      include_dirs=[np.get_include()]),
            ]

setup(name='regipca',
      version='0.7.1',
      description='Implements the regularized IPCA method of Bybee, Kelly, Su (2023)',
      url='https://github.com/lbybee/regipca',
      author='Leland Bybee',
      author_email='leland.bybee@gmail.com',
      license='MIT',
      packages=['regipca'],
      ext_modules=cythonize(ext_modules),
      zip_safe=False)
