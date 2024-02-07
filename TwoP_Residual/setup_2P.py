from setuptools import setup
from distutils.core import setup,Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

import numpy

with open("README.md", 'r') as f:
    long_description = f.read()

cmdclass = {}
ext_modules = []
ext_modules += [
    Extension("TwoP_Residual.Spacial_Discr",["./2P_Subcodes/Spacial_Discr/Spacial_Discr.pyx"],annotate=True,include_dirs=['TwoPhaseModels'],compiler_directives={'boundscheck': False,'wraparound':False,'cdivision':True})
    
]
ext_modules += [
    Extension("TwoP_Residual.Two_Phases_RESIDUAL",["./TwoP_Residual/Two_Phases_RESIDUAL.pyx"],annotate=True,include_dirs=['TwoPhaseModels'],compiler_directives={'boundscheck': False,'wraparound':False,'cdivision':True})
    
]
cmdclass.update({'build_ext': build_ext})

setup(
    name="TwoPhaseModels",
    version="0.0.1",
    packages=['TwoPhaseModels'],
    author="Tom Nelson",
    description="TCAT Two Phase Codes",
    long_description=long_description,
    long_description_content_type='text/markdown',
    cmdclass=cmdclass,
    ext_modules=cythonize(ext_modules,annotate=True),
    include_dirs=numpy.get_include(),
    install_requires=[
        'numpy>=1.22.3',
        'mpi4py>=3.1.3'
    ]
)
