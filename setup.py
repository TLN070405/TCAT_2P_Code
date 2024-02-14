from setuptools import setup
from distutils.core import Extension
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
    Extension("TCAT2Phase.Spacial_Discr_CGP",
              ["TCAT2Phase/Discretizations/Spacial_Discr_CGP.pyx",
               "TCAT2Phase/Discretizations/Spacial_Discr_RE.pyx",
               "TCAT2Phase/Discretizations/Spacial_Discr.pyx"],
              include_dirs=['TCAT2Phase/Discretizations'],
              compiler_directives={'boundscheck': False,'wraparound':False,'cdivision':True}),
    Extension("TCAT2Phase.Residual_Functions._Two_Phases_RESIDUAL",
               ["TCAT2Phase/Residual_Functions/_Two_Phases_RESIDUAL.pyx"],
              include_dirs=['TCAT2Phase/Residual_Functions']),
    Extension("TCAT2Phase.Residual_Functions.RE_RESIDUAL",
               ["TCAT2Phase/Residual_Functions/_RE_RESIDUAL.pyx"],
              include_dirs=['TCAT2Phase/Residual_Functions']),
    Extension("TCAT2Phase.Residual_Functions._TCAT_CGP_RESIDUAL",
               ["TCAT2Phase/Residual_Functions/_TCAT_CGP_RESIDUAL.pyx"],
              include_dirs=['TCAT2Phase/Residual_Functions'])
]
cmdclass.update({'build_ext': build_ext})

setup(
    name="TCAT2Phase",
    version="0.0.1",
    packages=['TCAT2Phase'],
    author="Tom Nelson",
    description="Two-Phase Flow in Porous Media",
    long_description=long_description,
    long_description_content_type='text/markdown',
    cmdclass=cmdclass,
    ext_modules=cythonize(ext_modules,annotate=True),
    include_dirs=numpy.get_include(),
    install_requires=[
        'numpy>=1.22.3',
        'mpi4py>=3.1.3',
    ]
)
