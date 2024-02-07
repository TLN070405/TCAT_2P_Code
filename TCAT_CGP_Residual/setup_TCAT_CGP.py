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
    Extension("TCAT_CGP_Residual.Spacial_Discr_CGP",["./TCAT_CGP_Subcodes/Spacial_Discr/Spacial_Discr_CGP.pyx"],annotate=True,include_dirs=['RE_2P_TCAT_CGP_Code'],compiler_directives={'boundscheck': False,'wraparound':False,'cdivision':True})
    
]
ext_modules += [
    Extension("TCAT_CGP_Residual.TCAT_CGP_RESIDUAL",["./TCAT_CGP_Residual/TCAT_CGP_RESIDUAL.pyx"],annotate=True,include_dirs=['RE_2P_TCAT_CGP_Code'],compiler_directives={'boundscheck': False,'wraparound':False,'cdivision':True})
    
]
cmdclass.update({'build_ext': build_ext})

setup(
    name="TCAT_CGP",
    version="0.0.1",
    packages=['TCAT_CGP'],
    author="Tom Nelson",
    description="TCAT CGP Codes",
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
