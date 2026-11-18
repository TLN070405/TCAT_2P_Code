import distutils.core
import Cython.Build
#remember to change the name of the code that you are working with
distutils.core.setup(ext_modules = Cython.Build.cythonize("MAALGORZATA_Parker.pyx",annotate=True,compiler_directives={'boundscheck': False,'wraparound':False,'cdivision':True}))

