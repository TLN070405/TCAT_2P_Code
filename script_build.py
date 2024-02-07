import os
import shutil
# Setting PYTHONPATH
# Type in Command Line: python script_build.py

Model = "TCAT_CGP"  # model = "TCAT_CGP", "RE" or "2P"

# Build codes if necessary
if Model == "TCAT_CGP":
    os.system("python ./TCAT_CGP_Residual/setup_TCAT_CGP.py build_ext --inplace")
elif Model == "RE":
    os.system("python ./RE_Residual/setup_RE.py build_ext --inplace")
elif Model == "2P":
    os.system("python ./TwoP_Residual/setup_2P.py build_ext --inplace")


shutil.rmtree('./build')
