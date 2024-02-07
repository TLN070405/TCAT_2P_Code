import os

# Setting PYTHONPATH

# General variables, which are valid for all models
Model = "2P"  # model = "TCAT_CGP", "RE" or "2P"
Visualization = "On"  # "On" or "Off"

# Variables for RE and CGP models
Test_Case = "Test_Case4"  # Test_Case="Test_Case1", "Test_Case2" ...   to 4 and they are related only to RE and CGP models

# Variables for 2P model
Paper = "Malgorzata"  # can be "Fucik", "Helmig" or   "Malgorzata"
Formulation = "SPP"  # can be "SPP" "PP" or "SP".

# part to run codes with the specified parameters
if Model == "TCAT_CGP":
    # in some testcases the ts_bdf_order can affect the convergence. For example in test case 4 use order 5
    os.system(f"python ./TCAT_CGP_Residual/run_TCAT_CGP.py -snes_fd_color_use_mat -ts_bdf_order 1 -pc_type sor -ts_monitor -Test_Case {Test_Case} -filenumber 1")
    if Visualization == "On":
        os.system(f"python ./TCAT_CGP_Subcodes/Visualization/Visualization.py -Test_Case {Test_Case} -filenumber 1")

elif Model == "RE":
    # in some testcases the ts_bdf_order can affect the convergence.
    os.system(f"python ./RE_Residual/run_RE.py -snes_fd_color_use_mat -ts_bdf_order 5 -pc_type sor -ts_monitor -Test_Case {Test_Case} -filenumber 1")
    if Visualization == "On":
        os.system(f"python ./RE_Subcodes/Visualization/Visualization.py -Test_Case {Test_Case} -filenumber 1")

elif Model == "2P":
    # in some papers the ts_bdf_order can affect the convergence.
    os.system(f"python ./TwoP_Residual/run_2P.py -snes_fd_color -ts_bdf_order 5  -pc_type sor -ts_monitor -filenumber 1 -Paper {Paper} -Formulation {Formulation} -Mass_Activation ON")
    if Visualization == "On":
        os.system(f"python ./2P_Subcodes/Visualization/Visualization.py -Paper {Paper} -Files_number 1")
