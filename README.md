
DESCRIPTION:
-----------
This code is an implementation of 3 models:
1- the traditional two-phase model using three formulations from Kees paper: SP (2 equations), PP (2 equations), and SPP formulation (SPP includes an additional equation, which is the dynamic capillary pressure equation).
2- Richards' equation uses the PR formulation (PR stands for pressure, and the model includes one equation).
3- The TCAT CGP model corresponds to the RE model; the system has six equations against six unknowns.

HOW TO RUN CODE:
-----------
The code can be efficiently executed using the Python scripts file.

The script_build.py file is used only to build cython files. If the files are already built, there is no need to redo it.

The script_running.py file is used to get results from the three models. The variables defined inside the script enable the user to choose which model and test problem to solve. Also, the visualization can be turned on if desired.

In the running commands such as "python run_TCAT_CGP.py -snes_fd_color_use_mat -ts_bdf_order 1 -pc_type sor -ts_monitor -Test_Case $Test_Case -filenumber 1", the filenumber is used to name the output files, so if for example, we solve TCAT CGP test_case1 with Nx=101, we get a file named "..._File1" (of course filenumber is set to 1 in the command). If I want to get another output with Nx=201 and save the old output file, I just set "file number 2 in the command."

Another Python script named "script_Multiple_Plots" is used to plot RE and CGP results in the same plot. The command is "python ./Multiple_Plots/Visualization.py -Test_Case $Test_Case -filenumber_RE 1 -filenumber_CGP 1 " this means that the code will look for the RE output file that its name contains (RE, Test_case, and filenumber_RE=1). On the other hand, it will look for the other output file corresponding to the CGP model; its name contains (TCAT_CGP, Test_Case, and filenumber_CGP=1). Of course, the output files should exist before running the multiple plot script.

Another example of multiple plots is: let's suppose that we run RE code from script_running.py  file with Test_Case="Test_Case1" and "filenumber_RE 2", then we get an output file named "Outputs_RE_1_file2.txt". We again run the CGP code with Test_Case="Test_Case1" and "filenumber_CGP 1". If we want to plot in the same window this two files we just run the "script_Multiple_Plots.py" with the command:

python ./Multiple_Plots/Visualization.py -Test_Case $Test_Case -filenumber_RE 2 -filenumber_CGP 1

CODE USAGE SUMMARY (TEST CASE 8: HYSTERESIS AND ENTRAPMENT):
-----------
This Python code runs the TCAT2Phase simulation framework to model two-phase flow with hysteresis and entrapment effects. To use it, install Python 3, petsc4py, mpi4py, numpy, h5py, and matplotlib. Then clone the TCAT_2P_Code repository from GitHub and install the package from the repo root:

pip install -e .

Open the project in VS Code and go to applications/Testcase8_complex_displacement_patterns/. Run either the two-phase driver (2p_malgorzata_hysteresis_entrapment_pdc.py) or the Richards equation driver (re_malgorzata_hysteresis_entrapment_pdc.py) with the Run button in the upper right. The solver writes .h5 output files to outputs/Testcase8_complex_displacement_patterns/outputs/.

To visualize results, run plot_results_2p.py or plot_results_re.py from outputs/Testcase8_complex_displacement_patterns/postprocessing/. These produce plots of the key variables.

The model uses Parker-Lenhard hysteresis, van Genuchten closure relations, and Land's model for entrapment. Parameters such as porosity, permeability, and boundary and initial conditions can be changed directly in the driver script to fit other cases or materials.
