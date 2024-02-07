
DESCRIPTION:
-----------
This code is an implementation of 3 models: 
1- the traditional two phases model using three formulation from Kees paper: SP (2 equations), PP (2 equations) and SPP formulation (SPP includes an additional equation which is the dynamic capillary pressure equation).
2- Richards equation using the PR formulation (PR stands for pressure and the model includes one equation).
3- The TCAT CGP model which correspond to the RE model, the system has 6 equations against 6 unknowns.

HOW TO RUN CODE:
-----------
The code can be easily executed using the script shell file.

The script.sh file can be used to get results from the 3 models. The variables defined inside the script enable the user to choose which model and which test problem to solve. Also the visualization can be turned on if wanted.


A variable named "Build" is used to build the cython files if necessary. If the files has been already built the "Build" variable can be turned off, in case you deleted the built files you just turn the variable into "Yes" and you get results. So, in general the "Build" variable is not necessary unless you delete the cython built files.


In the running commands such as "python run_TCAT_CGP.py -snes_fd_color_use_mat -ts_bdf_order 1 -pc_type sor -ts_monitor -Test_Case $Test_Case -filenumber 1", the filenumber is used to name the output files, so if for example we solve TCAT CGP test_case1 with Nx=101, we get a file named "..._File1" (of course filenumber is set to 1 in the command). If i want to get another output with Nx=201 and save the old output file, i just set "filenumber 2 in the command"

Another script named "script_Multiple_Plots" is used to plot RE and CGP results in same plot. The command is "python ./Multiple_Plots/Visualization.py -Test_Case $Test_Case -filenumber_RE 1 -filenumber_CGP 1 " , this means that the code will look for RE output file that its name contains (RE, Test_case and filenumber_RE=1). On the other hand, it will look for the other output file corresponding to CGP model, its name coints (TCAT_CGP, Test_Case and filenumber_CGP=1). Of course the output files should exist before running the multiple plot script. 


Another example of multiple plot is: lets suppose that i run RE code from script.sh file with Test_Case="Test_Case1" and "filenumber_RE 2", then we get an output file named "Outputs_RE_1_file2.txt". We again run the CGP code with Test_Case="Test_Case1" and "filenumber_CGP 1". If i want to plot in same window this two files i just run the "script_Multiple_Plots.sh" with command:

python ./Multiple_Plots/Visualization.py -Test_Case $Test_Case -filenumber_RE 2 -filenumber_CGP 1
