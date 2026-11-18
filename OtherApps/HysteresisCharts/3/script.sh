#!/bin/bash
#python RE_2D_Tocci.py -snes_fd_color -ts_bdf_order 5   -pc_type mg  -pc_mg_levels 1 -pc_mg_galerkin both  -mg_levels_ksp_type richardson -ts_monitor
 
python3 setup.py build_ext --inplace
python3 run.py 

 
