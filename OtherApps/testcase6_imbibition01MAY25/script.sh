#!/bin/bash
#python RE_2D_Tocci.py -snes_fd_color -ts_bdf_order 5   -pc_type mg  -pc_mg_levels 1 -pc_mg_galerkin both  -mg_levels_ksp_type richardson -ts_monitor
 
 #-ksp_converged_reason 
export PYTHONPATH=/home/farah/Desktop/petsc/arch-linux-c-debug/lib
python3 setup_TCAT.py build_ext --inplace
python3 run_TCAT.py -snes_fd_color_use_mat -ts_bdf_order 5 -pc_type ilu -ts_monitor 
 
 
