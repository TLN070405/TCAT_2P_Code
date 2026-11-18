#!/bin/bash
#python RE_2D_Tocci.py -snes_fd_color -ts_bdf_order 5   -pc_type mg  -pc_mg_levels 1 -pc_mg_galerkin both  -mg_levels_ksp_type richardson -ts_monitor
 
 #-ksp_converged_reason 
export PYTHONPATH=/home/farah/Desktop/petsc/arch-linux-c-debug/lib
python3 setup_TCAT.py build_ext --inplace
for nodes in 51 101 201 401 801
do
    echo "Running simulation with nodes=$nodes"
    python3 run_TCAT.py -Nx $nodes -snes_fd_color_use_mat -ts_bdf_order 5 -pc_type sor -ts_monitor
done


