#!/bin/bash


python setup.py build_ext --inplace
python run.py -snes_fd_color_use_mat -ts_bdf_order 5 -pc_type sor -ts_monitor 
 

 
