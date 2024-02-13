import numpy as np

def CreateArrays(Nx,time_interval):
    VPw_bc               = np.zeros([Nx], dtype=np.double) 
    ####################(Arrays to store results)#######################
    Ve_w_Time_Results    = np.zeros([len(time_interval),Nx], dtype=np.double) 
    Vp_w_Time_Results    = np.zeros([len(time_interval),Nx], dtype=np.double) 

    ##############(Arrays to store time of simulation)#################
    Clock_Array          = np.empty([len(time_interval)], dtype=float) 
    ftime                = np.empty([len(time_interval)], dtype=float) 
    
    return VPw_bc,Ve_w_Time_Results,Vp_w_Time_Results,Clock_Array,ftime
