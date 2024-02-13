import numpy as np
def CreateArrays(Nz,time_interval):
    # we created those arrays using numpy and not C malloc function, because they are used only few times, So they won't effect the speed of code
    Mw=np.zeros([len(time_interval)], dtype=np.double) #here we store the mass accumulation of wetting phase
    Sum_rho_Epsilonw=np.zeros([len(time_interval)], dtype=np.double) 
    Integral_qw_L=np.zeros([len(time_interval)], dtype=np.double) 
    Integral_qw_0=np.zeros([len(time_interval)], dtype=np.double)
    Error_Mass=np.zeros([len(time_interval)], dtype=float)# here we store Error mass at each output time
    Vp_w_t0= np.zeros([Nz], dtype=np.double) #here we store the pressure of w-phase at t=t0 (we need this for mass error calculation)
    VEpsilon_w_t0= np.zeros([Nz], dtype=np.double)#here we store the volume fraction of w-phase at t=t0 (we need this for mass error calculation)
    VEpsilonw=np.zeros([len(time_interval),Nz], dtype=np.double) #here we store the final solutions of Epsilonw at diffent output times and space position
    VPw=np.zeros([len(time_interval),Nz], dtype=np.double)#here we store the final solutions of Pw at diffent output times and space position
    VPn=np.zeros([len(time_interval),Nz], dtype=np.double)#here we store the final solutions of Pn at diffent output times and space position
    Clock_Array=np.zeros([len(time_interval)], dtype=float) # here we store the CPU time consumed to reach each output time
    return Mw, Sum_rho_Epsilonw,Integral_qw_L,Integral_qw_0,Error_Mass,Vp_w_t0,VEpsilon_w_t0,VEpsilonw,VPw,VPn,Clock_Array
