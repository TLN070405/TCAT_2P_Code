import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
Gamma_wn=35e-3
rho_w0 =1000
g =9.81
epsilon =0.369
Nex=1000
def  ew_re( x):

    n                = 3.4361          
    m                = 1.0 - (1.0 / n)     
    alpha            = 0.2001*1e-3          
    Epsilon_wr       =  0.0                       #[Epsilon_wr]=null
    Epsilon_nr       =  0.0                       #[Epsilon_nr]=null
    Const_Epsilon    = epsilon-Epsilon_nr - Epsilon_wr
    P_c=+1000+rho_w0*g*x

    if (P_c > 0.0):
       Se =  (1.0 + (alpha*P_c)**n)**(-m)
       Epsilon_w =Const_Epsilon*  Se + Epsilon_wr
    else:
       Epsilon_w = epsilon-Epsilon_nr
    return Epsilon_w


#######################################
def main():
   
#####################################################################################################################
###############################################TCAT################################################################
#####################################################################################################################
#######################################File1
#######################################
   # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
   Output_Path='Outputs_TCAT_Parameters'
   file = open(Output_Path, "r")
   line_count = 0
   for line in file:
       if line != "\n":
          line_count += 1      
   file.close()
   numbers_str=[None]*line_count #the lines will be stored inside "numbers_str"
   # we open again the file to get different results
   with open(Output_Path,"r") as f:
        i=-1
        for line in f:  
            i=i+1
            numbers_str[i] = line.split() #we read each line in the file and store its elements as strings in "numbers_str[i]"
       # store each line in the correct array 
       
       
   Nx=int(numbers_str[0][1]) # the first index in numbers_str is the line number and the second index is the value; we starts from 1 in the second index because 0 corresponds to the name (see output.txt)
   Omega_0= float(numbers_str[1][1]) 
   Omega_L= float(numbers_str[2][1])  
   time_interval=np.zeros([len(numbers_str[3])-1], dtype=np.double)
   for i in range(len(time_interval)):
       time_interval[i]= float(numbers_str[3][i+1])     
#######################################File2
#######################################
   Ve_w_Time_TCAT=np.zeros([len(time_interval),Nx], dtype=np.double) 
   Ve_w_Time_Exact=np.zeros([Nex], dtype=np.double) 

   # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
   Output_Path='Outputs_TCAT_Ew'
   file = open(Output_Path, "r")
   line_count = 0
   for line in file:
       if line != "\n":
          line_count += 1      
   file.close()
   numbers_str=[None]*line_count #the lines will be stored inside "numbers_str"
   # we open again the file to get different results
   with open(Output_Path,"r") as f:
        i=-1
        for line in f:  
            i=i+1
            numbers_str[i] = line.split() #we read each line in the file and store its elements as strings in "numbers_str[i]"
       # store each line in the correct array 
       
       
   for i in range(len(time_interval)):
       for j in range(0,Nx):
           Ve_w_Time_TCAT[i][j]=float(numbers_str[i][j+1]) 
   for j in range(0,Nex):
       dx = (Omega_L - Omega_0) / (Nex - 1.0)
       x=Omega_0+j*dx
       Ve_w_Time_Exact[j]= ew_re( x)
#######################################File3
#######################################
   Vp_w_Time_TCAT=np.zeros([len(time_interval),Nx], dtype=np.double) 
   Vp_w_Time_Exact=np.zeros([Nex], dtype=np.double) 

   # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
   Output_Path='Outputs_TCAT_Pw'
   file = open(Output_Path, "r")
   line_count = 0
   for line in file:
       if line != "\n":
          line_count += 1      
   file.close()
   numbers_str=[None]*line_count #the lines will be stored inside "numbers_str"
   # we open again the file to get different results
   with open(Output_Path,"r") as f:
        i=-1
        for line in f:  
            i=i+1
            numbers_str[i] = line.split() #we read each line in the file and store its elements as strings in "numbers_str[i]"
       # store each line in the correct array 
       
       
   for i in range(len(time_interval)):
       for j in range(0,Nx):
           Vp_w_Time_TCAT[i][j]=float(numbers_str[i][j+1])  
   for j in range(0,Nex):
       dx = (Omega_L - Omega_0) / (Nex - 1.0)
       x=Omega_0+j*dx
       Vp_w_Time_Exact[j]= -1000-rho_w0*g*x
#######################################File4
#######################################
   Ve_wn_Time_TCAT=np.zeros([len(time_interval),Nx], dtype=np.double)  
   Ve_wn_Time_Exact=np.zeros([Nex], dtype=np.double)  

   # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
   Output_Path='Outputs_TCAT_Ewn'
   file = open(Output_Path, "r")
   line_count = 0
   for line in file:
       if line != "\n":
          line_count += 1      
   file.close()
   numbers_str=[None]*line_count #the lines will be stored inside "numbers_str"
   # we open again the file to get different results
   with open(Output_Path,"r") as f:
        i=-1
        for line in f:  
            i=i+1
            numbers_str[i] = line.split() #we read each line in the file and store its elements as strings in "numbers_str[i]"
       # store each line in the correct array 
   for i in range(len(time_interval)):
       for j in range(0,Nx):
           Ve_wn_Time_TCAT[i][j]=float(numbers_str[i][j+1])

   for j in range(0,Nex):
       dx = (Omega_L - Omega_0) / (Nex - 1.0)
       x=Omega_0+j*dx
       Ve_wn_Time_Exact[j]=100*ew_re( x)
#######################################File4
#######################################
   Vj_wn_Time_TCAT=np.zeros([len(time_interval),Nx], dtype=np.double)  
   Vj_wn_Time_Exact=np.zeros([Nex], dtype=np.double)  

   # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
   Output_Path='Outputs_TCAT_Jwn'
   file = open(Output_Path, "r")
   line_count = 0
   for line in file:
       if line != "\n":
          line_count += 1      
   file.close()
   numbers_str=[None]*line_count #the lines will be stored inside "numbers_str"
   # we open again the file to get different results
   with open(Output_Path,"r") as f:
        i=-1
        for line in f:  
            i=i+1
            numbers_str[i] = line.split() #we read each line in the file and store its elements as strings in "numbers_str[i]"
       # store each line in the correct array 
   for i in range(len(time_interval)):
       for j in range(0,Nx):
           Vj_wn_Time_TCAT[i][j] = float(numbers_str[i][j+1])    
   for j in range(0,Nex):
       dx = (Omega_L - Omega_0) / (Nex - 1.0)
       x=Omega_0+j*dx
       Vj_wn_Time_Exact[j]=-Vp_w_Time_Exact[j]/Gamma_wn
#######################################File4
#######################################
   Vw_wn_Time_TCAT=np.zeros([len(time_interval),Nx], dtype=np.double)  
   Vw_wn_Time_Exact=np.zeros([Nex], dtype=np.double)  

   # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
   Output_Path='Outputs_TCAT_Wwn'
   file = open(Output_Path, "r")
   line_count = 0
   for line in file:
       if line != "\n":
          line_count += 1      
   file.close()
   numbers_str=[None]*line_count #the lines will be stored inside "numbers_str"
   # we open again the file to get different results
   with open(Output_Path,"r") as f:
        i=-1
        for line in f:  
            i=i+1
            numbers_str[i] = line.split() #we read each line in the file and store its elements as strings in "numbers_str[i]"
       # store each line in the correct array 
   for i in range(len(time_interval)):
       for j in range(0,Nx):
           Vw_wn_Time_TCAT[i][j]=float(numbers_str[i][j+1])   
   for j in range(0,Nex):
       Vw_wn_Time_Exact[j]=0.0
#######################################File4
#######################################
   Vk_wn_Time_TCAT=np.zeros([len(time_interval),Nx], dtype=np.double)  
   Vk_wn_Time_Exact=np.zeros([Nx], dtype=np.double) 

   # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
   Output_Path='Outputs_TCAT_Kwn'
   file = open(Output_Path, "r")
   line_count = 0
   for line in file:
       if line != "\n":
          line_count += 1      
   file.close()
   numbers_str=[None]*line_count #the lines will be stored inside "numbers_str"
   # we open again the file to get different results
   with open(Output_Path,"r") as f:
        i=-1
        for line in f:  
            i=i+1
            numbers_str[i] = line.split() #we read each line in the file and store its elements as strings in "numbers_str[i]"
       # store each line in the correct array
   for i in range(len(time_interval)):
       for j in range(0,Nx):
           Vk_wn_Time_TCAT[i][j]=float(numbers_str[i][j+1])
   for j in range(0,Nx):
       Vk_wn_Time_Exact[j]=Vk_wn_Time_TCAT[0][j]


#####################################################################################################################
###############################################RE################################################################
#####################################################################################################################
#######################################File2
#######################################
   Ve_w_Time_RE=np.zeros([len(time_interval),Nx], dtype=np.double)

   # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
   Output_Path='Outputs_RE_Ew'
   file = open(Output_Path, "r")
   line_count = 0
   for line in file:
       if line != "\n":
          line_count += 1      
   file.close()
   numbers_str=[None]*line_count #the lines will be stored inside "numbers_str"
   # we open again the file to get different results
   with open(Output_Path,"r") as f:
        i=-1
        for line in f:  
            i=i+1
            numbers_str[i] = line.split() #we read each line in the file and store its elements as strings in "numbers_str[i]"
       # store each line in the correct array 
       
       
   for i in range(len(time_interval)):
       for j in range(0,Nx):
           Ve_w_Time_RE[i][j]=float(numbers_str[i][j+1])    
#######################################File3
#######################################
   Vp_w_Time_RE=np.zeros([len(time_interval),Nx], dtype=np.double) 

   # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
   Output_Path='Outputs_RE_Pw'
   file = open(Output_Path, "r")
   line_count = 0
   for line in file:
       if line != "\n":
          line_count += 1      
   file.close()
   numbers_str=[None]*line_count #the lines will be stored inside "numbers_str"
   # we open again the file to get different results
   with open(Output_Path,"r") as f:
        i=-1
        for line in f:  
            i=i+1
            numbers_str[i] = line.split() #we read each line in the file and store its elements as strings in "numbers_str[i]"
       # store each line in the correct array 
       
       
   for i in range(len(time_interval)):
       for j in range(0,Nx):
           Vp_w_Time_RE[i][j]=float(numbers_str[i][j+1])    
#######################################File4
#######################################
   Vj_wn_Time_RE=np.zeros([len(time_interval),Nx], dtype=np.double)  

   # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
   Output_Path='Outputs_RE_Jwn'
   file = open(Output_Path, "r")
   line_count = 0
   for line in file:
       if line != "\n":
          line_count += 1      
   file.close()
   numbers_str=[None]*line_count #the lines will be stored inside "numbers_str"
   # we open again the file to get different results
   with open(Output_Path,"r") as f:
        i=-1
        for line in f:  
            i=i+1
            numbers_str[i] = line.split() #we read each line in the file and store its elements as strings in "numbers_str[i]"
       # store each line in the correct array 
   for i in range(len(time_interval)):
       for j in range(0,Nx):
           Vj_wn_Time_RE[i][j]=float(numbers_str[i][j+1])    
   
#####################################################################################################################
###############################################Visualization###########################################################
#####################################################################################################################
   mesh = np.linspace(Omega_0, Omega_L, Nx) # mesh just for plots
   mesh_exact =np.linspace(Omega_0, Omega_L, Nex) # mesh just for plots
   mpl.rcParams['text.usetex'] = True
   plt.rcParams.update({ "font.family": "monospace", "font.monospace": "Computer Modern Typewriter"})

    #####visualization  Volume fraction ew
   fs=40
   fs_legend=20
   fs_axes=16
   plt.xticks(fontsize=fs_axes)  # Increase font size of X-axis values
   plt.yticks(fontsize=fs_axes)  # Increase font size of Y-axis values
   plt.plot(mesh_exact,Ve_w_Time_Exact,label='Analytical solution; t(s)=100')
   for i in range(len(time_interval)):
       plt.plot(mesh,Ve_w_Time_TCAT[i][:],label='TCAT solution; t(s)=%g'%time_interval[i],linestyle='-', marker='o' )
       plt.plot(mesh,Ve_w_Time_RE[i][:],label='RE solution; t(s)=%g'%time_interval[i],linestyle="dashed")
       plt.legend(fontsize=fs_legend)
   plt.xlabel(r' $z$'r' $[m]$',fontsize=fs)
   plt.ylabel(r'$\epsilon^{\overline{\overline{w}}} $'r' $[-]$',fontsize=fs)
   plt.show()
    #####visualization  pressure pw
   plt.xticks(fontsize=fs_axes)  # Increase font size of X-axis values
   plt.yticks(fontsize=fs_axes)  # Increase font size of Y-axis values
   plt.plot(mesh_exact,Vp_w_Time_Exact,label='Analytical solution; t(s)=100')
   for i in range(len(time_interval)):
       plt.plot(mesh,Vp_w_Time_TCAT[i][:],label='TCAT solution; t(s)=%g'%time_interval[i],linestyle='-', marker='o')
       plt.plot(mesh,Vp_w_Time_RE[i][:],label='RE solution; t(s)=%g'%time_interval[i],linestyle="dashed")
       plt.legend(fontsize=fs_legend)
   
   plt.xlabel(r'$z$'r' $[m]$',fontsize=fs)
   plt.ylabel(r' $p^w$'r' $[Pa]$',fontsize=fs)
   plt.show()       
    #####visualization   Ewn
   plt.xticks(fontsize=fs_axes)  # Increase font size of X-axis values
   plt.yticks(fontsize=fs_axes)  # Increase font size of Y-axis values
   plt.plot(mesh_exact,Ve_wn_Time_Exact,label='Analytical solution; t(s)=100')
   for i in range(len(time_interval)):
       plt.plot(mesh,Ve_wn_Time_TCAT[i][:],label='TCAT solution; t(s)=%g'%time_interval[i],linestyle='-', marker='o')
       plt.legend(fontsize=fs_legend)
   plt.xlabel(r'$z$'r' $[m]$',fontsize=fs)
   plt.ylabel(r'$\epsilon^{\overline{\overline{wn}}} $'r' $[1/m]$',fontsize=fs)
   plt.show()    
    #####visualization   Jwn
   plt.xticks(fontsize=fs_axes)  # Increase font size of X-axis values
   plt.yticks(fontsize=fs_axes)  # Increase font size of Y-axis values
   plt.plot(mesh_exact,Vj_wn_Time_Exact,label='Analytical solution; t(s)=100')
   for i in range(len(time_interval)):
       plt.plot(mesh,Vj_wn_Time_TCAT[i][:],label='TCAT solution; t(s)=%g'%time_interval[i],linestyle='-', marker='o')
       plt.plot(mesh,Vj_wn_Time_RE[i][:],label='RE solution; t(s)=%g'%time_interval[i],linestyle="dashed")
       plt.legend(fontsize=fs_legend)
   plt.xlabel(r'$z$'r' $[m]$',fontsize=fs)
   plt.ylabel(r'$J^{wn}_n$'r' $[1/m]$',fontsize=fs)
   plt.show()
    #####visualization   Wwn
   plt.xticks(fontsize=fs_axes)  # Increase font size of X-axis values
   plt.yticks(fontsize=fs_axes)  # Increase font size of Y-axis values
   plt.plot(mesh_exact,Vw_wn_Time_Exact,label='Analytical solution; t(s)=100')
   for i in range(len(time_interval)):
       plt.plot(mesh,Vw_wn_Time_TCAT[i][:],label='TCAT solution; t(s)=%g'%time_interval[i],linestyle='-', marker='o')
       plt.legend(fontsize=fs_legend)
   plt.xlabel(r'$z$'r' $[m]$',fontsize=fs)
   plt.ylabel(r'$w^{wn}_z$'r' $[m/s]$',fontsize=fs)
   plt.show()
    #####visualization   Kwn
   plt.xticks(fontsize=fs_axes)  # Increase font size of X-axis values
   plt.yticks(fontsize=fs_axes)  # Increase font size of Y-axis values
   plt.plot(mesh,Vk_wn_Time_Exact,label='Analytical solution; t(s)=100')
   for i in range(len(time_interval)):
       plt.plot(mesh,Vk_wn_Time_TCAT[i][:],label='TCAT solution; t(s)=%g'%time_interval[i],linestyle='-', marker='o')
       plt.legend(fontsize=fs_legend)
   plt.xlabel(r'$z$'r' $[m]$',fontsize=fs)
   plt.ylabel(r'$K^{wn}_n$'r' $[1/m^2]$',fontsize=fs)
   plt.show()
   #######################################
if __name__ == "__main__":
    main()
