import petsc4py # we called petsc here just to use the PETSc.Options() function; to get Files_number from terminal
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
petsc4py.init(sys.argv)
from petsc4py import PETSc
OptDB      = PETSc.Options() #get access to options of petsc throw terminal or cmd, this help us to change or define some parameters from cmd.
Test_Case  = OptDB.getString('Test_Case', 'Test_Case4')
filenumber = OptDB.getInt('filenumber', 1)

   

#######################################
def main():
   # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
   Output_Path="TCAT_CGP_Subcodes/Outputs/Outputs_TCAT_CGP_%s"%Test_Case +'_file%d.txt'%filenumber
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
   Nx=int(numbers_str[0][1]) # the first index in numbers_str is the line number and the second index is the value; we starts from 1 in the second index because 0 corresponds to the name of values (see output.txt)
   Omega_0= float(numbers_str[1][1]) 
   Omega_L= float(numbers_str[2][1])  
   N_Output_Time=len(numbers_str[3])-1

   time_interval        = np.zeros([N_Output_Time], dtype=np.double) 
   Ve_w_Time_Results    = np.zeros([len(time_interval),Nx], dtype=np.double) 
   Vp_w_Time_Results    = np.zeros([len(time_interval),Nx], dtype=np.double) 
   Ve_wn_Time_Results   = np.zeros([len(time_interval),Nx], dtype=np.double) 
   Vj_wn_Time_Results   = np.zeros([len(time_interval),Nx], dtype=np.double) 
   Vw_xwn_Time_Results  = np.zeros([len(time_interval),Nx], dtype=np.double) 
   Vkn_wn_Time_Results  = np.zeros([len(time_interval),Nx], dtype=np.double) 
   ##################      
   for i in range(len(time_interval)):
       time_interval[i]=float(numbers_str[3][i+1])  #transform strings to floats 
   ##################    
   for i in range(len(time_interval)):
        for j in range(0,Nx):
             Ve_w_Time_Results[i][j]=float(numbers_str[4+i][j+1])
   ##################    
   for i in range(len(time_interval)):
        for j in range(0,Nx):
             Vp_w_Time_Results[i][j]=float(numbers_str[4+1*len(time_interval)+i][j+1])
   ##################    
   for i in range(len(time_interval)):
        for j in range(0,Nx):
             Ve_wn_Time_Results[i][j]=float(numbers_str[4+2*len(time_interval)+i][j+1])
   ##################    
   for i in range(len(time_interval)):
        for j in range(0,Nx):
             Vj_wn_Time_Results[i][j]=float(numbers_str[4+3*len(time_interval)+i][j+1])     
   ##################    
   for i in range(len(time_interval)):
        for j in range(0,Nx):
             Vw_xwn_Time_Results[i][j]=float(numbers_str[4+4*len(time_interval)+i][j+1])    
   ##################    
   for i in range(len(time_interval)):
        for j in range(0,Nx):
             Vkn_wn_Time_Results[i][j]=float(numbers_str[4+5*len(time_interval)+i][j+1])     
             
##################PLot results##################
   mesh = np.linspace(Omega_0, Omega_L, Nx)
   #####visualization  Volume fraction ew     
   for i in range(len(time_interval)):
        plt.plot(mesh,Ve_w_Time_Results[i][:],label='t=%5.4f'%time_interval[i])
        plt.legend()
   plt.xlabel("x (m)")
   plt.ylabel("ew")
   plt.title("TCAT CGP Solution ")
   plt.show()  
    #####visualization  pressure pw    
   for i in range(len(time_interval)):
        plt.plot(mesh,Vp_w_Time_Results[i][:],label='t=%5.4f'%time_interval[i])
        plt.legend()
   plt.xlabel("x (m)")
   plt.ylabel("Pw")
   plt.title("TCAT CGP Solution ")
   plt.show()       
    #####visualization  interfacial area ewn     
   for i in range(len(time_interval)):
        plt.plot(mesh,Ve_wn_Time_Results[i][:],label='t=%5.4f'%time_interval[i])
        plt.legend()
   plt.xlabel("x (m)")
   plt.ylabel("ewn")
   plt.title("TCAT CGP Solution ")
   plt.show()  
   #####visualization  mean curvature jwn    
   for i in range(len(time_interval)):
        plt.plot(mesh,Vj_wn_Time_Results[i][:],label='t=%5.4f'%time_interval[i])
        plt.legend()
   plt.xlabel("x (m)")
   plt.ylabel("Jwn")
   plt.title("TCAT CGP Solution ")
   plt.show()  
   #####visualization  normal interface velocity w_xwn
   for i in range(len(time_interval)):
        plt.plot(mesh,Vw_xwn_Time_Results[i][:],label='t=%5.4f'%time_interval[i])
        plt.legend()
   plt.xlabel("x (m)")
   plt.ylabel("wx_wn")
   plt.title("TCAT CGP Solution ")
   plt.show()  
   #####visualization  Gaussian curvature kn_wn    
   for i in range(len(time_interval)):
        plt.plot(mesh,Vkn_wn_Time_Results[i][:],label='t=%5.4f'%time_interval[i])
        plt.legend()
   plt.xlabel("x (m)")
   plt.ylabel("Kn_wn")
   plt.title("TCAT CGP Solution ")
   plt.show()  
#######################################

if __name__ == "__main__":
    main()      

