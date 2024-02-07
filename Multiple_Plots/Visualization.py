import petsc4py # we called petsc here just to use the PETSc.Options() function; to get Files_number from terminal
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
petsc4py.init(sys.argv)
from petsc4py import PETSc
OptDB      = PETSc.Options() #get access to options of petsc throw terminal or cmd, this help us to change or define some parameters from cmd.
Test_Case  = OptDB.getString('Test_Case', 'Test_Case4')
filenumber_RE = OptDB.getInt('filenumber_RE', 1)
filenumber_CGP = OptDB.getInt('filenumber_CGP', 1)
 

#######################################
def main():
   ####Reading CGP file
   # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
   Output_Path="TCAT_CGP_Subcodes/Outputs/Outputs_TCAT_CGP_%s"%Test_Case +'_file%d.txt'%filenumber_CGP
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
   Nx_CGP=int(numbers_str[0][1]) # the first index in numbers_str is the line number and the second index is the value; we starts from 1 in the second index because 0 corresponds to the name of values (see output.txt)
   Omega_0_CGP= float(numbers_str[1][1]) 
   Omega_L_CGP= float(numbers_str[2][1])  
   N_Output_Time=len(numbers_str[3])-1

   time_interval_CGP        = np.zeros([N_Output_Time], dtype=np.double) 
   Ve_w_Time_Results_CGP    = np.zeros([len(time_interval_CGP),Nx_CGP], dtype=np.double) 
   Vp_w_Time_Results_CGP    = np.zeros([len(time_interval_CGP),Nx_CGP], dtype=np.double) 
   ##################      
   for i in range(len(time_interval_CGP)):
       time_interval_CGP[i]=float(numbers_str[3][i+1])  #transform strings to floats 
   ##################    
   for i in range(len(time_interval_CGP)):
        for j in range(0,Nx_CGP):
             Ve_w_Time_Results_CGP[i][j]=float(numbers_str[4+i][j+1])
   ##################    
   for i in range(len(time_interval_CGP)):
        for j in range(0,Nx_CGP):
             Vp_w_Time_Results_CGP[i][j]=float(numbers_str[4+1*len(time_interval_CGP)+i][j+1])

             
             
             
   ####Reading RE file          
   # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
   Output_Path="RE_Subcodes/Outputs/Outputs_RE_%s"%Test_Case +'_file%d.txt'%filenumber_RE
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
   Nx_RE=int(numbers_str[0][1]) # the first index in numbers_str is the line number and the second index is the value; we starts from 1 in the second index because 0 corresponds to the name (see output.txt)
   Omega_0_RE= float(numbers_str[1][1]) 
   Omega_L_RE= float(numbers_str[2][1])  
   N_Output_Time=len(numbers_str[3])-1

   time_interval_RE        = np.zeros([N_Output_Time], dtype=np.double) 
   Ve_w_Time_Results_RE    = np.zeros([len(time_interval_RE),Nx_RE], dtype=np.double) 
   Vp_w_Time_Results_RE    = np.zeros([len(time_interval_RE),Nx_RE], dtype=np.double) 
   ##################      
   for i in range(len(time_interval_RE)):
       time_interval_RE[i]=float(numbers_str[3][i+1])  #transform strings to floats 
   ##################    
   for i in range(len(time_interval_RE)):
        for j in range(0,Nx_RE):
             Ve_w_Time_Results_RE[i][j]=float(numbers_str[4+i][j+1])
   ##################    
   for i in range(len(time_interval_RE)):
        for j in range(0,Nx_RE):
             Vp_w_Time_Results_RE[i][j]=float(numbers_str[4+1*len(time_interval_RE)+i][j+1])
                 
##################PLot results##################
   mesh_CGP = np.linspace(Omega_0_CGP, Omega_L_CGP, Nx_CGP)
   mesh_RE    = np.linspace(Omega_0_RE, Omega_L_RE, Nx_RE)
   
   #####visualization  Volume fraction ew     
   for i in range(len(time_interval_CGP)):
        plt.plot(mesh_CGP,Ve_w_Time_Results_CGP[i][:],label='CGP t=%5.4f'%time_interval_CGP[i])
   for i in range(len(time_interval_RE)):
        plt.plot(mesh_RE,Ve_w_Time_Results_RE[i][:],label='RE t=%5.4f'%time_interval_RE[i])
   
   plt.legend()
   plt.xlabel("x (m)")
   plt.ylabel("ew")
   plt.title("TCAT CGP and RE Solution ")
   plt.show()  
    #####visualization  pressure pw    
   for i in range(len(time_interval_CGP)):
        plt.plot(mesh_CGP,Vp_w_Time_Results_CGP[i][:],label='CGP t=%5.4f'%time_interval_CGP[i])
   for i in range(len(time_interval_RE)):
        plt.plot(mesh_RE,Vp_w_Time_Results_RE[i][:],label='RE t=%5.4f'%time_interval_RE[i])
   plt.legend()
   plt.xlabel("x (m)")
   plt.ylabel("Pw")
   plt.title("TCAT CGP and RE Solution ")
   plt.show()       

#######################################

if __name__ == "__main__":
    main()      

