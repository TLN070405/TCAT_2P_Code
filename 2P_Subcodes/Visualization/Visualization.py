import petsc4py # we called petsc here just to use the PETSc.Options() function; to get Files_number from terminal
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
petsc4py.init(sys.argv)
from petsc4py import PETSc
OptDB = PETSc.Options() #get access to options of petsc throw terminal or cmd, this help us to change or define some parameters from terminal.
Files_number=OptDB.getInt('Files_number', 1) # here we get the Files_number from terminal (in script)
Paper = OptDB.getString('Paper', 'Helmig')  # here we get the paper name that we will solve, we need that to decide files to read.

if(Paper=='Fucik'):
   sys.path.insert(0, './2P_Subcodes/inputs/inputs_Fucik')
   import inputs as inp 
elif( Paper=='Helmig'):
   sys.path.insert(0, './2P_Subcodes/inputs/inputs_Helmig')  
   import inputs as inp 
elif( Paper=='Malgorzata'):
   sys.path.insert(0, './2P_Subcodes/inputs/inputs_Malgorzata')  
   import inputs as inp 
   
phi=inp.phi
#######################################
def main():
   Nz=np.empty([Files_number], dtype=int) # each file has Nz 
   Omega_0=np.empty([Files_number], dtype=float) # each file has Omega_0 
   Omega_L=np.empty([Files_number], dtype=float)
   dz=np.empty([Files_number], dtype=float)
   tau=np.empty([Files_number], dtype=float) # each file can has a different tau 
   K=np.empty([Files_number], dtype=float)  # K is the intrinsic permeability 
   for k in range(Files_number): 
       # first we open the file and get number of non null lines, (we need it to size the list "numbers_str")
       Output_Path="./2P_Subcodes/Outputs/Outputs_%s"%Paper +"/OUTPUT_%s"%Paper+"_SPP_1D_%d.txt"%(k+1)
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
       Nz[k]=int(numbers_str[0][1]) # the first index in numbers_str is the line number and the second index is the value; we starts from 1 in the second index because 0 corresponds to the name (see output.txt)
       Omega_0[k]= float(numbers_str[1][1]) 
       Omega_L[k]= float(numbers_str[2][1])  
       dz[k]= float(numbers_str[3][1])
       tau[k]= float(numbers_str[4][1])
       K[k]= float(numbers_str[5][1])
       N_Output_Time=len(numbers_str[6])-1
       
   OutputTime=np.empty([Files_number,N_Output_Time], dtype=float)    
   CPU=np.empty([Files_number,N_Output_Time], dtype=float) 
   Mw=np.empty([Files_number,N_Output_Time], dtype=float) 
   Epsilonw= np.empty([Files_number,N_Output_Time,Nz[0]], dtype=float) 
   Sn= np.empty([Files_number,N_Output_Time,Nz[0]], dtype=float)
   Sw= np.empty([Files_number,N_Output_Time,Nz[0]], dtype=float)
   for k in range(Files_number): 
       # first we open file and get number of non null lines, (we need it to size the list "numbers_str")
       Output_Path="./2P_Subcodes/Outputs/Outputs_%s"%Paper +"/OUTPUT_%s"%Paper+"_SPP_1D_%d.txt"%(k+1)
       file = open(Output_Path, "r")
       line_count = 0
       for line in file:
           if line != "\n":
              line_count += 1
       file.close()
       numbers_str=[None]*line_count
       # we open again the file to get different results
       with open(Output_Path,"r") as f:
            i=-1
            for line in f:  
                i=i+1
                numbers_str[i] = line.split() #we read each line in the file and store its elements as strings in "numbers_str[i]"
                
       for i in range(N_Output_Time):
           OutputTime[k][i]=float(numbers_str[6][i+1])  #transform strings to floats 
       for i in range(N_Output_Time):
           CPU[k][i]=float(numbers_str[7][i+1])  #transform strings to floats     
       for i in range(N_Output_Time):
           Mw[k][i]=float(numbers_str[8][i+1])  #transform strings to floats  
       for j in range(N_Output_Time):
           for i in range(0,Nz[0]):
               Epsilonw[k][j][i]=float(numbers_str[9+j][i+1])
               Sn[k][j][i]=1-(Epsilonw[k][j][i])/phi
               Sw[k][j][i]=(Epsilonw[k][j][i])/phi
##################PLot results##################
   
   if(Paper=='Helmig'):
     ################plot  Epsilonw vs Space 
     Mesh_Space= np.linspace(Omega_0[0], Omega_L[0], Nz[0])
     for k in range(Files_number):  
         for j in range(N_Output_Time):
             plt.plot(Mesh_Space, Epsilonw[k][j],label=" Time(s)=%f"%OutputTime[k][j])
             plt.legend()
         plt.xlabel("position (m)")
         plt.ylabel("Epsilon w")
         plt.title("%s"%Paper +" paper 1D tau=%g"%tau[k]+" K=%g"%K[k])
         plt.show()
 
     ################plot  Mass accumulation vs time    
     for k in range(Files_number):    
         plt.plot(OutputTime[k], Mw[k],label="tau=%g"%tau[k]+" K=%g"%K[k])
         plt.legend()
     plt.xlabel("Time (s)")
     plt.ylabel("Mass accumulation Mw")
     plt.title("Mass accumulation vs time")
     plt.show() 
 
   if(Paper=='Fucik'):
      Mesh_Space= np.linspace(Omega_0[0], Omega_L[0], Nz[0])
      for k in range(Files_number):  
          for j in range(N_Output_Time):
              plt.plot(Mesh_Space, Sn[k][j],label=" Time(s)=%f"%OutputTime[k][j])
              plt.legend()
          plt.xlabel("position (m)")
          plt.ylabel("Epsilon n")
          plt.title("%s"%Paper +" paper 1D tau=%g"%tau[k]+" K=%g"%K[k])
          plt.show()
   if(Paper=='Malgorzata'):
      Mesh_Space= np.linspace(Omega_0[0], Omega_L[0], Nz[0])
      for k in range(Files_number):  
          for j in range(N_Output_Time):
              plt.plot(Mesh_Space, Sw[k][j],label=" Time(s)=%f"%OutputTime[k][j])
              plt.legend()
          plt.xlabel("position (m)")
          plt.ylabel("Saturation w")
          plt.title("%s"%Paper +" paper 1D tau=%g"%tau[k]+" K=%g"%K[k])
          plt.show()
#######################################
   
if __name__ == "__main__":
    main()      

