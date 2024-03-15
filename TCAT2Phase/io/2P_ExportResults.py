
def Export_Outputs(Mw0,Error_Mass,Paper,filenumber,Nz,Omega_0,Omega_L,dz,tau,K,time_interval,Clock_Array,Mw,VEpsilonw):
    print("Initial_Mass=")
    print(Mw0)
    print("Error mass =") ## print error mass on screen
    print(Error_Mass)
    ###### write into external file the results
    Output_Path="2P_Subcodes/Outputs/Outputs_%s"%Paper +'/OUTPUT_%s'%Paper +'_SPP_1D_%d.txt'%filenumber #the name of the output file change with the paper.
    file = open(Output_Path, "w")
    file.write('Nz= %g ' %Nz)
    file.write('\nOmega_0= %g ' %Omega_0)
    file.write('\nOmega_L= %g ' %Omega_L)
    file.write('\ndz= %g ' %dz)
    file.write('\ntau= %g ' %tau)
    file.write('\nK= %g ' %K)
    file.write('\nOutputTime(s)= ') 
    for i in range(len(time_interval)):
        file.write('%f ' %time_interval[i])
    file.write('\nCPU(s)= ')    
    for i in range(len(time_interval)):
        file.write('%f ' %Clock_Array[i])
    file.write('\nMw= ') 
    for i in range(len(time_interval)):
        file.write('%f ' %Mw[i])   
    for i in range(len(time_interval)): 
        file.write('\nEpsilonw= ')        
        for k in range(0,Nz):
            file.write('%f ' %VEpsilonw[i][k])
    file.close()        
