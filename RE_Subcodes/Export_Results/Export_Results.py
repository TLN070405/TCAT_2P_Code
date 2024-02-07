
def Export_Results(inp,Test_Case,filenumber,Ve_w_Time_Results,Vp_w_Time_Results):    
    ###### write into external file the results
    Output_Path="RE_Subcodes/Outputs/Outputs_RE_%s"%Test_Case +'_file%d.txt'%filenumber
    ######
    Length=len(inp.time_interval)
    Nx=inp.Nx
    ######
    file = open(Output_Path, "w")
    file.write('Nx= %g ' %Nx)
    file.write('\nOmega_0= %g ' %inp.Omega_0)
    file.write('\nOmega_L= %g ' %inp.Omega_L)
    file.write('\nOutputTime(s)= ') 
    ######
    for i in range(Length):
        file.write('%f ' %inp.time_interval[i])
    ###### Export solutions
    for i in range(Length): 
        file.write('\ne_w= ')        
        for j in range(0,Nx):
            file.write('%f ' %Ve_w_Time_Results[i][j])
    ######        
    for i in range(Length): 
        file.write('\np_w= ')        
        for j in range(0,Nx):
            file.write('%f ' %Vp_w_Time_Results[i][j])
            
    file.close()        
