import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
#######################################

OptDB = PETSc.Options()
Formulation = OptDB.getString('Formulation', 'SPP')
Paper = OptDB.getString('Paper', 'Helmig') # get paper name from terminal so we can decide which inputs and PSK model to use
if(Paper=='Fucik'):
   sys.path.insert(0, './inputs/inputs_Fucik')
   import inputs as inp # import the inupts.py file as a module so we can access directly the parameters defined inside it Genuchten
elif( Paper=='Helmig'):
   sys.path.insert(0, './inputs/inputs_Helmig')  
   import inputs as inp # import the inupts.py file as a module so we can access directly the parameters defined inside it
elif( Paper=='Malgorzata'):
   sys.path.insert(0, './inputs/inputs_Malgorzata')  
   import inputs as inp # import the inupts.py file as a module so we can access directly the parameters defined inside it

#######################################
Ew_bc_0= inp.Ew_bc_0
Ew_bc_L= inp.Ew_bc_L
Pw_bc_0= inp.Pw_bc_0
Pw_bc_L= inp.Pw_bc_L
Pn_bc_0= inp.Pn_bc_0
Pn_bc_L= inp.Pn_bc_L
Ew_IC= inp.Ew_IC
Pw_IC= inp.Pw_IC
Pn_IC= inp.Pn_IC
Nz=inp.Nz
#######################################

def Initial_Conditions(ts,U):
# a private function that we created to initialize the solution vector at t=t0
    da=ts.getDM()
    u=da.getVecArray(U)
    if(Formulation=='SPP'):
        u[0][0] = Ew_bc_0
        u[0][1] = Pw_bc_0 
        u[0][2] = Pn_bc_0
        u[Nz-1][0] = Ew_bc_L
        u[Nz-1][1] = Pw_bc_L 
        u[Nz-1][2] = Pn_bc_L
        for i in range(1,Nz-1):
            u[i][0] = Ew_IC
            u[i][1] = Pw_IC
            u[i][2] = Pn_IC
    elif(Formulation=='SP'):
        u[0][0] = Ew_bc_0
        u[0][1] = Pw_bc_0 
        u[Nz-1][0] = Ew_bc_L
        u[Nz-1][1] = Pw_bc_L 
        for i in range(1,Nz-1):
            u[i][0] = Ew_IC
            u[i][1] = Pw_IC
    elif(Formulation=='PP'):
        u[0][0] = Pw_bc_0
        u[0][1] = Pn_bc_0 
        u[Nz-1][0] = Pw_bc_L
        u[Nz-1][1] = Pn_bc_L 
        for i in range(1,Nz-1):
            u[i][0] = Pw_IC
            u[i][1] = Pn_IC 
            
            
            
