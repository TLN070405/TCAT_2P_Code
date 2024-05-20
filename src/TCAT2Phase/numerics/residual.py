import numpy as np
from TCAT2Phase.state_equations import state_equations
from TCAT2Phase.state_equations import state_equations_hysteresis
from TCAT2Phase.numerics import utils

def model_residual(ts,t,U,Udot,F,models):
    """
    Residual function for the selected model. 
    """
    global counter_hysteresis

    da = ts.getDM()
    localU = da.getLocalVec() 
    da.globalToLocal(U,localU) 
    
    # Collect solution variables and residual 
    u = da.getVecArray(localU)
    udot = da.getVecArray(Udot)
    f = da.getVecArray(F)
    
    model_residual_internal(f,u,udot,models)
    model_residual_boundary(t,f,u,models)
    model_hysteresis(ts,counter_hysteresis,models)


def model_residual_internal(f,u,udot,models):
    """
    residual at internal nodes.
    """
    model_imb = models[0]
    model_dr  = models[1]
    # Collect hysteresis data arrays

    e_w,p_w,p_n,p_c = model_primary_variables(u,models)
    mesh = model_imb.mesh
    
    # Constant Pameters
    k = model_imb.solid.permeability
    gravity =model_imb.solid.gravity
    mu_w = model_imb.w_phase.viscosity
    mu_n = model_imb.n_phase.viscosity
    porosity = model_imb.solid.porosity

    # Cell Centered Variables
    rho_w = utils.cell_center(p_w,model_imb.w_phase.density_function,mesh)
    rho_n = utils.cell_center(p_n,model_imb.n_phase.density_function,mesh)
    e_n   = porosity - e_w
    k_rw  = utils.cell_center(p_c,model_imb.w_phase.rel_perm_function,mesh)
    k_rn  = utils.cell_center(p_c,model_imb.n_phase.rel_perm_function,mesh)

    # Face Centered Values
    del_pw = utils.gradient(p_w,mesh)
    del_pn = utils.gradient(p_n,mesh)

    rho_w_face = utils.interpolate(rho_w,mesh)
    rho_n_face = utils.interpolate(rho_n,mesh)

    q_w = darcy_velocity(
                   mesh,
                   k,
                   utils.interpolate(k_rw*rho_w,mesh),
                   mu_w,
                   del_pw,
                   rho_w_face,
                   gravity)

    q_n = darcy_velocity(
                   mesh,
                   k,
                   utils.interpolate(k_rn*rho_n,mesh),
                   mu_n,
                   del_pn,
                   rho_n_face,
                   gravity)

    num_u = model_imb.model.num_u
    
    drainage_new    = model_imb.data_hysteresis[1]
    ew_reversal     = model_imb.data_hysteresis[4]
    ew_endpoint     = model_imb.data_hysteresis[5]
    pc_reversal     = model_imb.data_hysteresis[6]
    pc_endpoint     = model_imb.data_hysteresis[7]

    for cell in mesh.cells:
        if cell.boundary:
            continue
           
        n = cell.ID
        od_w  = 1./cell.length*(q_w[n+1] - q_w[n])
        od_n  = 1./cell.length*(q_n[n+1] - q_n[n])
        od_dp = p_w[n]-p_n[n]+ p_c[n]
        od    = [od_dp,od_w[0],od_n[0]]
        
        # Time derivative matrix
        data_d_pc_hysteresis=[drainage_new[n],ew_reversal[n],ew_endpoint[n],pc_reversal[n],pc_endpoint[n] ]
        A = model_A_matrix(e_w[n],p_w[n],p_n[n],p_c[n],rho_w[n],rho_n[n],e_n[n],data_d_pc_hysteresis,models)
        # residuals
        f[n] = np.dot(A, udot[n]) -  od[ int(num_u*(-0.5*num_u+1.5)):int(num_u*(-0.5*num_u+2.5))]
    


def model_primary_variables(u,models):
    """
    primary variables for each formulation
    """
    model_imb = models[0]
    model_dr  = models[1]
    nodes = model_imb.nodes
    form  = model_imb.model.name
    if form == 'extended-two-phase-spp':
        # Solution Variables
        e_w=u[:,0]
        p_w=u[:,1]
        p_n=u[:,2]
        # Primary Variables 
        p_c = np.zeros_like(p_w)
        for n in range(0,nodes):
            p_c[n] = model_imb.interface.pc_function(e_w[n])
    elif form == 'two-phase-sp':
        # Solution Variables
        e_w= u[:,0]
        p_w= u[:,1]
        # Primary Variables 
        p_c = np.zeros_like(p_w)
        p_n = np.zeros_like(p_w)

        drainage        = model_imb.data_hysteresis[1]
        ew_reversal     = model_imb.data_hysteresis[4]
        ew_endpoint     = model_imb.data_hysteresis[5]
        pc_reversal     = model_imb.data_hysteresis[6]
        pc_endpoint     = model_imb.data_hysteresis[7]
        for n in range(0,nodes):
            data_pc_hysteresis=[drainage[n],ew_reversal[n],ew_endpoint[n],pc_reversal[n],pc_endpoint[n] ]
            p_c[n] = model_imb.interface.pc_hysteretic(model_imb,model_dr, e_w[n], data_pc_hysteresis)
            p_n[n] = p_w[n] + p_c[n]
    elif form == 'two-phase-pp':
        # Solution Variables
        p_w = u[:,0]
        p_n = u[:,1]
        # Primary Variables 
        p_c = np.zeros_like(p_w)
        e_w = np.zeros_like(p_w)
        for n in range(0,nodes):
            p_c[n] = p_n[n] - p_w[n]
            e_w[n] = model_imb.w_phase.volume_fraction_function(p_c[n])
    elif form == 'richards-pr':
        # Solution Variables
        p_w = u[:]
        # Primary Variables 
        p_c = np.zeros_like(p_w)
        e_w = np.zeros_like(p_w)
        p_n = np.zeros_like(p_w)
        for n in range(0,nodes):
            p_c[n] = - p_w[n]
            e_w[n] = model_imb.w_phase.volume_fraction_function(p_c[n])
    else:
        print('Error Model Form is unknown')

    return e_w,p_w,p_n,p_c

def model_A_matrix(e_w,p_w,p_n,p_c,rho_w,rho_n,e_n,data_d_pc_hysteresis,models):
    """
    A matrix components for each formulation
    """
    model_imb = models[0]
    model_dr  = models[1]

  

    d_rho_n_p_n = model_imb.n_phase.d_density_function(p_n)
    d_rho_w_p_w = model_imb.w_phase.d_density_function(p_w)
    d_p_c_e_w   = model_imb.interface.d_pc_function(e_w)
    d_e_w_p_c   = model_imb.w_phase.d_volume_fraction_function(p_c)
    
    A = np.zeros([model_imb.model.num_u,model_imb.model.num_u])

    form = model_imb.model.name
    
    if form == 'extended-two-phase-spp':
        A[0,0] = model_imb.solid.dynamic_capillary_pressure_parameter/model_imb.solid.porosity
        A[1,0] = rho_w                   
        A[1,1] = e_w*d_rho_w_p_w     
        A[2,0] = -rho_n  
        A[2,2] = e_n*d_rho_n_p_n
    elif form == 'two-phase-sp':
        d_p_c_e_w   = model_imb.interface.d_pc_hysteretic(model_imb,model_dr, e_w, data_d_pc_hysteresis)
        A[0,0] = rho_w
        A[0,1] = e_w*d_rho_w_p_w
        A[1,0] = e_n*d_rho_n_p_n*d_p_c_e_w - rho_n
        A[1,1] = e_n*d_rho_n_p_n
    elif form == 'two-phase-pp':
        A[0,0] = -rho_w*d_e_w_p_c + e_w*d_rho_w_p_w
        A[0,1] = rho_w*d_e_w_p_c
        A[1,0] = rho_n*d_e_w_p_c
        A[1,1] = -rho_n*d_e_w_p_c + e_n*d_rho_n_p_n
    elif form == 'richards-pr':
        A[0,0] = -rho_w*d_e_w_p_c + e_w*d_rho_w_p_w
    else:
        print('Error Model Form is unknown')

    return A    

def model_residual_boundary(t,f,u,model):
    """
    boundary residuals 
    """
    nodes = model.nodes
    u_names= model.model.u_names
    boundary_type  = model.boundary_type
    boundary_name  = model.boundary_name
    boundary_value = model.boundary_value


    e_w,p_w,p_n,p_c = model_primary_variables(u,model)
    mesh = model.mesh
    
    # Constant Pameters
    k = model.solid.permeability
    gravity =model.solid.gravity
    mu_w = model.w_phase.viscosity
    mu_n = model.n_phase.viscosity
    porosity = model.solid.porosity

    # Cell Centered Variables
    rho_w = utils.cell_center(p_w,model.w_phase.density_function,mesh)
    rho_n = utils.cell_center(p_n,model.n_phase.density_function,mesh)
    k_rw  = utils.cell_center(p_c,model.w_phase.rel_perm_function,mesh)
    k_rn  = utils.cell_center(p_c,model.n_phase.rel_perm_function,mesh)

    # Face Centered Values
    del_pw = utils.gradient(p_w,mesh)
    del_pn = utils.gradient(p_n,mesh)

    rho_w_face = utils.interpolate(rho_w,mesh)
    rho_n_face = utils.interpolate(rho_n,mesh)

    q_w = darcy_velocity(
                   mesh,
                   k,
                   utils.interpolate(k_rw*rho_w,mesh),
                   mu_w,
                   del_pw,
                   rho_w_face,
                   gravity)

    q_n = darcy_velocity(
                   mesh,
                   k,
                   utils.interpolate(k_rn*rho_n,mesh),
                   mu_n,
                   del_pn,
                   rho_n_face,
                   gravity)

    flux_bc ={'q_w':[q_w[0],q_w[nodes-1]],
              'q_n':[q_n[0],q_n[nodes-1]]}
    
    if model.model.num_u == 1:
    # boundary conditions for one variable formulations   

       #Residual at Boundary 1 
       if(boundary_type[u_names[0]][0][0]==0):
          f[0]= u[0] - boundary_value[0][0][0](t)  
       elif(boundary_type[u_names[0]][0][0]==1):
          f[0]= flux_bc[boundary_name[0][0][0]][0] - boundary_value[0][0][0](t)
       else:
          print('Boundary type non supported')   
       # Residual at Boundary 2
       if(boundary_type[u_names[0]][0][1]==0):
          f[nodes-1] = u[nodes-1] - boundary_value[0][0][1](t)  
       elif(boundary_type[u_names[0]][0][1]==1):
          f[nodes-1] =  flux_bc[boundary_name[0][0][1]][1] - boundary_value[0][0][1](t) 
       else:
          print('Boundary type non supported')   
       
    else:
    # boundary conditions for two or more variables formulations    
    
       for n in range(model.model.num_u):
           #Residual at Boundary 1 
           if(boundary_type[u_names[n]][0][0]==0):
              f[0][n]= u[0][n] - boundary_value[n][0][0](t)  
           elif(boundary_type[u_names[n]][0][0]==1):
              f[0][n]= flux_bc[boundary_name[n][0][0]][0] - boundary_value[n][0][0](t) 
           else:
               print('Boundary type non supported')   
           # Residual at Boundary 2
           if(boundary_type[u_names[n]][0][1]==0):
              f[nodes-1][n]= u[nodes-1][n] - boundary_value[n][0][1](t)  
           elif(boundary_type[u_names[n]][0][1]==1):
              f[nodes-1][n]= flux_bc[boundary_name[n][0][1]][1] - boundary_value[n][0][1](t)
           else:
               print('Boundary type non supported')   


def model_hysteresis(ts,counter_hysteresis,models):
    """
    Hysteresis function
    """
    model_imb = models[0]
    model_dr  = models[1]
    
    drainage_old    = model_imb.data_hysteresis[0]
    drainage_new    = model_imb.data_hysteresis[1]
    pc_old          = model_imb.data_hysteresis[2]
    ew_old          = model_imb.data_hysteresis[3]
    ew_reversal     = model_imb.data_hysteresis[4]
    ew_endpoint     = model_imb.data_hysteresis[5]
    pc_reversal     = model_imb.data_hysteresis[6]
    pc_endpoint     = model_imb.data_hysteresis[7]
    pc_endpoint_old = model_imb.data_hysteresis[8]

    n_dt=ts.getStepNumber()
    nodes = model_imb.nodes
    if(n_dt==counter_hysteresis):
      solution_now = ts.getSolution()

      for n in range(0,nodes):  
          data_pc_hysteresis = [solution_now[2*n], drainage_new[n], ew_reversal[n], ew_endpoint[n], pc_reversal[n], pc_endpoint[n]]
          pc_old[n]   = model_imb.interface.pc_hysteretic(model_imb,model_dr, data_pc_hysteresis)
          if( solution_now[2*n]-ew_old[n]>1e-9 ):
            drainage_old[n] = drainage_new[n]
            drainage_new[n] = -1
          else:      
            drainage_old[n] = drainage_new[n]
            drainage_new[n] = 1
          #####  
          ew_old[n]=solution_now[2*n] 
          #####
          if(drainage_new[n] != drainage_old[n]): 
            data_pc_hysteresis = [solution_now[2*n], drainage_old[n], ew_reversal[n], ew_endpoint[n], pc_reversal[n], pc_endpoint_old[n]]
            pc_endpoint[n]     = pc_reversal[n] 
            pc_reversal[n]     = model_imb.interface.pc_hysteretic(model_imb,model_dr, data_pc_hysteresis)
            pc_endpoint_old[n] = pc_endpoint[n]
            ew_endpoint[n]     = ew_reversal[n]
            ew_reversal[n]     = solution_now[2*n]

      counter_hysteresis=counter_hysteresis+1


def darcy_velocity(mesh,k,k_r,mu,del_p,rho,g):
    """
    Calaculate the Darcy velocity as:
        q^a = -k*k_r^a/mu^a(\grad p^a - rho^a*g)
    """
    q = np.zeros(mesh.num_faces)
    for face in mesh.faces:
        if face.boundary:
            continue
        q[face.ID] = k*k_r[face.ID]/mu*(del_p[face.ID] - rho[face.ID]*g)
    return q