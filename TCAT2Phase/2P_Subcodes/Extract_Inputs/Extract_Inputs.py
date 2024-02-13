def ExtractInputs(inp,OptDB):
	Nz=inp.Nz
	t0=inp.t0
	time_interval=inp.time_interval
	Omega_0= inp.Omega_0
	Omega_L= inp.Omega_L
	rtol=inp.rtol
	atol= inp.atol
	MaxStepRejections=inp.MaxStepRejections
	MaxSNESFailures=inp.MaxSNESFailures  
	MaxTimeSteps=inp.MaxTimeSteps 
	phi= inp.phi
	g= inp.g
	d=inp.d
	tau=  inp.tau
	K= inp.K
	mu_w=inp.mu_w 
	Ew_bc_0= inp.Ew_bc_0
	Ew_bc_L=  inp.Ew_bc_L
	Pw_bc_0= inp.Pw_bc_0
	Pw_bc_L= inp.Pw_bc_L
	Pn_bc_0= inp.Pn_bc_0
	Pn_bc_L= inp.Pn_bc_L
	Ew_IC= inp.Ew_IC
	Pw_IC= inp.Pw_IC
	Pn_IC= inp.Pn_IC
	dz =inp.dz  
	Cz = inp.Cz
	k_mun_half=inp.k_mun_half
	k_muw_half=inp.k_muw_half
	K  = OptDB.getScalar('K', K) # here K is equal to K in the inputs but will change if we change it from terminal.
	tau = OptDB.getScalar('tau', tau) #here tau is equal to tau in the inputs but will change if we change it from terminal.
	filenumber = OptDB.getInt('filenumber', 1) # here filenumber is to be read from terminal, if its not provided a value of 1 is attributed.
	
	return Nz,t0,time_interval,Omega_0,Omega_L,rtol,atol,MaxStepRejections,MaxSNESFailures,MaxTimeSteps,phi,g,d,tau,K,mu_w,Ew_bc_0,Ew_bc_L,Pw_bc_0,Pw_bc_L,Pn_bc_0,Pn_bc_L,Ew_IC,Pw_IC,Pn_IC,dz ,Cz,k_mun_half,k_muw_half,filenumber
