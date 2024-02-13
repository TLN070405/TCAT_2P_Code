import sys
def ImportModules(Test_Case):
    if(Test_Case=='Test_Case1'):
       sys.path.insert(0, './TCAT_CGP_Subcodes/Inputs/Test_Case1')
       import Inputs1 as inp # import the inputs.py file as a module so we can access directly the parameters defined inside it
       import BC_IC1  as BIC
    elif( Test_Case=='Test_Case2'):
       sys.path.insert(0, './TCAT_CGP_Subcodes/Inputs/Test_Case2') 
       import Inputs2 as inp # import the inputs.py file as a module so we can access directly the parameters defined inside it
       import BC_IC2  as BIC
    elif( Test_Case=='Test_Case3'):
       sys.path.insert(0, './TCAT_CGP_Subcodes/Inputs/Test_Case3') 
       import Inputs3 as inp # import the inputs.py file as a module so we can access directly the parameters defined inside it
       import BC_IC3  as BIC  
    elif( Test_Case=='Test_Case4'):
       sys.path.insert(0, './TCAT_CGP_Subcodes/Inputs/Test_Case4')
       import Inputs4 as inp # import the inputs.py file as a module so we can access directly the parameters defined inside it
       import BC_IC4  as BIC
       
       
    ####This modules are the same for all test cases #### 
    sys.path.insert(0, './TCAT_CGP_Subcodes/Create_Arrays')
    import Create_Arrays as CA
    sys.path.insert(0, './TCAT_CGP_Subcodes/State_Equations')
    import State_Equations as SE
    sys.path.insert(0, './TCAT_CGP_Subcodes/Spacial_Discr')
    import Spacial_Discr_CGP as SD
    sys.path.insert(0, './TCAT_CGP_Subcodes/Export_Results')
    import Export_Results as ER
    return inp,BIC,CA,SE,SD,ER
