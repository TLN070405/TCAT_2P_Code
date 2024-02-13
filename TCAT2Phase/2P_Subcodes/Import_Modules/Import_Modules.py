import sys
def ImportModules(Paper):
    if(Paper=='Fucik'):
       sys.path.insert(0, './2P_Subcodes/inputs/inputs_Fucik')
       import inputs as inp # import the inputs.py file as a module so we can access directly the parameters defined inside it
       sys.path.insert(0, './2P_Subcodes/PSK_Models')
       import Brooks_Corey_Model as PSKM # PSKM: pressure-saturation-permeability model, if the paper is Fucik choose Brooks model and not Genuchten
    elif( Paper=='Helmig'):
       sys.path.insert(0, './2P_Subcodes/inputs/inputs_Helmig')  
       import inputs as inp # import the inputs.py file as a module so we can access directly the parameters defined inside it
       sys.path.insert(0, './2P_Subcodes/PSK_Models')
       import Van_Genuchten_Model as PSKM  # if the paper is Helmig choose Van-Genuchten model
    elif( Paper=='Malgorzata'):
       sys.path.insert(0, './2P_Subcodes/inputs/inputs_Malgorzata')  
       import inputs as inp # import the inputs.py file as a module so we can access directly the parameters defined inside it
       sys.path.insert(0, './2P_Subcodes/PSK_Models')
       import Van_Genuchten_Model as PSKM  # if the paper is Malgorzata choose Van-Genuchten model

    sys.path.insert(0, './2P_Subcodes/Mass_Calculation')
    import Mass_Calculation as MC # import mass calculation module
    sys.path.insert(0, './2P_Subcodes/Initialization')
    import Initialization as IC   # import initial conditions module
    sys.path.insert(0, './2P_Subcodes/Export_Results')
    import ExportResults as ER   # import Export Results module
    sys.path.insert(0, './2P_Subcodes/Extract_Inputs')
    import Extract_Inputs as EI  # import Extract inputs module
    sys.path.insert(0, './2P_Subcodes/Create_Arrays')
    import Create_Arrays as CA   # import Create Arrays module
    sys.path.insert(0, './2P_Subcodes/Spacial_Discr')
    import Spacial_Discr as SD   # import Spacial discretization module
    
    return inp,PSKM,MC,IC,ER,EI,CA,SD
