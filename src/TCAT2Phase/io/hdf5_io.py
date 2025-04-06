import h5py
import numpy as np
from pathlib import Path

def save_solutions_to_hdf5(path, filename, mesh,solutions):
    """
    function to save the results of "solutions" into hdf5 format 
    """
    # Get the directory where application.py is located
    #app_dir = Path(__file__).parent  # This gets the path of application.py

    # Create the full path to save the HDF5 file in the same directory
    path = Path(path)  # This will be a relative path
    file_path = path / filename
    print(f"Saving file to: {file_path}")

    with h5py.File(file_path, 'w') as f:
        # Save the mesh points
        f.create_dataset("mesh_points", data=np.array(mesh.points))
        # Iterate over the solutions dictionary and store each time step
        for time, variables in solutions.items():
            # Create a group for each time step
            time_group = f.create_group(f"solution_{time}")
            for var_name, data in variables.items():
                time_group.create_dataset(var_name, data=np.array(data))

        # Optionally, save the time steps as a separate dataset for easy access
        time_steps = list(solutions.keys())
        f.create_dataset("time_steps", data=np.array(time_steps))

def load_solutions_from_hdf5(path, filename):
    """
    Function to load the results from an HDF5 file into a dictionary using a relative path.
    
    Parameters:
    - path: Directory path where the HDF5 file is located.
    - filename: Name of the HDF5 file to load.
    
    Returns:
    - solutions: Dictionary containing time steps and corresponding variables.
    """
    # Get the full path to the HDF5 file
    file_path = Path(path) / filename
    print(f"Loading file from: {file_path}")
    
    solutions = {}
    mesh_points = None
    
    with h5py.File(file_path, 'r') as f:
        # Load the mesh points
        mesh_points = f["mesh_points"][:]
        
        # Load the time steps
        time_steps = f["time_steps"][:]
        
        # Iterate over the time steps and extract the variables
        for time in time_steps:
            time_group = f[f"solution_{time}"]
            variables = {var_name: time_group[var_name][:] for var_name in time_group}
            solutions[time] = variables
    
    return solutions, mesh_points
