import h5py
import numpy as np
from pathlib import Path

def save_solutions_to_hdf5(path, filename, mesh, solutions, metadata=None):
    """
    Save the results of "solutions" into HDF5 format, including optional metadata.
    
    Parameters:
    - path: directory to save the file
    - filename: name of the HDF5 file
    - mesh: mesh object with .points
    - solutions: dictionary with time steps as keys and variable dictionaries as values
    - metadata: (optional) dictionary with extra information to store
    """
    path = Path(path)
    file_path = path / filename
    print(f"Saving file to: {file_path}")

    with h5py.File(file_path, 'w') as f:
        # Save mesh points
        f.create_dataset("mesh_points", data=np.array(mesh.points))

        # Save solutions
        for time, variables in solutions.items():
            time_group = f.create_group(f"solution_{time}")
            for var_name, data in variables.items():
                time_group.create_dataset(var_name, data=np.array(data))

        # Save time steps
        time_steps = list(solutions.keys())
        f.create_dataset("time_steps", data=np.array(time_steps))

        # Save metadata if provided
        if metadata is not None:
            meta_group = f.create_group("metadata")
            for key, value in metadata.items():
                # Try to store metadata as attributes or datasets
                try:
                    meta_group.attrs[key] = value  # Best for strings, numbers
                except TypeError:
                    # If value is a list or array, store it as a dataset
                    meta_group.create_dataset(key, data=np.array(value))

def load_solutions_from_hdf5(path, filename):
    """
    Load results and metadata from an HDF5 file into dictionaries.
    
    Parameters:
    - path: Directory path where the HDF5 file is located.
    - filename: Name of the HDF5 file to load.
    
    Returns:
    - solutions: Dictionary containing time steps and corresponding variables.
    - mesh_points: Numpy array of mesh points.
    - metadata: Dictionary containing extra information (if any).
    """
    file_path = Path(path) / filename
    print(f"Loading file from: {file_path}")
    
    solutions = {}
    metadata = {}
    mesh_points = None

    with h5py.File(file_path, 'r') as f:
        # Load the mesh points
        mesh_points = f["mesh_points"][:]

        # Load the time steps
        time_steps = f["time_steps"][:]

        # Iterate over the time steps and extract variables
        for time in time_steps:
            time_group = f[f"solution_{time}"]
            variables = {var_name: time_group[var_name][:] for var_name in time_group}
            solutions[time] = variables

        # Load metadata if it exists
        if "metadata" in f:
            meta_group = f["metadata"]
            # First load attributes
            for key, value in meta_group.attrs.items():
                metadata[key] = value
            # Then load any datasets (arrays/lists)
            for key in meta_group.keys():
                metadata[key] = meta_group[key][()]
                # Convert byte strings if needed
                if isinstance(metadata[key], bytes):
                    metadata[key] = metadata[key].decode('utf-8')

    return solutions, mesh_points, metadata