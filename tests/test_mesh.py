import numpy as np
import TCAT2Phase

# Domain
domain_size = np.array([[0.0,1.0]])
nodes = 101

domain =  TCAT2Phase.properties.domain.initialize_domain(
                            dims = 1,
                            domain_size = domain_size,
                            )


mesh = TCAT2Phase.mesh.grid.Mesh()
mesh.uniform_mesh_1d(domain,nodes)

print(mesh.cells)