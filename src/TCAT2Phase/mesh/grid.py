"""grid.py"""
import numpy as np
import dataclasses

class Points:
    """
    Mesh Points
    """
    def __init__(self,domain,num_points):
        self.domain = domain
        self.num_points = num_points
        self.points = None

    def gen_uniform_points(self):
        """
        Generate pointm locations
        """
        self.points = np.zeros([self.num_points,self.domain.dims])
        for n,domain_size in enumerate(self.domain.domain_size):
            self.points[:,n] = np.linspace(
                domain_size[0],
                domain_size[1],
                self.num_points
                )

        return self.points,self.num_points


@dataclasses.dataclass
class Cell:
    """
    Cell data
    """
    ID: int
    point_ID: int
    boundary: bool
    faces: list[int] = dataclasses.field(default_factory=list)
    neighbors: list[int] = dataclasses.field(default_factory=list)
    # distance to neighbor cell centroid
    distance: list[float] = dataclasses.field(default_factory=list)
    length: float = -1


class Cells:
    """
    Mesh cell 
    """
    def __init__(self,points):
        self.points = points
        self.cells = []

    def gen_uniform_cells(self):
        """
        Generate cells based on unifrom grid spacing
        """
        num_cells = len(self.points)
        for n_p,p in enumerate(self.points):
            if n_p == 0:
                _cell =  Cell( ID = n_p, point_ID = n_p, boundary = True)
                _cell.faces = [n_p, n_p+1]
                _cell.neighbors = [n_p-1, n_p+1]
                _cell.distance = [-1, self.points[n_p+1] - p ]
                _cell.length = self.points[n_p+1] - self.points[n_p]

            elif n_p == num_cells - 1:
                _cell =  Cell( ID = n_p, point_ID = n_p, boundary = True)
                _cell.faces = [n_p,n_p+1]
                _cell.neighbors = [n_p-1, -1]
                _cell.distance = [p - self.points[n_p-1], -1 ]
                _cell.length = self.points[n_p] - self.points[n_p-1]

            else:
                _cell =  Cell( ID = n_p, point_ID = n_p, boundary = False)
                _cell.faces = [n_p, n_p+1]
                _cell.neighbors = [n_p-1, n_p+1]
                _cell.distance = [p - self.points[n_p-1], self.points[n_p+1] - p ]
                _cell.length = self.points[n_p+1] - self.points[n_p]

            self.cells.append(_cell)

        return self.cells,num_cells


@dataclasses.dataclass
class Face:
    """
    Face data:
        ID: face ID
        boundary: boundary Face
        owner: Cell ID that owns face
        neighbor: Cell ID that is adjacent
        x: Location of face centroid
        distance: Distance from face centroid to cell center
    """
    ID: int
    boundary: bool
    owner: int
    neighbor: int
    x: list[float] = dataclasses.field(default_factory=list)
    distance: list[float] = dataclasses.field(default_factory=list)

class Faces:
    """
    Mesh Face
    """
    def __init__(self,points):
        """
        Set the face index and cell owner/neighbor
        """
        self.faces = []
        self.points = points

    def gen_uniform_faces(self,domain):
        """
        Generate faces based on unirom grid spacing
        """
        num_faces = len(self.points) + 1
        for n_f in range(num_faces):

            if n_f == 0:
                _face = Face(ID=n_f, boundary=True, owner=n_f, neighbor= -1)
                x = domain.domain_size[0][0] - 0.5*(self.points[n_f+ 1] - self.points[n_f])
                _face.x = x
                distance = x - self.points[n_f]
                _face.distance = distance

            elif n_f == num_faces - 1:
                _face = Face(ID=n_f, boundary=True, owner = n_f-1, neighbor= -1)
                x = domain.domain_size[0][1] + 0.5*(self.points[n_f - 1] - self.points[n_f - 2])
                _face.x = x
                distance = x - self.points[n_f-1]
                _face.distance = distance

            else:
                _face = Face( ID = n_f, boundary=False, owner = n_f - 1, neighbor = n_f + 1)
                x = self.points[n_f] - 0.5*(self.points[n_f] - self.points[n_f-1])
                _face.x = x
                distance = x - self.points[n_f-1]
                _face.distance = distance

            self.faces.append(_face)

        return self.faces,num_faces

class Mesh:
    """
    Mesh class   
    """
    def __init__(self):
        self.points = None
        self.num_points = -1
        self.cells = None
        self.num_cells = -1
        self.faces = None
        self.num_faces = -1

    def uniform_mesh_1d(self,domain,nodes):
        """
        Generate 1-Dimensional Uniform Grid
        """
        _points = Points(domain,nodes)
        self.points,self.num_points = _points.gen_uniform_points()

        _cells = Cells(self.points)
        self.cells,self.num_cells = _cells.gen_uniform_cells()

        _faces = Faces(self.points)
        self.faces,self.num_faces = _faces.gen_uniform_faces(domain)
