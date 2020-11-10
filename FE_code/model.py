import numpy as np
from scipy.sparse import coo_matrix
from numpy.polynomial.legendre import leggauss
import math
from Auxilary import hierarchic_shape_functions as hsf

from .node import Node
from .edge import Edge
from .quadrilateral_st_element import QuadrilateralSpaceTimeElement
from .element_1d import Element1D

class Model:
    """
    class model to construct and solve a space-time FEM problem

    Parameters
    ----------
    degree : int
        degree of the polynomials of the shape functions
    
    Attributes
    ----------
    nodes : array of objects of type `Node`
        all the nodes of the model
    elements : array of objects
        all the elements of the model
    """

    def __init__(self, degree):
        self._nodes = dict()
        self._edges = dict()
        self._elements = dict()
        self.total_num_dofs = None
        self.degree = degree


    @property
    def nodes(self):
        return self._nodes.values()


    @property
    def elements(self):
        return self._elements.values()


    def add_node(self, id, x, y):
        """
        Adds a node to the model

        Parameters
        ----------
        id : int
            id of the node
        x : float
            x position of the node
        y : float
            y position of the node
        """
        self._nodes[id] = Node(id, x, y)


    def add_edge(self, id, node_1, node_2):
        """
        Adds an edge to the model

        Parameters
        ----------
        id : int
            id of the edge
        node_1 : object of type `Node`
            the first node of the edge
        node_2 : object of type `Node`
            the second node of the edge
        """
        self._edges[id] = Edge(id, node_1, node_2)


    def add_quad_st_element(self, id, nodes, c, k, degree, dofs):
        """
        Adds a quad space-time element to the model

        Parameters
        ----------
        id : int
            id of the element
        nodes : array of objects of type `Node`
            the nodes of the element
        c : float
            heat capacity
        k : float
            heat conductivity
        degree : int
            degree of the polynomial of the shape functions
        dofs : array-like
            the global degrees of freedom of the element
        """
        self._elements[id] = QuadrilateralSpaceTimeElement(id, nodes, c, k, degree, dofs)


    def add_1d_element(self, id, nodes, degree, dofs):
        self._elements[id] = Element1D(id, nodes, degree, dofs)


    def get_node(self, id):
        """
        Get the node using its id

        Parameters
        ----------
        id : int
            the id of the node
        """
        return self._nodes[id]


    def get_edge(self, id):
        """
        Get the edge using its id

        Parameters
        ----------
        id : int
            the id of the node
        """
        return self._edges[id]


    def get_element(self, id):
        """
        Get the element using its id

        Parameters
        ----------
        id : int
            the id of the node
        """
        return self._elements[id]


    def find_element(self, x, y):
        """
        find the element using its coordinates

        Parameters
        ----------
        x : float
            x-coordinate of the point
        y : float
            y-coordinate of the point
        """
        for element in self.elements:
            coords = element.coords
            x1, y1 = coords[0]
            x3, y3 = coords[2]
            if x >= x1 and x <= x3 and y >= y1 and y <= y3:
                return element


    def find_element_1d(self, x):
        """
        find the element using its coordinates

        Parameters
        ----------
        x : float
            x-coordinate of the point
        """
        for element in self.elements:
            coords = element.coords
            x1, x2 = coords
            if x >= x1 and x <= x2:
                return element


    def assemble_stiffness_matrix(self):
        """
        assemble the stiffness matrix of the model

        Returns
        -------
        k : csr sparse matrix
            the assembled stiffness matrix of the model
        """
        if self.total_num_dofs is None:
            return

        num_dofs = self.total_num_dofs
        dofs_rows = list()
        dofs_columns = list()
        data = list()

        for element in self.elements:
            ke = element.k_matrix()

            data.extend(ke.reshape(-1,))
            element_dofs = element.dofs

            dofs_rows.extend(np.repeat(element_dofs, len(element_dofs)))
            dofs_columns.extend(np.tile(element_dofs, len(element_dofs)))

        k = coo_matrix((data, (dofs_rows, dofs_columns)), shape=(num_dofs, num_dofs))

        return k.tocsr()


    def assemble_mass_matrix(self):
        """
        assemble the mass matrix of the model

        Returns
        -------
        m : csr sparse matrix
            the assembled stiffness matrix of the model
        """
        if self.total_num_dofs is None:
            return

        num_dofs = self.total_num_dofs
        dofs_rows = list()
        dofs_columns = list()
        data = list()

        for element in self.elements:
            me = element.m_matrix()

            data.extend(me.reshape(-1,))
            element_dofs = element.dofs

            dofs_rows.extend(np.repeat(element_dofs, len(element_dofs)))
            dofs_columns.extend(np.tile(element_dofs, len(element_dofs)))

        m = coo_matrix((data, (dofs_rows, dofs_columns)), shape=(num_dofs, num_dofs))

        return m.tocsr()


    def assemble_stiffness_matrix_uniform_mesh(self):
        """
        assemble the stiffness matrix of the model.\n
        This is an optimized version that uses the mesh's uniformity

        Returns
        -------
        k : csr sparse matrix
            the assembled stiffness matrix of the model
        """
        if self.total_num_dofs is None:
            return

        num_dofs = self.total_num_dofs
        dofs_rows = list()
        dofs_columns = list()
        data = list()

        ke = self.get_element(1).k_matrix()

        for element in self.elements:

            data.extend(ke.reshape(-1,))
            element_dofs = element.dofs

            dofs_rows.extend(np.repeat(element_dofs, len(element_dofs)))
            dofs_columns.extend(np.tile(element_dofs, len(element_dofs)))

        k = coo_matrix((data, (dofs_rows, dofs_columns)), shape=(num_dofs, num_dofs))

        return k.tocsr()


    def assemble_force_vector(self, force_global_function, *args):
        """
        assemble the force vector of the model.

        Returns
        -------
        f : ndarray
            the assembled force vector of the model
        """
        if self.total_num_dofs is None:
            return

        num_dofs = self.total_num_dofs
        f = np.zeros(num_dofs)

        for element in self.elements:
            fe = element.f_vector(force_global_function, *args)

            element_dofs = element.dofs
            for i in range(len(element_dofs)):
                dof = element_dofs[i]
                f[dof] += fe[i]

        return f


    def assemble_force_vector_uniform_mesh(self, force_global_function):
        """
        assemble the force vector of the model.\n
        This is an optimized version that uses the mesh's uniformity

        Returns
        -------
        f : ndarray
            the assembled force vector of the model
        """
        if self.total_num_dofs is None:
            return

        num_dofs = self.total_num_dofs
        f = np.zeros(num_dofs)

        element_1_coords = self.get_element(1).coords
        x1, t1 = element_1_coords[0]
        x3, t3 = element_1_coords[2]
        area = (x3 - x1) * (t3 - t1)
        det_j = area / 4

        shape_functions_info = dict()
        num_gps = math.ceil(self.degree + 1)
        xg, wg = leggauss(num_gps)

        for j in range(len(xg)):
            gp_eta = xg[j]
            gp_wt_eta = wg[j]
            for i in range(len(xg)):
                gp_xi = xg[i]
                gp_wt_xi = wg[i]
                N = hsf.shape_functions_2d(gp_xi, gp_eta, self.degree)
                shape_functions_info[(gp_xi, gp_eta, gp_wt_xi*gp_wt_eta)] = N

        for element in self.elements:
            fe = element.f_vector(force_global_function, shape_functions_info, det_j)

            element_dofs = element.dofs
            for i in range(len(element_dofs)):
                dof = element_dofs[i]
                f[dof] += fe[i]

        return f
