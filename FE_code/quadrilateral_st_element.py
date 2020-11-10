import numpy as np
from numpy.polynomial.legendre import leggauss
import math

from Auxilary import hierarchic_shape_functions as hsf


class QuadrilateralSpaceTimeElement:
    """
    class that constructs a quadrilateral space-time
    element for the finite element analysis.

    Parameters
    ----------
    id : int
        id of the element
    nodes : array of objects of type `Node`
        the nodes of the element
    c : flaot
        heat capacity
    k : float
        heat conductivity
    degree : int
        degree of the polynomial of the shape functions
    dofs : array-like
        the global degrees of freedom of the element

    Attributes
    ----------
    nodal_ids : ndarray
        IDs of the nodes of the element
    coords : ndarray
        coordinates of the nodes of the element
    """

    def __init__(self, id, nodes, c, k, degree, dofs):
        self.id = id
        if len(nodes) != 4: raise RuntimeError('nodes must be of size 4')
        self.nodes = nodes
        self._c = c
        self._k = k
        self.degree = degree
        self.dofs = dofs


    @property
    def nodal_ids(self):
        ids = np.array( [node.id for node in self.nodes] )
        return ids

    
    @property
    def coords(self):
        coords = np.array( [node.coordinates for node in self.nodes] )
        return coords

    
    def f_vector(self, force_global_function, shape_functions_info=None, det_j=None):
        """
        compute the f vector of the element

        Parameters
        ----------
        force_global_function : callabe
            the function of force applied on the model
        shape_functions_info : dict
            the gauss points local coordiates and the array of the shape functions
        
        Returns
        -------
        f_e : ndarray
            force vector of the element
        """
        degree = self.degree
        num_dofs = (degree + 1)**2
        f_e = np.zeros(num_dofs)

        coords = self.coords
        x1, t1 = coords[0]
        x3, t3 = coords[2]
        
        if shape_functions_info and det_j:
            for key, value in shape_functions_info.items():
                xi, eta, weight = key
                N = value

                x = x1 + 0.5*(xi + 1)*(x3 - x1)
                t = t1 + 0.5*(eta + 1)*(t3 - t1)

                f = force_global_function(x, t)

                f_e += f * N * det_j * weight
        
        else:
            num_gps = math.ceil(degree + 1)
            xg, wg = leggauss(num_gps)
            for j in range(len(xg)):
                gp_eta = xg[j]
                gp_wt_eta = wg[j]
                for i in range(len(xg)):
                    gp_xi = xg[i]
                    gp_wt_xi = wg[i]

                    N = hsf.shape_functions_2d(gp_xi, gp_eta, degree)
                    dNodal = hsf.shape_functions_derivatives_2d(gp_xi, gp_eta, 1)
                    j = np.dot(dNodal, coords)
                    det_j = np.linalg.det(j)

                    x = x1 + 0.5*(gp_xi + 1)*(x3 - x1)
                    t = t1 + 0.5*(gp_eta + 1)*(t3 - t1)

                    f = force_global_function(x, t)

                    f_e += f * N * det_j * gp_wt_xi * gp_wt_eta
                
        return f_e

    
    def k_matrix(self):
        """
        compute the k matrix of the element
        
        Returns
        -------
        k_e : ndarray
            stiffness matrix of the element
        """
        degree = self.degree
        num_dofs = (degree + 1)**2

        k_e = np.zeros((num_dofs, num_dofs))
        A = np.zeros((num_dofs, num_dofs))
        B = np.zeros((num_dofs, num_dofs))

        num_gps = math.ceil(degree + 1)
        xg, wg = leggauss(num_gps)

        for j in range(len(xg)):
            gp_eta = xg[j]
            gp_wt_eta = wg[j]
            for i in range(len(xg)):
                gp_xi = xg[i]
                gp_wt_xi = wg[i]
                
                N = hsf.shape_functions_2d(gp_xi, gp_eta, degree)
                dN = hsf.shape_functions_derivatives_2d(gp_xi, gp_eta, degree)
                dNodal = hsf.shape_functions_derivatives_2d(gp_xi, gp_eta, 1)
                nodal_coordinates = self.coords
                jacobian = np.dot(dNodal, nodal_coordinates)
                inv_jacobian = np.linalg.inv(jacobian)
                det_jacobian = np.linalg.det(jacobian)
                
                dN_global = np.dot(inv_jacobian, dN)
                dNdx = dN_global[0]
                dNdt = dN_global[1]

                A = np.outer(N, dNdt) * self._c * gp_wt_xi * gp_wt_eta * det_jacobian
                B = np.outer(dNdx, dNdx) * self._k * det_jacobian * gp_wt_xi * gp_wt_eta

                k_e += ( A + B )

        return k_e


    def get_solution_point_from_solution_vector(self, x, y, u):
        coords = self.coords
        degree = self.degree
        x1, t1 = coords[0]
        x3, t3 = coords[2]
        xi = 2/(x3 - x1) * (x - x1) - 1
        eta = 2/(t3 - t1) * (y - t1) - 1
        N = hsf.shape_functions_2d(xi, eta, degree)
        return np.dot(N,u)


    def get_solution_to_plot(self, global_u_vector):
        coords = self.coords
        degree = self.degree
        x1, t1 = coords[0]
        x3, t3 = coords[2]
        
        local_u = global_u_vector[self.dofs]

        x = list()
        t = list()
        u = list()
        for eta in np.linspace(-1, 1, degree + 1):
            for xi in np.linspace(-1, 1, degree + 1):
                N = hsf.shape_functions_2d(xi, eta, degree)
                u.append(np.dot(N, local_u))

                i_x = x1 + 0.5*(xi + 1)*(x3 - x1)
                i_t = t1 + 0.5*(eta + 1)*(t3 - t1)
                x.append(i_x)
                t.append(i_t)

        return np.array(x), np.array(t), np.array(u)


    def integrate_uAnlytical_minus_uFEM_squared(self, u_analytical, local_u_FEM):
        integral = 0

        coords = self.coords
        degree = self.degree
        x1, t1 = coords[0]
        x3, t3 = coords[2]

        num_gps = math.ceil(degree + 1)
        xg, wg = leggauss(num_gps)

        for j in range(len(xg)):
            gp_eta = xg[j]
            gp_wt_eta = wg[j]
            for i in range(len(xg)):
                gp_xi = xg[i]
                gp_wt_xi = wg[i]

                N = hsf.shape_functions_2d(gp_xi, gp_eta, degree)
                dNodal = hsf.shape_functions_derivatives_2d(gp_xi, gp_eta, 1)
                nodal_coordinates = self.coords
                jacobian = np.dot(dNodal, nodal_coordinates)
                det_jacobian = np.linalg.det(jacobian)

                x = x1 + 0.5*(gp_xi + 1)*(x3 - x1)
                t = t1 + 0.5*(gp_eta + 1)*(t3 - t1)

                u = u_analytical(x, t)
                u_FEM = np.dot(N, local_u_FEM)

                integral += (u - u_FEM)**2 * gp_wt_xi * gp_wt_eta * det_jacobian
        
        return integral


    def integrate_uAnlytical_squared(self, u_analytical):
        integral = 0

        coords = self.coords
        degree = self.degree
        x1, t1 = coords[0]
        x3, t3 = coords[2]

        num_gps = math.ceil(degree + 1)
        xg, wg = leggauss(num_gps)

        for j in range(len(xg)):
            gp_eta = xg[j]
            gp_wt_eta = wg[j]
            for i in range(len(xg)):
                gp_xi = xg[i]
                gp_wt_xi = wg[i]

                dNodal = hsf.shape_functions_derivatives_2d(gp_xi, gp_eta, 1)
                nodal_coordinates = self.coords
                jacobian = np.dot(dNodal, nodal_coordinates)
                det_jacobian = np.linalg.det(jacobian)

                x = x1 + 0.5*(gp_xi + 1)*(x3 - x1)
                t = t1 + 0.5*(gp_eta + 1)*(t3 - t1)

                u = u_analytical(x, t)

                integral += (u)**2 * gp_wt_xi * gp_wt_eta * det_jacobian
            
        return integral
