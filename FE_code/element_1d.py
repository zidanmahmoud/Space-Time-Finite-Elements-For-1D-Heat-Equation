import numpy as np
from numpy.polynomial.legendre import leggauss
import math

from Auxilary import hierarchic_shape_functions as hsf

class Element1D:

    def __init__(self, id, nodes, degree, dofs):
        self.id = id
        self.nodes = nodes
        self.degree = degree
        self.dofs = dofs


    @property
    def coords(self):
        coords = np.array( [node.coordinates[0] for node in self.nodes] )
        return coords


    def k_matrix(self):
        """
        compute the k matrix of the element
        
        Returns
        -------
        k_e : ndarray
            stiffness matrix of the element
        """
        degree = self.degree
        num_dofs = degree + 1
        x1, x2 = self.coords

        k_e = np.zeros((num_dofs, num_dofs))

        num_gps = math.ceil(degree + 1)
        xg, wg = leggauss(num_gps)

        for i in range(len(xg)):
            xi = xg[i]
            weight = wg[i]
            
            dN = hsf.shape_functions_derivatives_1d(xi, degree)
            B = dN
            Le = x2 - x1
            k_e += np.outer(B, B) * 2 / Le * weight
        
        return k_e


    def m_matrix(self):
        """
        compute the m matrix of the element
        
        Returns
        -------
        m_e : ndarray
            stiffness matrix of the element
        """
        degree = self.degree
        num_dofs = degree + 1
        x1, x2 = self.coords

        m_e = np.zeros((num_dofs, num_dofs))

        num_gps = math.ceil(degree + 1)
        xg, wg = leggauss(num_gps)

        for i in range(len(xg)):
            xi = xg[i]
            weight = wg[i]

            N = hsf.shape_functions_1d(xi, degree)
            Le = x2 - x1
            m_e += np.outer(N, N) * Le / 2 * weight
        
        return m_e


    def f_vector(self, force_global_function, time_value):
        """
        compute the f vector of the element

        Parameters
        ----------
        force_global_function : callabe
            the function of force applied on the model
        
        Returns
        -------
        f_e : ndarray
            force vector of the element
        """
        degree = self.degree
        num_dofs = degree + 1
        x1, x2 = self.coords
        Le = x2 - x1

        f_e = np.zeros(num_dofs)

        num_gps = math.ceil(degree + 1)
        xg, wg = leggauss(num_gps)

        for i in range(len(xg)):
            xi = xg[i]
            weight = wg[i]

            N = hsf.shape_functions_1d(xi, degree)
            x = x1 + 0.5*(xi + 1)*(x2 - x1)
            f = force_global_function(x, time_value)

            f_e += f * N * Le / 2 * weight
        
        return f_e


    def integrate_uAnlytical_minus_uFEM_squared(self, u_analytical, local_u_FEM, time_value):
        degree = self.degree
        x1, x2 = self.coords
        Le = x2 - x1

        num_gps = math.ceil(degree + 1)
        xg, wg = leggauss(num_gps)

        integral = 0
        for i in range(len(xg)):
            xi = xg[i]
            weight = wg[i]

            N = hsf.shape_functions_1d(xi, degree)
            x = x1 + 0.5*(xi + 1)*(x2 - x1)
            u = u_analytical(x, time_value)
            u_FEM = np.dot(N, local_u_FEM)

            integral += (u - u_FEM)**2 * 2 / Le * weight
        
        return integral


    def integrate_uAnlytical_squared(self, u_analytical, time_value):
        degree = self.degree
        x1, x2 = self.coords
        Le = x2 - x1

        num_gps = math.ceil(degree + 1)
        xg, wg = leggauss(num_gps)

        integral = 0
        for i in range(len(xg)):
            xi = xg[i]
            weight = wg[i]

            x = x1 + 0.5*(xi + 1)*(x2 - x1)
            u = u_analytical(x, time_value)

            integral += (u)**2 * 2 / Le * weight
        
        return integral


    def get_solution_point_from_solution_vector(self, x, local_u_FEM):
        x1, x2 = self.coords
        xi = 2/(x2 - x1) * (x - x1) - 1
        N = hsf.shape_functions_1d(xi, self.degree)
        u = np.dot(N, local_u_FEM)
        return u
