import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

class determine_shape_functions:
    def __init__(self, polynomial_degree):
        self.polynomial_degree = polynomial_degree

    def Legendre(self, domain, degree):
        leg = legendre(degree)
        P_domain = leg(domain)
        return P_domain
 
    def Legendre_derivative(self, domain, degree, diff_degree):
        leg = legendre(degree)
        leg_derivative = leg.deriv(diff_degree)
        Pprime = leg_derivative(domain)
        return Pprime 

    def phi(self, domain, degree):
        phi = (1/np.sqrt(4*degree - 2))*(self.Legendre(domain, degree) - self.Legendre(domain, degree - 2))
        return phi

    def phi_derivative(self, domain, degree, diff_degree):
        phiprime = (1/np.sqrt(4*degree - 2))*(self.Legendre_derivative(domain, degree, diff_degree) - self.Legendre_derivative(domain, degree - 2, diff_degree))
        return phiprime 

    def nodal_shape_functions(self, xi, eta):
        N1 = 1/4 * (1 - xi)*(1 - eta)
        N2 = 1/4 * (1 + xi)*(1 - eta)
        N3 = 1/4 * (1 + xi)*(1 + eta)
        N4 = 1/4 * (1 - xi)*(1 + eta)
        
        return np.array([[N1, N2, N3, N4]])
    
    def nodal_shape_functions_derivatives(self, xi, eta):
        dN1dxi = -1/4 * (1 - eta)
        dN2dxi =  1/4 * (1 - eta)
        dN3dxi =  1/4 * (1 + eta)
        dN4dxi = -1/4 * (1 + eta)
        
        dN1deta = -1/4 * (1 - xi)
        dN2deta = -1/4 * (1 + xi)
        dN3deta =  1/4 * (1 + xi)
        dN4deta =  1/4 * (1 - xi)

        return np.matrix([[dN1dxi,  dN2dxi,  dN3dxi,  dN4dxi ],
                          [dN1deta, dN2deta, dN3deta, dN4deta]])

    def edge_shape_functions(self, xi, eta):
        if self.polynomial_degree < 2: raise RuntimeError('Edge shape functions just possible if polynomial_degree >= 2')
        edge_shape_functions = np.zeros((4, self.polynomial_degree - 1))
        for i in range(self.polynomial_degree - 1):
            edge_shape_functions[0, i] = (1/2)*(1 - eta)*self.phi(xi, i + 2)
            edge_shape_functions[1, i] = (1/2)*(1 + xi)*self.phi(eta, i + 2)
            edge_shape_functions[2, i] = (1/2)*(1 + eta)*self.phi(xi, i + 2)
            edge_shape_functions[3, i] = (1/2)*(1 - xi)*self.phi(eta, i + 2)
 
        return edge_shape_functions

    def edge_shape_functions_derivatives(self, xi, eta):
        if self.polynomial_degree < 2: raise RuntimeError('Edge shape functions just possible if polynomial_degree >= 2')
        dNEdge = np.zeros((2, 4*(self.polynomial_degree - 1))) 
        for i in range(self.polynomial_degree - 1):
            #Edge 1, in row 0 : derivatives w.r.t xi, in row 1 : derivatives w.r.t eta
            dNEdge[0, i] = (1/2) * (1 - eta) * self.phi_derivative(xi, i + 2, 1)
            dNEdge[1, i] = -(1/2) * self.phi(xi, i + 2) 
            #Edge 2
            dNEdge[0, i + self.polynomial_degree - 1] = (1/2) * self.phi(eta, i + 2)
            dNEdge[1, i + self.polynomial_degree - 1] = (1/2) * (1 + xi) * self.phi_derivative(eta, i + 2, 1) 
            #Edge 3
            dNEdge[0, i + 2*(self.polynomial_degree - 1)] = (1/2) * (1 + eta) * self.phi_derivative(xi, i + 2, 1) 
            dNEdge[1, i + 2*(self.polynomial_degree - 1)] = (1/2) * self.phi(xi, i + 2)
            #Edge 4
            dNEdge[0, i + 3*(self.polynomial_degree - 1)] = -(1/2) * self.phi(eta, i + 2)
            dNEdge[1, i + 3*(self.polynomial_degree - 1)] = (1/2) * (1 - xi) * self.phi_derivative(eta, i + 2, 1) 

        return dNEdge

    def internal_shape_functions(self, xi, eta):
        if self.polynomial_degree < 2: raise RuntimeError('Edge shape functions just possible if polynomial_degree >= 4')
        n = (self.polynomial_degree - 1)*(self.polynomial_degree - 1)
        internal_shape_functions = np.zeros((1, n))
        k = 0
        for i in range (2, self.polynomial_degree + 1):
            for j in range (2, self.polynomial_degree + 1):
                internal_shape_functions[0, k] = self.phi(xi, i) * self.phi(eta, j)
                k = k + 1
        return internal_shape_functions
    
    def internal_shape_functions_derivatives(self, xi, eta):
        if self.polynomial_degree < 2: raise RuntimeError('Edge shape functions just possible if polynomial_degree >= 4')
        n = (self.polynomial_degree - 1)*(self.polynomial_degree - 1)
        dNInternal = np.zeros((2, n))
        k = 0
        for i in range (2, self.polynomial_degree + 1):
            for j in range (2, self.polynomial_degree + 1):
                dNInternal[0, k] = self.phi_derivative(xi, i, 1) * self.phi(eta, j) 
                dNInternal[1, k] = self.phi(xi, i) * self.phi_derivative(eta, j, 1)
                k = k + 1
        
        return dNInternal

    def shape_functions(self, xi, eta):
        if self.polynomial_degree == 1:
            nodal = self.nodal_shape_functions(xi, eta)
            return np.concatenate((nodal), axis = None)
        else:
            nodal = self.nodal_shape_functions(xi, eta)
            edge = self.edge_shape_functions(xi, eta)
            internal = self.internal_shape_functions(xi, eta)
            return np.concatenate((nodal, edge, internal), axis = None)

    def shape_functions_derivatives(self, xi, eta):
        if self.polynomial_degree == 1:
            return self.nodal_shape_functions_derivatives(xi, eta)       
        else:
            dN = np.concatenate((self.nodal_shape_functions_derivatives(xi, eta), self.edge_shape_functions_derivatives(xi, eta), self.internal_shape_functions_derivatives(xi, eta)), axis = 1)
            return dN
           

'''
# FOR TEST
degree = 2
xi = 0.2
eta = 0.28
shape_fun = determine_shape_functions(degree)
N_nodal = shape_fun.nodal_shape_functions(xi, eta)
N_edge = shape_fun.edge_shape_functions(xi, eta)
N_internal = shape_fun.internal_shape_functions(xi, eta)
N = shape_fun.shape_functions(xi, eta)

dN_nodal = shape_fun.nodal_shape_functions_derivatives(xi, eta)
dN_edge = shape_fun.edge_shape_functions_derivatives(xi, eta)
dN_internal = shape_fun.internal_shape_functions_derivatives(xi, eta)
dN = shape_fun.shape_functions_derivatives(xi, eta)
print("nodal shape fucntions : \n", N_nodal)
print("edge shape fucntions : \n", N_edge)
print("internal shape fucntions : \n", N_internal)
print("shape functions : \n", N)
print("nodal shape functions derivatives : \n", dN_nodal)
print("edge shape functions derivatives : \n", dN_edge)
print("internal shape functions derivatives : \n", dN_internal)
print("shape functions derivatives: ", dN)
print("")
'''
