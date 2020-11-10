##### Legendre polynomials #####

import sympy as sp
import numpy as np

def Legendre_Polynomial(variable, n):
    y = (variable**2 - 1)**n
    L = (1/((2**n)*sp.factorial(n)))*y.diff(variable, n)
    return L
    
def phi(variable, n):
    if n == 1:
        phi = (1/sp.sqrt(2))*(variable + 1)
        return phi
    else:
        phi = (1/(sp.sqrt(4*n - 2)))*(Legendre_Polynomial(variable, n) - Legendre_Polynomial(variable, n-2))
        return phi

