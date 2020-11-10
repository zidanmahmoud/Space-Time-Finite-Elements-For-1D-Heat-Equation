#=============================================
# Implementation p-FEM: Main File
#=============================================

# Importing Required Modules
import Kernels.Implementation_p_FEM_Kernel as kernel
import sympy as sp

# Declaration of a symbol to define the load as a function
X = sp.symbols('X')


#=============================================
# User Input
#=============================================
num_elements = 1
polynomial_degree = 2 # from 1 to 8
num_GP = 8 # 0 .. Analytical, from 1 to 8 .. Numerical
E = 1
A = 1
L = 1
F = 0
f_x = (-sp.sin(8*X)) 
verbosity = 0


#=============================================
# Solution
#============================================= 
instance = kernel.Truss(num_elements, polynomial_degree,
                        num_GP, E, A, L, F, f_x, verbosity)
instance.solve()
