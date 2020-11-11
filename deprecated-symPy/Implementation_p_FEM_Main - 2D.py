#=============================================
# Implementation p-FEM: Main File
#=============================================

# Importing Required Modules
import Kernels.Implementation_p_FEM_Kernel_2D as kernel
import sympy as sp

# Declaration of a symbol to define the load as a function
X = sp.symbols('X')


#=============================================
# User Input
#=============================================
height = 4
width = 4
num_elements_x = 15
num_elements_y = 20
polynomial_degree = 1 # from 1 to 8
num_GP = 2 # 0 .. Analytical, from 1 to 8 .. Numerical
k = 1
f = 1 
verbosity = 2

#=============================================
# Solution
#============================================= 
instance = kernel.Truss(height, width, num_elements_x, num_elements_y, polynomial_degree,
                        num_GP, k, f, verbosity)
instance.solve()
