#=============================================
# Implementation p-FEM: Main File
#=============================================

# Importing Required Modules
import Kernels.Implementation_ST_FEM_Kernel_2D as kernel
import sympy as sp

# Declaration of a symbol to define the load as a function
X, t = sp.symbols('X t')


#=============================================
# User Input
#=============================================
pi = sp.pi
def sin(x): return sp.sin(x)
def cos(x): return sp.cos(x)

time_domain = 1
length_x = 1
num_elements_x = 50
num_elements_t = 50
polynomial_degree = 1 # from 1 to 8
num_GP = 2 # 0 .. Analytical, from 1 to 8 .. Numerical
k = 1
#f = sin(3*pi*X/2) + sin(3*pi*X/2)*(3*pi/2)**2*t
#f = sp.sin(sp.pi*X)*sp.sin(sp.pi*t)
#f = 2*pi*sin(2*pi*X)*cos(2*pi*t) + 4*pi*pi*sin(2*pi*X)*sin(2*pi*t)
f = sin(2*pi*X) + 4*pi*pi*sin(2*pi*X)*t
verbosity = 0

#=============================================
# Solution
#============================================= 
instance = kernel.Truss(time_domain, length_x, num_elements_x, num_elements_t, polynomial_degree,
                        num_GP, k, f, verbosity)
instance.solve()
instance.transient_plot()
