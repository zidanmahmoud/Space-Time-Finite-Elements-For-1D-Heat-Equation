#=============================================
# Implementation p-FEM: Main File
#=============================================

# Importing Required Modules
import Kernels.Implementation_p_FEM_Kernel_Transient as kernel
import sympy as sp

# Declaration of a symbol to define the load as a function
X = sp.symbols('X')
t = sp.symbols('t')


#=============================================
# User Input
#=============================================
num_elements = 1
polynomial_degree = 8 # from 1 to 8
num_GP = 8 # 0 .. Analytical, from 1 to 9 .. Numerical
E = 1
A = 1
L = 1
F = 0
f_x_t = sp.sin(3*sp.pi*X/2) + sp.sin(3*sp.pi*X/2) * (3*sp.pi/2)**2 * t 
verbosity = 0
theta = 1 # 0 := Explicit ... 1 := Implicit ... 0.5 := Crank-Nicolson
rho = 1
c = 1
T = 0.1
timesteps = 5
dt = T/timesteps


#=============================================
# Solution
#============================================= 
instance = kernel.Truss(num_elements, polynomial_degree,
                        num_GP, E, A, L, F, f_x_t, verbosity,
                        theta, rho, c, dt, timesteps)
instance.solve(show_disp=True)