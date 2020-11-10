#=============================================
# Implementation p-FEM: Main File
#=============================================

# Importing Required Modules
import Kernels.Implementation_p_FEM_Kernel as kernel
import sympy as sp
import numpy as np
from matplotlib import pyplot as plt

# Declaration of a symbol to define the load as a function
X = sp.symbols('X')


#=============================================
# User Input
#=============================================
total_elements = 8
final_polynomial_degree = 8
num_GP = 8
E = A = L = 1
F = 0
f_x = (-sp.sin(8*X))


#=============================================
# Solution
#============================================= 
verbosity = 0
h_version = np.zeros(total_elements)
p_version = np.zeros(final_polynomial_degree)
dof_h = np.linspace(2,total_elements+1,total_elements)
dof_p = np.linspace(2,final_polynomial_degree+1,final_polynomial_degree)

for polynomial_degree in range(1,final_polynomial_degree+1):
    num_elements = 1
    instance = kernel.Truss(num_elements, polynomial_degree,
                            num_GP, E, A, L, F, f_x, verbosity)
    instance.solve(show_disp=False)
    p_version[polynomial_degree-1] = instance.error * 100


for num_elements in range(1,total_elements+1):
    polynomial_degree = 1
    instance = kernel.Truss(num_elements, polynomial_degree,
                            num_GP, E, A, L, F, f_x, verbosity)
    instance.solve(show_disp=False)
    h_version[num_elements-1] = instance.error * 100


plt.figure()
plt.plot(dof_h, h_version, linestyle='--', marker='o', color='b', label='uniform h-version')
plt.plot(dof_p, p_version, linestyle='-.', marker='^', color='r', label='uniform p-version')
plt.xlabel('degrees of Freedom N')
plt.ylabel('relative error in energy norm [%]')
plt.legend(loc='best')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which='both')
plt.savefig('Error in the Strain Energy Norm', format='jpeg', dpi=300)
plt.show()