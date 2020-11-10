#=============================================
# Implementation p-FEM: Main File
#=============================================

# Importing Required Modules
import Kernels.Implementation_p_FEM_Kernel_Transient as kernel
import sympy as sp
import numpy as np
from matplotlib import pyplot as plt
import math

# Declaration of a symbol to define the load as a function
X = sp.symbols('X')
t = sp.symbols('t')


#=============================================
# User Input
#=============================================
num_GP = 9 # 0 .. Analytical, from 1 to 9 .. Numerical
E = 1
A = 1
L = 1
F = 0
f_x_t = sp.sin(3*sp.pi*X/2) + sp.sin(3*sp.pi*X/2) * (3*sp.pi/2)**2 * t 
verbosity = 0

theta = 0.5 # 0 := Explicit ... 1 := Implicit ... 0.5 := Crank-Nicolson
rho = 1
c = 1
dt = 0.01
timesteps = 100

total_elements = 8
final_polynomial_degree = 8

#=============================================
# Solution
#============================================= 
h_version = np.zeros(total_elements)
p_version = np.zeros(final_polynomial_degree)
dof_h = np.linspace(2,total_elements+1,total_elements)
dof_p = np.linspace(2,final_polynomial_degree+1,final_polynomial_degree)

#for polynomial_degree in range(1,final_polynomial_degree+1):
#    num_elements = 1
#    instance = kernel.Truss(num_elements, polynomial_degree,
#                        num_GP, E, A, L, F, f_x_t, verbosity,
#                        theta, rho, c, dt, timesteps)
#    instance.solve(show_disp=False)
#    print('finished iteration ', polynomial_degree+1)
#    p_version[polynomial_degree-1] = instance.error_L2 * 100
#print('########################################\nFinished p-version')
#
#for num_elements in range(1,total_elements+1):
#    polynomial_degree = 1
#    instance = kernel.Truss(num_elements, polynomial_degree,
#                        num_GP, E, A, L, F, f_x_t, verbosity,
#                        theta, rho, c, dt, timesteps)
#    instance.solve(show_disp=False)
#    print('finished iteration ', num_elements+1)
#    h_version[num_elements-1] = instance.error_L2 * 100
#
#plt.figure()
#plt.plot(dof_h, h_version, linestyle='--', marker='o', color='b', label='uniform h-version')
#plt.plot(dof_p, p_version, linestyle='-.', marker='^', color='r', label='uniform p-version')
#if theta == 0:
#    plt.title('Time Discretization Scheme: Explicit, $dt=0.00001$')
#elif theta == 0.5:
#    plt.title('Time Discretization Scheme: Crank Nicolson, $dt=0.00001$')
#elif theta == 1:
#    plt.title('Time Discretization Scheme: Implicit, $dt=0.00001$')
#else:
#    plt.title('$\\theta = $', theta, '$dt=0.00001$')
#plt.xlabel('degrees of Freedom N')
#plt.ylabel('L2 Error [%]')
#plt.legend(loc='best')
#plt.xscale('log')
#plt.yscale('log')
#plt.grid(True, which='both')
#if theta == 0:
#    plt.savefig('L2 Error Explicit', format='jpeg', dpi=300)
#elif theta == 0.5:
#    plt.savefig('L2 Error Crank', format='jpeg', dpi=300)
#elif theta == 1:
#    plt.savefig('L2 Error Implicit', format='jpeg', dpi=300)
#else:
#    plt.savefig('L2 Error theta '+str(theta), format='jpeg', dpi=300)
##plt.show()
#
#
#
#
#
#
dt_range = [0.1]
for i in range (4):
    dt_range.append(dt_range[i]/2.0)
dt_version = np.zeros(len(dt_range))
for i in range(np.size(dt_range)):
    
    num_elements = 1
    polynomial_degree = 8
    dt = dt_range[i]
    timesteps = int(1.0/dt)
    print('iteration ', i,', ', 'dt = ', dt)
    instance = kernel.Truss(num_elements, polynomial_degree,
                        num_GP, E, A, L, F, f_x_t, verbosity,
                        theta, rho, c, dt, timesteps)
    instance.solve(show_disp=False)
    dt_version[i] = instance.error_L2 * 100
plt.figure()
plt.plot(dt_range, dt_version, linestyle='-', marker='o', color='b', label='L2 error with varying dt\nusing Implicit Scheme')
plt.plot(dt_range,dt_range, linestyle='-.', color='g', label='$y = x$')
dt2 = np.zeros(len(dt_range))
for i in range(len(dt_range)):
    dt2[i] = dt_range[i] * dt_range[i]
plt.plot(dt_range, dt2, linestyle='--', color='r', label='$y = x^2$')
plt.legend(loc='best')
plt.title('L2 Error for 1 element of polynomial degree 8')
plt.xlabel('dt')
plt.ylabel('L2 Error [%]')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which='both')
plt.savefig('L2 Error', format='jpeg', dpi=300)
plt.show()