
from FE_code.model_constructor import ModelConstructor
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter

import numpy as np
pi = np.pi
def sin(x): return np.sin(x)
def cos(x): return np.cos(x)
def f(x, t):
    # return sin(2*pi*x) + 4*pi*pi*sin(2*pi*x)*t
    return 2*pi*cos(2*pi*t)*sin(2*pi*x) + 4*pi*pi*sin(2*pi*t)*sin(2*pi*x) # u=sin(2*pi*x)sin(2*pi*t)
    # return 1

def u_an(x, t):
    """Analytical Solution"""
    return sin(2*pi*x)*sin(2*pi*t)

def plot_a_curve(x, y, color, l, xl, yl, t, xs, ys):
    fig, ax = plt.subplots()
    #ax.ticklabel_format(axis='x', style='plain', useMathText=False)
    ax.plot(x, y, color, label= l)
    ax.set(
        xlabel=xl,
        ylabel=yl,
        title=t,
        xscale=xs,
        yscale=ys,
        xlim=[0,d]
    )
    #for axis in [ax.xaxis, ax.yaxis]:
    #    axis.set_major_formatter(ScalarFormatter(useOffset=False))
    #    axis.set_major_formatter(FormatStrFormatter('%.7f'))
    
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%i'))

    ax.grid(True)
    ax.legend()
    plt.show()

'''
# Plotting error for each element
n = 7
dofs = list()
for i in range (1, n + 1):
    elements_x = 2**i 
    elements_t = 2**i
    model = ModelConstructor(k=1, f=f)
    model.construct_2d_uniform_mesh(length_x=1, length_t=1, elements_x = elements_x, elements_t = elements_t, degree = 1)
    model.solve_2d_uniform_mesh(print_info=True)
    #d_freedom = (elements_x + 1)*(elements_t + 1)
    d_freedom = 1/elements_x
    L2 = model.L2_norm(u_an)

    dofs.append(d_freedom)
    error_p1.append(L2)
'''
# Error vs polynomial degree
d = 8
polynomial_degree = np.linspace(1, d, d)
error_p = list()
for i in range (1, d + 1):
    elements_x = 3
    elements_t = 3
    model = ModelConstructor(degree=i, print_welcome=False)
    model.construct_2d_uniform_mesh(
        length_x = 1,
        length_t = 1,
        c = 1,
        kappa = 1,
        elements_x = elements_x,
        elements_t = elements_t,
        f = f,
    )
    model.solve_2d_uniform_mesh(print_info=False)
    L2 = model.L2_norm(u_an)*100 # error percentage
    
    error_p.append(L2)

plot_a_curve(polynomial_degree, 
             error_p,
             '-bo', 
             'uniform p version', 
             'polynomial degree p', 
             'Relative error in L2 norm [%]', 
             'Error in Space-Time FEM', 
             'log', 
             'log')
