import numpy as np
pi = np.pi
sin = np.sin
cos = np.cos

from FE_code.model_constructor import ModelConstructor

def f(x, t):
    return 2*pi*cos(2*pi*t)*sin(2*pi*x) + 4*pi*pi*sin(2*pi*t)*sin(2*pi*x)

def u_analytical(x, t):
    return sin(2*pi*x)*sin(2*pi*t)


model = ModelConstructor(degree=1)
model.construct_2d_uniform_mesh(
    length_x = 1,
    length_t = 1,
    c = 1,
    kappa = 1,
    elements_x = 100,
    elements_t = 100,
    f = f,
)
model.solve_2d_uniform_mesh(print_info=False)
model.plot_solution_3d()
