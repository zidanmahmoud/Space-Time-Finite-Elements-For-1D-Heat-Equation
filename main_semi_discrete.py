import numpy as np
pi = np.pi
sin = np.sin
cos = np.cos

from FE_code.model_constructor import ModelConstructor

def f(x, t):
    return 2*pi*cos(2*pi*t)*sin(2*pi*x) + 4*pi*pi*sin(2*pi*t)*sin(2*pi*x)

def u_an(x, t):
    return sin(2*pi*x)*sin(2*pi*t)

timesteps=100
time=1
dt = time / timesteps

model = ModelConstructor(degree=1)
model.construct_1d_semi_discrete_uniform_mesh(
    length = 1,
    elements = 20,
    c = 1,
    kappa = 1,
    f = f,
    theta = 1,
    dt = dt,
    timesteps = timesteps,
)
model.solve_1d_semi_discrete_uniform_mesh(print_info=True)
model.L2_norm_semi_discrete(u_an)
model.transient_plot_semi_discrete()
