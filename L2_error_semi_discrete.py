import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation

from FE_code.model_constructor import ModelConstructor

pi = np.pi
sin = np.sin
cos = np.cos

def f(x, t):
    return 2*pi*cos(2*pi*t)*sin(2*pi*x) + 4*pi*pi*sin(2*pi*t)*sin(2*pi*x)

def u_an(x, t):
    return sin(2*pi*x)*sin(2*pi*t)


number_of_elements = np.linspace(3, 10, 8, dtype=int)
x_axis = 1 / number_of_elements
error1 = list()
error2 = list()
error3 = list()
error4 = list()
error5 = list()

for elements_x in number_of_elements:
    timesteps = elements_x*100
    dt = 1 / timesteps

    model = ModelConstructor(degree=1, print_welcome=False)
    model.construct_1d_semi_discrete_uniform_mesh(
        length = 1,
        elements = elements_x,
        rho = 1,
        c = 1,
        f = f,
        theta = 1,
        dt = dt,
        timesteps = timesteps
    )
    model.solve_1d_semi_discrete_uniform_mesh(print_info=False)
    error1.append(model.L2_norm_semi_discrete(u_an))

for elements_x in number_of_elements:
    timesteps = elements_x*100
    dt = 1 / timesteps

    model = ModelConstructor(degree=2, print_welcome=False)
    model.construct_1d_semi_discrete_uniform_mesh(
        length = 1,
        elements = elements_x,
        rho = 1,
        c = 1,
        f = f,
        theta = 1,
        dt = dt,
        timesteps = timesteps
    )
    model.solve_1d_semi_discrete_uniform_mesh(print_info=False)
    error2.append(model.L2_norm_semi_discrete(u_an))

for elements_x in number_of_elements:
    timesteps = elements_x*100
    dt = 1 / timesteps

    model = ModelConstructor(degree=3, print_welcome=False)
    model.construct_1d_semi_discrete_uniform_mesh(
        length = 1,
        elements = elements_x,
        rho = 1,
        c = 1,
        f = f,
        theta = 1,
        dt = dt,
        timesteps = timesteps
    )
    model.solve_1d_semi_discrete_uniform_mesh(print_info=False)
    error3.append(model.L2_norm_semi_discrete(u_an))

for elements_x in number_of_elements:
    timesteps = elements_x*100
    dt = 1 / timesteps

    model = ModelConstructor(degree=4, print_welcome=False)
    model.construct_1d_semi_discrete_uniform_mesh(
        length = 1,
        elements = elements_x,
        rho = 1,
        c = 1,
        f = f,
        theta = 1,
        dt = dt,
        timesteps = timesteps
    )
    model.solve_1d_semi_discrete_uniform_mesh(print_info=False)
    error4.append(model.L2_norm_semi_discrete(u_an))

for elements_x in number_of_elements:
    timesteps = elements_x*100
    dt = 1 / timesteps

    model = ModelConstructor(degree=5, print_welcome=False)
    model.construct_1d_semi_discrete_uniform_mesh(
        length = 1,
        elements = elements_x,
        rho = 1,
        c = 1,
        f = f,
        theta = 1,
        dt = dt,
        timesteps = timesteps
    )
    model.solve_1d_semi_discrete_uniform_mesh(print_info=False)
    error5.append(model.L2_norm_semi_discrete(u_an))

# timesteps_list = np.array([50, 100, 300, 500, 1000])
# x_axis = 1 / timesteps_list
# for timesteps in timesteps_list:
#     dt = 1 / timesteps

#     model = ModelConstructor(degree=1, print_welcome=False)
#     model.construct_1d_semi_discrete_uniform_mesh(
#         length = 1,
#         elements = 4,
#         rho = 1,
#         c = 1,
#         f = f,
#         theta = 1,
#         dt = dt,
#         timesteps = timesteps
#     )
#     model.solve_1d_semi_discrete_uniform_mesh(print_info=False)
#     error1.append(model.L2_norm_semi_discrete(u_an))

# for timesteps in timesteps_list:
#     dt = 1 / timesteps

#     model = ModelConstructor(degree=1, print_welcome=False)
#     model.construct_1d_semi_discrete_uniform_mesh(
#         length = 1,
#         elements = 10,
#         rho = 1,
#         c = 1,
#         f = f,
#         theta = 1,
#         dt = dt,
#         timesteps = timesteps
#     )
#     model.solve_1d_semi_discrete_uniform_mesh(print_info=False)
#     error2.append(model.L2_norm_semi_discrete(u_an))

# for timesteps in timesteps_list:
#     dt = 1 / timesteps

#     model = ModelConstructor(degree=1, print_welcome=False)
#     model.construct_1d_semi_discrete_uniform_mesh(
#         length = 1,
#         elements = 50,
#         rho = 1,
#         c = 1,
#         f = f,
#         theta = 1,
#         dt = dt,
#         timesteps = timesteps
#     )
#     model.solve_1d_semi_discrete_uniform_mesh(print_info=False)
#     error3.append(model.L2_norm_semi_discrete(u_an))

fig, ax = plt.subplots()
ax.plot(x_axis, error1, linestyle='-', marker='o', label='p=1')
ax.plot(x_axis, error2, linestyle='--', marker='o', label='p=2')
ax.plot(x_axis, error3, linestyle='-.', marker='o', label='p=3')
ax.plot(x_axis, error4, linestyle='-', marker='^', label='p=4')
ax.plot(x_axis, error5, linestyle='-', marker='*', label='p=5')

ax.legend(loc='best')
ax.set(
   title='L2 Error',
   xlabel='h',
   ylabel='L2 Error',
   xscale='log',
   yscale='log',
)
ax.grid(True, which='both')
plt.show()
