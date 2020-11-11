import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation
from scipy.sparse.linalg import spsolve
import math
from numpy.polynomial.legendre import leggauss

from .model import Model
from Auxilary import welcome
from Auxilary import hierarchic_shape_functions as hsf

np.set_printoptions(precision=5, suppress=True)

class ModelConstructor:
    """
    Constructs a model to be solved using space-time finite
    elements.

    Parameters
    ----------
    degree : int
        polynomial degree of the shape functions
    """

    def __init__(self, degree=1, print_welcome=True):
        if print_welcome:
            welcome.print_welcome_message()
        self.degree = degree
        if int(degree) != degree:
            raise RuntimeError('degree: {} is not an integer'.format(degree))
        
        # Construct an empty model
        self._FEmodel = Model(degree)


    def construct_2d_uniform_mesh(self, **kwrgs):
        """
        Construct a 2D uniform quadrilateral mesh for space-time
        application. Uniform refers to the fact that the mesh is
        a rectangular mesh that has a constant step size (but may
        vary in the second dimension) and a constant polynomial
        degree of the shape function.
        Arguments
        ---------
        length_x : float
            length in x direction
        length_t : float
            length in t direction
        elements_x : int
            number of elements in x direction
        elements_t : int
            number of elements in t direction
        c : float
            heat capacity
        kappa : float
            heat conductivity
        f : callable
            function of the heat flux in space and time
        """

        self.length_x = kwrgs.get('length_x')
        self.length_t = kwrgs.get('length_t')
        self.elements_x = kwrgs.get('elements_x')
        self.elements_t = kwrgs.get('elements_t')
        self._c = kwrgs.get('c')
        self._k = kwrgs.get('kappa')
        self._force_function = kwrgs.get('f')

        # Error checks
        if any(entry is None for entry in (self.length_x, self.length_t, self.elements_x, self.elements_x)):
            raise RuntimeError('Error entering the arguments for making a structured mesh')
        if type(self.elements_t) != int or type(self.elements_x) != int or type(self.degree) != int:
            raise RuntimeError('Number of Elements must be an integer')

        # Get the coordinates of all nodes of the model
        num_faces = self.elements_x * self.elements_t
        num_nodes = (self.elements_x + 1)*(self.elements_t + 1)
        num_edges = (num_faces + 1) + num_nodes - 2 #using Euler's formula including the outer infinitely large face
        dx = self.length_x / self.elements_x
        dt = self.length_t / self.elements_t
        x = list()
        t = list()
        for i in range(self.elements_t + 1):
            for j in range(self.elements_x + 1):
                t.append(i*dt)
                x.append(j*dx)

        # Add all nodes
        for i in range(num_nodes):
            self._FEmodel.add_node(id=i+1, x=x[i], y=t[i])

        # Add all edges
        id = num_nodes
        k = 0
        node_1 = 0
        node_2 = 1
        for j in range(self.elements_t + 1):
            for i in range(1, self.elements_x + 1):
                id = id + 1
                node_1 = node_1 + 1 
                node_2 = node_2 + 1
                self._FEmodel.add_edge(id, node_1, node_2)
                k = k + 1
            if k < num_edges:
                node_1 = node_2
                for i in range(1, self.elements_x + 2):
                    id = id + 1 
                    node_1 = node_1 + 1
                    node_2 = node_1 - self.elements_x - 1 
                    self._FEmodel.add_edge(id, node_1, node_2)
                    k = k + 1
                node_1 = node_1 - self.elements_x - 1
                node_2 = node_1 + 1
    
        # Add all elements
        num_dofs_per_element = (self.degree + 1)**2
        num_nodal_dofs = 4
        num_edge_dofs = self.degree - 1
        num_internal_dofs = num_dofs_per_element - num_nodal_dofs - num_edge_dofs*4
        edge_dof_increment = 2*self.elements_x + 1

        counter_internal_dofs = num_nodes + num_edge_dofs*num_edges
        self._FEmodel.total_num_dofs = counter_internal_dofs + num_faces*num_internal_dofs

        print('\nnum of elemenets  : ', num_faces)
        print('num of nodes      : ', num_nodes)
        print('polynomial degree : ', self.degree)
        print('num of dofs       : ', self._FEmodel.total_num_dofs)

        counter=0
        for i in range(self.elements_t):
            for j in range(self.elements_x):
                counter += 1
                node1_id = (self.elements_x+1)*i + (j+1)
                node2_id = node1_id + 1
                node3_id = (self.elements_x+1)*(i+1) + j + 2
                node4_id = node3_id - 1

                # Determine the dofs
                element_dofs = np.zeros(num_dofs_per_element, dtype=int)
                nodal_dofs = [
                    i*(self.elements_x+1) + j,
                    i*(self.elements_x+1) + j + 1,
                    (i+1)*(self.elements_x+1) + j + 1,
                    (i+1)*(self.elements_x+1) + j
                ]
                element_dofs[0 : num_nodal_dofs] = nodal_dofs

                if self.degree > 1:
                    edge_dofs1 = np.zeros(num_edge_dofs, dtype=int)
                    edge_dofs1[-1] = num_nodes + num_edge_dofs*(i*edge_dof_increment + j + 1) - 1
                    for k in range(num_edge_dofs-1, 0, -1):
                        edge_dofs1[k-1] = edge_dofs1[k] - 1
                    element_dofs[num_nodal_dofs : num_nodal_dofs+num_edge_dofs] = edge_dofs1

                    edge_dofs2 = np.zeros(num_edge_dofs, dtype=int)
                    edge_dofs2[-1] = num_nodes + num_edge_dofs*(i*edge_dof_increment + j + 2 + self.elements_x) - 1
                    for k in range(num_edge_dofs-1, 0, -1):
                        edge_dofs2[k-1] = edge_dofs2[k] - 1
                    element_dofs[num_nodal_dofs+num_edge_dofs : num_nodal_dofs+2*num_edge_dofs] = edge_dofs2

                    edge_dofs3 = np.zeros(num_edge_dofs, dtype=int)
                    edge_dofs3[-1] = num_nodes + num_edge_dofs*((i+1)*edge_dof_increment + j + 1) - 1
                    for k in range(num_edge_dofs-1, 0, -1):
                        edge_dofs3[k-1] = edge_dofs3[k] - 1
                    element_dofs[num_nodal_dofs+2*num_edge_dofs : num_nodal_dofs+3*num_edge_dofs] = edge_dofs3

                    edge_dofs4 = np.zeros(num_edge_dofs, dtype=int)
                    edge_dofs4[-1] = num_nodes + num_edge_dofs*(i*edge_dof_increment + j + 1 + self.elements_x) - 1
                    for k in range(num_edge_dofs-1, 0, -1):
                        edge_dofs4[k-1] = edge_dofs4[k] - 1
                    element_dofs[num_nodal_dofs+3*num_edge_dofs : num_nodal_dofs+4*num_edge_dofs] = edge_dofs4

                    internal_dofs = np.zeros(num_internal_dofs, dtype=int)
                    for k in range(num_internal_dofs):
                        internal_dofs[k] = counter_internal_dofs
                        counter_internal_dofs += 1
                    element_dofs[num_nodal_dofs+4*num_edge_dofs : num_nodal_dofs+4*num_edge_dofs+num_internal_dofs] = internal_dofs

                self._FEmodel.add_quad_st_element(
                    id = counter,
                    nodes = [
                        self._FEmodel.get_node(node1_id),
                        self._FEmodel.get_node(node2_id),
                        self._FEmodel.get_node(node3_id),
                        self._FEmodel.get_node(node4_id)
                    ],
                    c = self._c,
                    k = self._k,
                    degree = self.degree,
                    dofs = element_dofs
                )

        print('(1/4) Constructed the mesh')


    def solve_2d_uniform_mesh(self, print_info=False, penalty=10e10):
        """
        Solves the model using FEM by assembling the k matrix
        and the f vector and solving for the u vector. This
        function is optimized for the use of the 2D uniform
        mesh.
        
        Parameters
        ----------
        print_info : bool
            prints the solution information if True
        penalty : float
            penalty value to apply boundary conditions weakly
        """

        self.K = self._FEmodel.assemble_stiffness_matrix_uniform_mesh()
        self.F = self._FEmodel.assemble_force_vector_uniform_mesh(self._force_function)

        print('(2/4) System matrices are constructed')
        if print_info:
            print('System Matrix:\n', self.K.toarray(), '\n')
            print('Force Vector:\n', self.F, '\n')

        """
        EDGE DEFINITION:
        ****************
           time-axis     edge3
              ----------------------------
              |                          |
              |                          |
              |                          |
         edge4|                          |edge2
              |                          |
              |                          |
              |                          |
              ---------------------------- space-axis
                         edge1
        """

        degree = self.degree
        elements_x = self.elements_x
        elements_t = self.elements_t
        num_edge_dofs = degree - 1
        
        edge_1_element_ids = np.linspace(1, elements_x,  elements_x, dtype=int)
        edge_1_outer_dofs = np.array([], dtype=int)
        for id in edge_1_element_ids:
            element = self._FEmodel.get_element(id)
            outer_dofs = np.append(element.dofs[0:2], element.dofs[4:4+num_edge_dofs])
            edge_1_outer_dofs = np.append(edge_1_outer_dofs, outer_dofs)
        edge_1_outer_dofs = np.unique(edge_1_outer_dofs)

        edge_2_element_ids = np.linspace(elements_x, elements_x*elements_t, elements_t, dtype=int)
        edge_2_outer_dofs = np.array([], dtype=int)
        for id in edge_2_element_ids:
            element = self._FEmodel.get_element(id)
            outer_dofs = np.append(element.dofs[1:3], element.dofs[4+num_edge_dofs:4+2*num_edge_dofs])
            edge_2_outer_dofs = np.append(edge_2_outer_dofs, outer_dofs)
        edge_2_outer_dofs = np.unique(edge_2_outer_dofs)

        edge_4_element_ids = np.linspace(1, elements_x*(elements_t-1)+1, elements_t, dtype=int)
        edge_4_outer_dofs = np.array([], dtype=int)
        for id in edge_4_element_ids:
            element = self._FEmodel.get_element(id)
            outer_dofs = np.append([element.dofs[0], element.dofs[3]], element.dofs[4+3*num_edge_dofs:4+4*num_edge_dofs])
            edge_4_outer_dofs = np.append(edge_4_outer_dofs, outer_dofs)
        edge_4_outer_dofs = np.unique(edge_4_outer_dofs)

        dbc = np.unique(np.append(np.append(edge_1_outer_dofs, edge_2_outer_dofs), edge_4_outer_dofs))
        # nbc = edge_2_outer_dofs

        self.K[dbc,dbc] += penalty
        self.F[dbc] += 0*penalty

        (print('(3/4) Applied boundary conditions'))

        # self.F[nbc] = 0

        self.U = spsolve(self.K, self.F)

        print('(4/4) Solved')

        if print_info:
            print('\nSolution Vector:\n', self.U)


    def plot_solution_3d(self):
        """
        Plots the solution surface being the x-axis the spatial
        dimension, the y-axis the temporal dimension, and the 
        z-axis the solution.
        """
        accuracy = self.elements_x + 1 + self.elements_x*(self.degree - 1)
        u = np.zeros((accuracy, accuracy))
        x = np.linspace(0, self.length_x, accuracy)
        y = np.linspace(0, self.length_t, accuracy)
        X, Y = np.meshgrid(x, y)
        for i in range(accuracy):
            current_y = y[i]
            for j in range(accuracy):
                current_x = x[j]
                element = self._FEmodel.find_element(current_x, current_y)
                u[i, j] = element.get_solution_point_from_solution_vector(current_x, current_y, self.U[element.dofs])
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap='coolwarm')
        ax.set(
            xlabel = '< X >',
            ylabel = '< Time >',
            zlabel = '< Solution >'
        )
        plt.show()


    def transient_plot(self, time_interval=2):
        """
        Plots an animated solution with time

        Parameters
        ----------
        time_interval : float
            time (in seconds) required to perform one animation loop
        """
        fig, ax = plt.subplots()
        ax.grid()
        ax.set(
            title='Space-Time Solution',
            xlim=[0, self.length_x],
            ylim=[np.min(self.U), np.max(self.U)],
            xlabel = 'x (spatial axis)',
            ylabel = 'Solution',
        )

        line, = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            return line,

        x_plot = np.linspace(0, self.length_x, self.elements_x+1)
        u_plot = list()
        for time_step in range(self.elements_t + 1):
            u_plot_list = list()
            index = time_step * (self.elements_x + 1)
            while (index == time_step*(self.elements_x+1)) or (index % (self.elements_x+1) != 0):
                u_plot_list.append(self.U[index])
                index += 1
            u_plot.extend([u_plot_list])

        def animate(frame):
            line.set_data(x_plot, u_plot[frame])
            plt.savefig('plot_'+str(frame)+'.png')
            return line,

        anim = animation.FuncAnimation(fig, animate, frames=self.elements_t+1,
            interval=time_interval*1000/(self.elements_t+1), init_func=init, blit=True)
        # anim.save('stfem.gif', fps=30)
        plt.show()


    def L2_norm(self, u_analytical):
        """
        Calculates the L2 norm of the error
                  integral((u_analytical(x, t) - u_FEM(x, t))^2 * d_A)**2
        error = (------------------------------------------------------) ** 0.5
                        integral((u_analytical(x, t))^2 * d_A)**2
        """
        nominator = 0
        denominator = 0
        for element in self._FEmodel.elements:
            nominator += element.integrate_uAnlytical_minus_uFEM_squared(u_analytical, self.U[element.dofs])
            denominator += element.integrate_uAnlytical_squared(u_analytical)
        L2 = np.sqrt( nominator / denominator )
        return L2


    def construct_1d_semi_discrete_uniform_mesh(self, **kwargs):
        self._length = kwargs.get('length')
        self._elements = kwargs.get('elements')
        self._c = kwargs.get('c')
        self._k = kwargs.get('kappa')
        self._force_function = kwargs.get('f')
        self._theta = kwargs.get('theta')
        self._dt = kwargs.get('dt')
        self._timesteps = kwargs.get('timesteps')

        num_nodes = self._elements + 1
        self._FEmodel.total_num_dofs = num_nodes + self._elements * (self.degree - 1)
        Le = self._length / self._elements

        for i in range(num_nodes):
            self._FEmodel.add_node(id=i+1, x=i*Le, y=0)

        internal_dofs_counter = 0
        for i in range(self._elements):
            dofs = np.zeros(self.degree+1, dtype=int)
            dofs[0:2] = [i, i+1]
            internal_dofs = list()
            for j in range(self.degree+1 - 2):
                internal_dofs.append(num_nodes + j + internal_dofs_counter)
            dofs[2:] = internal_dofs

            node_1_id = i+1
            node_2_id = i+2
            self._FEmodel.add_1d_element(
                id = i+1,
                nodes = [
                    self._FEmodel.get_node(node_1_id),
                    self._FEmodel.get_node(node_2_id)
                ],
                degree=self.degree,
                dofs=dofs
            )
            internal_dofs_counter += self.degree - 1
        
        print('(1/4) Constructed the mesh')


    def solve_1d_semi_discrete_uniform_mesh(self, print_info=False, penalty=10e10):
        self.K = self._FEmodel.assemble_stiffness_matrix_uniform_mesh()
        self.M = self._FEmodel.assemble_mass_matrix()

        print('(2/4) System matrices are constructed')

        if print_info:
            print('K matrix:')
            print(self.K.toarray())
            print('\nM matrix:')
            print(self.M.toarray())
        
        dbc = list()
        dbc.append(0)
        dbc.append(self._elements)
        dbc = np.array(dbc)
        self.K[dbc, dbc] += penalty

        print('(3/4) Applied boundary conditions')

        K_dynamic = (self._k * self._c / self._dt) * self.M + self._theta * self.K
        RHS_dynamic = (self._k * self._c / self._dt) * self.M - (1-self._theta)*self.K

        u = np.zeros(self._FEmodel.total_num_dofs)
        self.U = np.zeros((self._timesteps + 1, self._FEmodel.total_num_dofs))

        time_loop = np.linspace(0, (self._timesteps-1)*self._dt, self._timesteps)
        counter = 0
        self.U[0] = u
        for t in time_loop:

            f_n = self._FEmodel.assemble_force_vector(self._force_function, t)
            f_n_plus_1 = self._FEmodel.assemble_force_vector(self._force_function, t+self._dt)

            # Here, we change RHS_dynamic to a dense matrix because f is a dense array
            # TODO: make a better fix
            RHS = np.dot(RHS_dynamic.toarray(), u) + self._theta*f_n_plus_1 + (1-self._theta)*f_n
            u = spsolve(K_dynamic, RHS)

            counter += 1
            self.U[counter] = u


        print('(4/4) Solved')


    def transient_plot_semi_discrete(self):
        fig, ax = plt.subplots()
        ax.grid()
        ax.set(
            title='Semi-Discrete Solution',
            xlim=[0, self._length],
            # ylim=[np.min(self.U), np.max(self.U)],
            ylim=[-1, 1],
        )

        line, = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            return line,

        x_plot = np.linspace(0, self._length, self._elements*self.degree + 1)
        def animate(frame):
            u = list()
            for x in x_plot:
                element = self._FEmodel.find_element_1d(x)
                sol = element.get_solution_point_from_solution_vector(
                    x, self.U[frame, element.dofs])
                u.append(sol)
            line.set_data(x_plot, u)
            # plt.savefig('plot_'+str(frame)+'.png')
            return line,

        anim = animation.FuncAnimation(fig, animate, frames=self._timesteps+1,
            interval=2000/(self._timesteps+1), init_func=init, blit=True,)

        plt.show()
#        anim.save('semidiscrete.gif', fps=30)
        return anim


    def L2_norm_semi_discrete(self, u_analytical):
        """
        Calculates the L2 norm of the error
                  integral((u_analytical(x, t) - u_FEM(x, t))^2 * d_A)**2
        error = (------------------------------------------------------) ** 0.5
                        integral((u_analytical(x, t))^2 * d_A)**2
        """
        nominator = 0
        denominator = 0
        for i in range(1, self._timesteps):
            for element in self._FEmodel.elements:
                nominator += element.integrate_uAnlytical_minus_uFEM_squared(
                    u_analytical,
                    self.U[i, element.dofs],
                    i*self._dt
                )
                denominator += element.integrate_uAnlytical_squared(
                    u_analytical,
                    i*self._dt
                )
        L2 = np.sqrt( nominator / denominator )
        return L2
