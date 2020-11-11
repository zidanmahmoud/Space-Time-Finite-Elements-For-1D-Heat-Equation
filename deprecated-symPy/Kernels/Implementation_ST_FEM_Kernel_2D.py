#=============================================
# Implementation p-FEM: Kernel
#=============================================

# importing required modules
import numpy as np
from numpy import sin
import sympy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import Auxilary.welcome as welcome
import Auxilary.shape_funtions_2D as shape_fun
import Auxilary.get_gauss_points as gp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Declaration of necessary symbols
xi, eta, X, t = sp.symbols('xi eta X t')
# F = sp.symbols('F')


class Truss:
    """
    Class that creates an instance to solve and does
    1D hp version of FEM
    """

    def __init__(self,height, width, num_elements_x, num_elements_y, polynomial_degree, num_GP, k, f, verbosity):
        """
        Constructor
        """

        # Print a welcome Message
        welcome.print_welcome_message()
        print('An instance to be solved has been created!')
        print('------------------------------------------')

        # Assign Variables
        self._height = height
        self._width = width
        self._num_elements_x = num_elements_x
        self._num_elements_y = num_elements_y
        self._num_elements_tot = self._num_elements_x * self._num_elements_y
        self._sizex = self._width / self._num_elements_x
        self._sizey = self._height / self._num_elements_y
        self._polynomial_degree = polynomial_degree
        self._num_GP = num_GP
        # Check for the need of Analytical Integration
        if (self._num_GP > 0):
            self._numerical_integration = True
        elif(self._num_GP < 0 or self._num_GP > 8):
            raise ValueError('Number of Gauss Points not implemented')
        else:
            self._numerical_integration = False
        self._k = k
        self._f = f
        self._verbosity = verbosity
        
        # Declaration of required variables
        self._num_dof = (self._num_elements_x + 1)*(self._num_elements_y + 1)
        self._num_nodes = (self._num_elements_x + 1)*(self._num_elements_y + 1)
        self._num_elements = self._num_elements_x*self._num_elements_y
        self._nodes_coords = list()
        dx = self._width / self._num_elements_x
        dy = self._height / self._num_elements_y
        counter_x=0; counter_y=0
        for inode in range(self._num_nodes):
            self._nodes_coords.append([counter_x*dx,counter_y*dy])
            counter_x += 1
            if (counter_x % (self._num_elements_x+1)) == 0:
                counter_y += 1
                counter_x = 0
        self._dof_node = 4 # or 8??? :( 
        self._dof_edge = 0
        self._dof_internal = 0
        if self._polynomial_degree >= 2:
            self._dof_edge = (self._polynomial_degree - 1) * 4
        if self._polynomial_degree >= 4:
            n = self._polynomial_degree - 3
            self._dof_internal = (n * (n + 1)) / 2
			
        self._dof_element = self._dof_node + self._dof_edge + self._dof_internal
        self._location_matrix = np.zeros([self._dof_element, self._num_elements_tot], 'int')
        self._plot_accuracy = 50
        
        # Print the instance's info
        if (self._verbosity > 0):                     
            self._print_info()
        
        # Plotting options
        self.plot_accuracy = 50
        self.anti_aliasing = False
        self.z_ticks = np.linspace(-1,1,11)


    def _print_info(self):
        """
        Print info of the system before solving
        """
        print('\nInstance info:')
        print("\tNumber of Elements in x: ", self._num_elements_x)
        print("\tNumber of Elements in y: ", self._num_elements_y)
        print("\tNumber of Elements in domain: ", self._num_elements_tot)
        print("\tPolynomial Degree: ", self._polynomial_degree)
        print("\tNumber of Nodes per Element: ", self._dof_element)
        print("\tUsing Numerical Integration:", self._numerical_integration, "\n")
    

    def solve(self, show_disp = True):
        """
        A method called by the main file
        to initiate the process of solution
        """
        print('\nThe solution process has been initiated ...')

        self._set_shape_functions()
        self._assemble_stiffness_matrix_and_force_vector()
        self._apply_boundary_conditions_and_solve_the_system()
        self._post_processing()
        #if(show_disp == True):
        #    self._plot_results()


    def _print_shape_functions(self):
        """
        Print the shape functions
        """

        print("#####################")
        print("Nodal Shape Functions")
        for i in range(len(self._nodal_shape_functions)):
            print("N[", i+1, "] = ", self._nodal_shape_functions[i])
            
        if self._polynomial_degree >= 2 :
            print("#####################")
            print("Edge Shape Functions")
            for i in range (4):
                print("Edge : ", i + 1)
                for j in range (self._polynomial_degree - 1):
                    print("N[", i, ", ", j, "] = ", self._edge_shape_functions[i, j])
                
        if self._polynomial_degree >= 4 :
            print("#####################")
            print("Internal Shape Functions")
            print("number of internal Shape Functions = ", len(self._internal_shape_functions))
            print((self._polynomial_degree - 2)*(self._polynomial_degree - 3)// 2)
            for i in range (self.number_int_shape_fun):
                print("N[", 0, ", ", i, "] = ", self._internal_shape_functions[0, i])
    

    def SymbolicEvaluation(self,N,x_1,x_2): # Function for convertig symbolic to numeric expressions       
        """
        This is just useful for the function _plot_shape_functions
        """
        if np.shape(x_1) != np.shape(x_2): # check input for correct shape:
            raise ValueError('Input of Symbolic Evaluation is not consistent!')
        
        expr = sp.lambdify((xi,eta), N, "numpy")
        n_eval = expr(x_1, x_2)
        
        return n_eval
    

    def _plot_shape_functions(self):
        """
        Plot the shape functions
        """
        num_col_im = 0
        if self._polynomial_degree >= 4:
            num_col_im = self._polynomial_degree - 3
            
        fig, axes = plt.subplots(nrows = self._polynomial_degree, ncols = 4 + num_col_im, figsize = (3, 3), subplot_kw = {'projection': '3d'})
        
        X = np.linspace(-1,1,self.plot_accuracy)
        Y = np.linspace(-1,1,self.plot_accuracy)
        X, Y = np.meshgrid(X, Y)
        
        # Plotting Nodal Shape Functions 
        for i in range (4):
            Z = self.SymbolicEvaluation(self._nodal_shape_functions[i],X,Y)
            axes[0,i].plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.nipy_spectral, linewidth=0., antialiased=self.anti_aliasing)
            #axes[0,i].set_xlabel(r'$\xi$',fontsize=15)
            #axes[0,i].set_ylabel(r'$\eta$',fontsize=15)
            #axes[0,i].set_title("Nodal Shape Function " + str(i))
        
        # Plotting Edge Shape Functions if they are defined
        l = 0
        if self._polynomial_degree >= 2 : 
            for j in range (self._polynomial_degree - 1):
                for i in range (4):
                    Z = self.SymbolicEvaluation(self._edge_shape_functions[i, j],X,Y)
                    axes[j + 1,i].plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.nipy_spectral, linewidth=0., antialiased=self.anti_aliasing)
                    axes[j + 1,i].set_xlabel(r'$\xi$',fontsize=15)
                    axes[j + 1,i].set_ylabel(r'$\eta$',fontsize=15)
                    #axes[j + 1,i].set_title("Edge Shape Function " + str(i))
                    
                    # Plotting Internal Shape Functions if they are defined
                if j >= 2:
                    for k in range (j - 1):
                        Z = self.SymbolicEvaluation(self._internal_shape_functions[0, l],X,Y)
                        axes[j + 1, 4 + k].plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.nipy_spectral, linewidth=0., antialiased=self.anti_aliasing)
                        axes[j + 1, 4 + k].set_xlabel(r'$\xi$',fontsize=15)
                        axes[j + 1, 4 + k].set_ylabel(r'$\eta$',fontsize=15)
                        l = l + 1
        
        plt.show()


    def _set_shape_functions(self):
        """
        Set the shape functions
        """
        print("\n... Setting Shape Functions ...")
		
        total_dof = (self._dof_element)*self._num_elements_tot - 3*(self._num_elements_tot - 1)

        # Nodal shape functions
        N1 = 1/4 * (1 - xi)*(1 - eta)
        N2 = 1/4 * (1 + xi)*(1 - eta)
        N3 = 1/4 * (1 + xi)*(1 + eta)
        N4 = 1/4 * (1 - xi)*(1 + eta)
        
        self._nodal_shape_functions = [N1, N2, N3, N4]
        
        # Edge shape functions
        if self._polynomial_degree >= 2:
            self._edge_shape_functions = sp.zeros(4, self._polynomial_degree - 1)
            
            for i in range(self._polynomial_degree - 1):
                self._edge_shape_functions[0, i] = (1/2) * (1 - eta) * shape_fun.phi(xi, i + 2)
                self._edge_shape_functions[1, i] = (1/2) * (1 - xi) * shape_fun.phi(eta, i + 2)
                self._edge_shape_functions[2, i] = (1/2) * (1 + eta) * shape_fun.phi(xi, i + 2)
                self._edge_shape_functions[3, i] = (1/2) * (1 + xi) * shape_fun.phi(eta, i + 2)

        # Internal shape functions
        if self._polynomial_degree >= 4:
            self.number_int_shape_fun = (self._polynomial_degree - 2)*(self._polynomial_degree - 3)// 2
            self._internal_shape_functions = sp.zeros(1, self.number_int_shape_fun)
        
            k = 0
            for j in range (4, self._polynomial_degree + 1):
                for i in range (2, j - 1):
                    self._internal_shape_functions[0, k] = shape_fun.phi(xi, i) * shape_fun.phi(eta, j - 2)
                    k = k + 1
                    j = j - 1
        
        j = 0 # Counter to assign index to degree of freedom (node, edge and internal modes)
        for i in range(1, self._num_elements_tot + 1):
            # nodal dof
            self._location_matrix[0, i-1] = i + j  
            self._location_matrix[1, i-1] = i + j + 1
            self._location_matrix[2, i-1] = i + self._num_elements_x + j + 2
            self._location_matrix[3, i-1] = i + self._num_elements_x + j + 1
			
            if (i % self._num_elements_x) == 0:
                j = j + 1 
        
        if (self._verbosity > 0):
            self._print_shape_functions()
            #self._plot_shape_functions()
         
        if (self._verbosity > 1):
            print("Location Matrix:")
            print(self._location_matrix, "\n")
            #print("Nodal Positions X-Direction:")
            #print(self._x_vector, "\n")
            #print("Number of Degrees of freedom:")
            #print(self._num_dof, "\n")
            #print("Lenght of N:")
            #print(len(self._N), "\n")


    def _assemble_stiffness_matrix_and_force_vector(self):
        """
        Assemble Element Stiffness Matrix and Force Vector based on the Location Matrix
        """
        print("\n... Assembling Stiffness Matrix and Force Vector ...")
        
        self._K = np.zeros((self._num_dof,self._num_dof))
        self._f_body = np.zeros((self._num_dof,1))
        
        #TODO: xg, wg = gp.get_gauss_points(self._num_GP)
        xg =  np.matrix([[-0.577350269189626, -0.577350269189626],
                        [ 0.577350269189626, -0.577350269189626],
                        [ 0.577350269189626,  0.577350269189626],
                        [-0.577350269189626,  0.577350269189626]])
        wg = np.array([1,1,1,1])

        for e in range(self._num_elements): # loop over elements
            # Declaration of Matrices
            A = np.zeros((4,4))
            B = np.zeros((4,4))
            Ke = np.zeros((4,4))
            Fe = np.zeros((4,1))
            dNdxi  = sp.zeros(4,1)
            dNdeta = sp.zeros(4,1)
            coords = np.zeros((4,2))
            x = y = 0

            for i in range(4):
                # Differentiating shape functions
                dNdxi[i]  = sp.diff(self._nodal_shape_functions[i], xi)
                dNdeta[i] = sp.diff(self._nodal_shape_functions[i],eta)
                # Getting the nodal coordinates of each elements
                coords[i] = self._nodes_coords[self._location_matrix[i,e]-1]
                # Linear mapping from global to local
                x += self._nodal_shape_functions[i] * coords[i,0]
                y += self._nodal_shape_functions[i] * coords[i,1]
            
            f = self._f.subs({X:x, t:y})
            for iGP in range(4):
                # Gauss Points:
                xiGP  = xg[iGP,0]
                etaGP = xg[iGP,1]
                weight= wg[iGP]

                dN = np.zeros((2,4))
                N = np.zeros((1,4))
                for i in range(4):
                    # Matrices for shape functions
                    N[0,i]  = self._nodal_shape_functions[i].subs({xi:xiGP,eta:etaGP})
                    dN[0,i] = dNdxi[i].subs({xi:xiGP,eta:etaGP})
                    dN[1,i] = dNdeta[i].subs({xi:xiGP,eta:etaGP})
                
                # Mapping "f" to local coords
                f_gauss = np.zeros((1,1))
                f_gauss[0] = f.subs({xi:xiGP, eta:etaGP})

                # Jacobian
                J = np.dot(dN,coords)
                J_inv = np.linalg.inv(J)
                det_J = np.linalg.det(J)
                
                # from parametric differentials to coordinate differentials
                dN_global = np.dot(J_inv,dN)
                dNdx = np.array([dN_global[0]])
                dNdt = np.array([dN_global[1]])

                # Calculating element K Matrix
                A = np.dot(N.T , dNdt) * weight * det_J
                B = np.dot(dNdx.T , dNdx) * self._k * weight * det_J
                Ke += (A + B)
                Fe += f_gauss * N.T * weight *det_J
            
            # Assembly of the global K Matrix
            for ii in range(4):
                dof1 = self._location_matrix[ii,e] - 1
                for jj in range(4):
                    dof2 = self._location_matrix[jj,e] - 1
                    self._K[dof1,dof2] += Ke[ii,jj]
                self._f_body[dof1] += Fe[ii] 

        if (self._verbosity > 0):
            print("K:\n", self._K, "\n")
            print("f_body:\n", self._f_body, "\n")
        
    
    def _apply_boundary_conditions_and_solve_the_system(self):
        """
        Reduce the system of Equations assuming u[0]=0 and solving using numpy
        """
        print("\n... Applying Boundary Conditions and Solving the System ...")

        corner1 = 0
        corner2 = self._num_elements_x
        corner3 = self._num_elements_x*self._num_elements_y + self._num_elements_y
        corner4 = self._num_elements_x*self._num_elements_y + self._num_elements_x + self._num_elements_y
        dbc1 = np.linspace(corner1 , corner2 , self._num_elements_x + 1, dtype=int)
        #dbc2 = np.linspace(corner2 , corner4 , self._num_elements_y + 1, dtype=int)
        #dbc3 = np.linspace(corner3 , corner4 , self._num_elements_x + 1, dtype=int)
        dbc4 = np.linspace(corner1 , corner3 , self._num_elements_y + 1, dtype=int)
        dbc = np.append(dbc1,dbc4)
        penalty = 10e10

        for i in range(len(dbc)):
            self._K[dbc[i],dbc[i]] += penalty
            self._f_body[dbc[i]] += 0*penalty
    
        nbc = np.linspace(corner2 , corner4 , self._num_elements_y + 1, dtype=int)
        for i in range(len(nbc)):
            self._f_body[nbc] = 0

        #self.u = np.linalg.solve(self._K,self._f_body)
        sparse_k = csr_matrix(self._K)
        u = spsolve(sparse_k,self._f_body)
        self.u = np.array([u]).T

        if (self._verbosity > 0):
            print("Solution u:\n", self.u, "\n")
            
            
    def _post_processing(self):
        """
        Necessary calculations to plot the displacement field
        """
        print("\n... Post-Processing ...")

        temperature = np.reshape(self.u, (self._num_elements_y+1,self._num_elements_x+1))
        x = np.linspace(0,self._width,self._num_elements_x+1)
        y = np.linspace(0,self._height,self._num_elements_y+1)
        X,Y = np.meshgrid(x,y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X,Y,temperature, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        #ax.set_zlabel('Temperature')

        #plt.savefig('SpaceTime1D(2).jpg',format='jpeg', dpi=300)
        plt.show()

    
    def transient_plot(self):
        """
        Transient Plot
        """
        print('\n... Transient Plot')

        x_plot = np.linspace(0,self._width,self._num_elements_x+1)
        fig = plt.figure()
        for i in range(self._num_elements_y+1):
            u_plot = list()
            index = i*(self._num_elements_x+1)
            while (index == i*(self._num_elements_x+1)) or (index % (self._num_elements_x+1) != 0):
                u_plot.append(self.u[index])
                index += 1
            plt.plot(x_plot, u_plot, label='Space-Time FEM Solution')
            plt.grid(True)
            plt.legend(loc='best')
            plt.ylim(np.min(self.u),np.max(self.u))
            dx = self._width/self._num_elements_x
            dt = self._height/self._num_elements_y
            t = "{:.2f}".format(i*dt)
            plt.title('Temperature (dx:'+str(dx)+' dt:'+str(dt)+'), t = '+t)
            plt.pause(0.1)
            plt.hold(False)
            #plt.savefig('Transient'+str(i)+'.jpg',format='jpeg', dpi=300)
        plt.show()
