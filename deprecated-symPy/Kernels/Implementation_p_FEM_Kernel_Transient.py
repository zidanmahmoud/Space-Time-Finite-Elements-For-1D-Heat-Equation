#=============================================
# Implementation p-FEM: Kernel
#=============================================

# importing required modules
import numpy as np
from numpy import sin
import sympy as sp
from matplotlib import pyplot as plt
import Auxilary.welcome as welcome
import Auxilary.get_gauss_points as gp
from scipy.integrate import simps

# Declaration of necessary symbols
xi, X, t = sp.symbols('xi X t')
# F = sp.symbols('F')


class Truss:
    """
    Class that creates an instance to solve and does
    1D hp version of FEM
    """

    def __init__(self,num_elements, polynomial_degree, num_GP, E, A, L, F, f_x_t, verbosity, theta, rho, c, dt, timesteps):
        """
        Constructor
        """

        # #print a welcome Message
        #welcome.print_welcome_message()
        #print('An instance to be solved has been created!')
        #print('------------------------------------------')

        # Assign Variables
        self._num_elements = num_elements
        self._polynomial_degree = polynomial_degree
        self._num_GP = num_GP
        # Check for the need of Analytical Integration
        if (self._num_GP > 0):
            self._numerical_integration = True
        elif(self._num_GP < 0 or self._num_GP > 8):
            raise ValueError('Number of Gauss Points not implemented')
        else:
            self._numerical_integration = False
        self._E = E
        self._A = A
        self._L = L
        self._F = F
        self._f_x_t = f_x_t
        self._verbosity = verbosity
        self._theta = theta
        self._rho = rho
        self._c = c
        self._dt = dt
        self._timesteps = timesteps
        
        # Declaration of required variables
        self._Le = self._L / self._num_elements
        self._dof_node = 1
        self._dof_element = self._polynomial_degree + 1
        self._location_matrix = np.zeros([self._dof_element, self._num_elements], 'int')
        self._N = [0] * self._dof_element
        self._plot_accuracy = 50
        
        # #print the instance's info
        if (self._verbosity > 0):                     
            self._print_info()


    def _print_info(self):
        """
        #print info of the system before solving
        """
        #print('\nInstance info:')
        #print("\tNumber of Elements: ", self._num_elements)
        #print("\tPolynomial Degree: ", self._polynomial_degree)
        #print("\tTheta = ", self._theta)
        #print("\tNumber of Nodes per Element: ", self._dof_element)
        #print("\tUsing Numerical Integration:", self._numerical_integration, "\n")
    

    def solve(self, show_disp = True):
        """
        A method called by the main file
        to initiate the process of solution
        """
        #print('\nThe solution process has been initiated ...')
        self.show_disp = show_disp
        self._set_shape_functions()
        self._assemble_stiffness_matrix_and_force_vector()
        self._apply_boundary_conditions_and_solve_the_system()


    def _print_shape_functions(self):
        """
        #print the shape functions
        """

        for i in range(len(self._N)):
            #print("N[", i+1, "] = ", self._N[i])
            pass
    

    def _plot_shape_functions(self,i):
        """
        Plot the shape functions
        """

        # x-axis
        x_plot = np.linspace(-1,1,self._plot_accuracy)
        colors = ['-b', '-r', '-k', 'c', 'm', 'y', 'g', 'darkviolet', 'peru']
        
        fig_SF = plt.figure()
        
        expr = sp.lambdify(xi, self._N[i], "numpy")
        N_plot = np.zeros([self._polynomial_degree+1, self._plot_accuracy])
        
        N_plot[i,:] = expr(x_plot)
        label_plot = "N" + str(i+1)
        plt.plot(x_plot, N_plot[i,:], colors[i%len(colors)], lw=1.5, label=label_plot)
        
        plt.title('Shape Functions',fontsize=20)
        plt.xlabel(r'$\xi$',fontsize=15)
        fig_SF.canvas.set_window_title("Shape Functions")
        
        plt.ylim(-1,1)
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig('N'+str(i+1), format='jpeg', dpi=300)
        plt.show()
        
        
    def _set_shape_functions(self):
        """
        Set the shape functions
        """
        #print("\n... Setting Shape Functions ...")

        total_dof = self._polynomial_degree * self._num_elements + 1
        self._num_dof = total_dof * self._dof_node
        self._x_vector = [0] * total_dof
        
        # Legendre Shape functions
        N1 = 1/2 * (1 - xi)
        N2 = 1/2 * (1 + xi)
        N3 = 1/4 * np.sqrt(6) * (xi**2 - 1)
        N4 = 1/4 * np.sqrt(10) * (xi**2 - 1) * xi
        N5 = 1/16 * np.sqrt(14) * (5*xi**4 - 6*xi**2 + 1)
        N6 = 3/16 * np.sqrt(2) * xi * (7*xi**4 - 10*xi**2 + 3)
        N7 = 1/32 * np.sqrt(22) * (21*xi**6 - 35*xi**4 + 15*xi**2 - 1)
        N8 = 1/32 * np.sqrt(26) * xi * (33*xi**6 - 63*xi**4 + 35*xi**2 - 5)
        N9 = 1/256 * np.sqrt(30) * (-140*xi**2 - 924*xi**6 + 630*xi**4 + 5 + 429*xi**8)
        
        shape_functions = [N1, N2, N3, N4, N5, N6, N7, N8, N9]
        
        for i in range(len(self._N)):
            self._N[i] = shape_functions[i]
         
        for i in range(self._num_elements):
            for j in range(self._dof_element):
                if (j == 0):
                    self._location_matrix[j,i] = self._polynomial_degree*i + j + 1
                if (j == 1):
                    self._location_matrix[j,i] = self._polynomial_degree*i + self._dof_element 
                if (j > 1):
                    self._location_matrix[j,i] = self._polynomial_degree*i + j 
        
        for i in range(total_dof):
            self._x_vector[i] = i * self._Le / self._polynomial_degree 
        
        if (self._verbosity > 0):
            self._print_shape_functions()
            #for i in range(self._polynomial_degree + 1):
            #    self._plot_shape_functions(i)
         
        if (self._verbosity > 1):
            #print("Location Matrix:")
            #print(self._location_matrix, "\n")
            #print("Nodal Positions X-Direction:")
            #print(self._x_vector, "\n")
            #print("Number of Degrees of freedom:")
            #print(self._num_dof, "\n")
            #print("Lenght of N:")
            #print(len(self._N), "\n")
            pass


    def _perform_numerical_integration(self, x, B, J, e):
        """
        Gaussian Integration for RHS and LHS
        """
        xg, wg = gp.get_gauss_points(self._num_GP)

        Me = sp.zeros(self._dof_element, self._dof_element)
        Ke = sp.zeros(self._dof_element, self._dof_element)
        Fe = sp.zeros(self._dof_element,1)

        # f(x(xi)) .. Global to local mapping
        f_x_xi = self._f_x_t.subs({X:x})
        
        M_expr = sp.lambdify(xi, sp.Matrix(np.multiply(self._N*np.array([self._N]).T, (self._Le/2))), "numpy")
        K_expr = sp.lambdify(xi, sp.Matrix(np.multiply(B*np.array([B]).T, (2/self._Le)* self._E*self._A)), "numpy")
        fe_expr = sp.lambdify(xi, sp.Matrix(np.multiply(self._N, (self._Le/2) * f_x_xi)), "numpy")
                
        for i in range(self._num_GP):
            Me += wg[i] * M_expr(xg[i])
            Ke += wg[i] * K_expr(xg[i])
            Fe += wg[i] * fe_expr(xg[i])
        
        if (self._verbosity > 1):
            e += 1
            #print("K_e_" , e , ":\n", Ke , "\n")
            #print("f_e_" , e , ":\n", Fe , "\n")
        
        Me = Me * self._rho * self._c / self._dt

        return Me, Ke, Fe


    def _gaussian_integration(self, func, num_GP):
        """FIXME"""
        xg, wg = gp.get_gauss_points(num_GP)
        func_expr = sp.lambdify(xi, func, "numpy")
        fun = 0
        for i in range(self._num_GP):
            fun += wg[i] * func_expr(xg[i])
        return fun




    def _perform_analytical_integration(self, x, B, J, index):
        """
        Analytical Integration using the symbolic library in Python
        """
        
        Me = self._rho * self._c / self._dt * sp.integrate(sp.Matrix(np.multiply(self._N*np.array([self._N]).T, (2/self._Le))), (xi, -1, 1))
        Ke = sp.integrate(sp.Matrix(np.multiply(B*np.array([B]).T, (2/self._Le) * self._E * self._A)),(xi, -1, 1))
        
        # f(x(xi)) .. Global to local mapping
        f_x_xi = self._f_x_t.subs({X:x})
        Fe = sp.integrate(sp.Matrix(np.multiply(self._N, (self._Le/2) * f_x_xi)),(xi, -1, 1))              
        
        if (self._verbosity > 1):   
            index += 1
            #print("M_e_" + str(index), ":\n", Me, "\n")
            #print("K_e_" + str(index), ":\n", Ke, "\n")
            #print("f_e_" + str(index), ":\n", Fe, "\n")
        
        return Me, Ke, Fe
        

    def _assemble_stiffness_matrix_and_force_vector(self):
        """
        Assemble Element Stiffness Matrix and Force Vector based on the Location Matrix
        """
        #print("\n... Assembling Stiffness Matrix and Force Vector ...")
        
        self._M = sp.zeros(self._num_dof,self._num_dof)
        self._K = sp.zeros(self._num_dof,self._num_dof)
        self._f_body = sp.zeros(1, self._num_dof)
        
        self.x = [None] * self._num_elements
        for e in range(self._num_elements): # loop over elements
            self.x[e] = 0
            for i in range(2): # defines the mapping of x to the parametric coordinate xi, in our case should be 2, as x = N1*X1 + N2*X2
                active_nodes = self._location_matrix[i,e]
                self.x[e] += self._N[i] * self._x_vector[active_nodes - 1]
            
            J = sp.diff(self.x[e], xi)
            
            if (self._verbosity > 0):
                #print("x =", self.x[e])
                #print("J =", J)
                pass
                
            B = [0] * self._dof_element # B-Matrix, Shape Function Derivatives
            for i in range(self._dof_element):
                for j in range(self._dof_node):
                    B[self._dof_node*i + j] = sp.diff(self._N[i], xi) 
            
            if (self._numerical_integration): # numerical Integration
                Me, Ke, Fe = self._perform_numerical_integration(self.x[e], B, J, e)
            else: # analytical Integration
                Me, Ke, Fe = self._perform_analytical_integration(self.x[e], B, J, e)
            
            # Assemble Stiffness Matrix and Force Vector
            for i in range(self._dof_element):
                for j in range(self._dof_element):
                    ii = self._location_matrix[i,e]
                    jj = self._location_matrix[j,e]
                    self._M[ii-1, jj-1] = self._M[ii-1, jj-1] + Me[i,j]
                    self._K[ii-1, jj-1] = self._K[ii-1, jj-1] + Ke[i,j]
                self._f_body[ii-1] = self._f_body[ii-1] + Fe[i]
        
        if (self._verbosity > 0):
            #print("K:\n", self._K, "\n")
            #print("M:\n", self._M, "\n")
            #print("f_body:\n", self._f_body, "\n")
            pass
        
    
    def _apply_boundary_conditions_and_solve_the_system(self):
        """
        Reduce the system of Equations assuming u[0]=0 and solving using numpy
        """
        #print("\n... Applying Boundary Conditions and Solving the System ...")

        self.M_red = self._M[1:,1:]
        self.K_red = self._K[1:,1:]
        self.f_red = sp.zeros(self._num_dof - 1, 1)
        self.f_red[-1] = self._F
        
        for i in range(self._num_dof - 1):
            self.f_red[i] = self.f_red[i] + self._f_body[i+1]
        
        if (self._verbosity > 0):
            #print("M_red:\n", self.M_red, "\n")
            #print("K_red:\n", self.K_red, "\n")
            #print("f_red:\n", self.f_red, "\n")
            pass


        u_red = sp.zeros(self._num_dof - 1, 1)
        if (self.show_disp == True):
            fig_disp = plt.figure()
        
        # error veriable will be used in _post_processing
        self.error = [0] * self._timesteps

        K_dynamic = self.M_red + self._theta * self.K_red
        RHS_dynamic = self.M_red - (1-self._theta)*self.K_red
        
        for iteration in range(self._timesteps):
            if(self._verbosity > 0):
                #print('\nIteration '+str(iteration)+' :')
                #print('----------------------------')
                pass
            time = iteration * self._dt
            
            # To shut down the heat flux
            #if(time>1000):
            #    f = sp.zeros(self._num_dof-1, 1)
            #    f[-1] = self._F
            #else:
            f_n = self.f_red.subs(t , time)
            f_n1 = self.f_red.subs(t , time+self._dt)
            #print(f_n1-f_n)
            
            u = u_red
            u_red = K_dynamic.LUsolve(RHS_dynamic*u + self._theta*f_n1 + (1-self._theta)*f_n)
            self.u = sp.zeros(1, 1).col_join(u_red) # Full Displacement Vector
            
            if (self._verbosity > 0):
                #print("Displacement Vector u:\n", self.u, "\n")
                pass

            self._post_processing(time, iteration)
            
            if(iteration > 0 and self.show_disp == True):
                self._plot_results(time, fig_disp)
        
        plt.show()
        # Trapezoidal Integration
        if(self._verbosity > -1):
            print('error list = ', self.error)
        time_integration = sum(self.error)*self._dt
        self.error_L2 = np.sqrt(time_integration)
        if(self._verbosity > -1):
            print('Error L_2 = ', self.error_L2)
        
                
            
            
    def _post_processing(self, time, iteration):
        """
        Necessary calculations to plot the displacement field
        """
        ##print("\n... Post-Processing ...")

        self.u_plot = [0] * self._num_elements
        self.x_plot = [0] * self._num_elements
        
        
        dummy_array = np.linspace(0, self._Le, 1000)
        for i in range(self._num_elements):
            disp=0
            for j in range(self._dof_element):
                disp += self._N[j] * self.u[self._location_matrix[j,i]-1]
            
            # Analytical Manufactured Solution
            u_analytical = sp.sin(3*sp.pi/2 * X) * (time+self._dt)
            u_an = u_analytical.subs(X, self.x[i])

            print('u_analytical = ', u_an)
            print('u_approx = ', disp)

            e = (u_an - disp)**2
            error_el = self._gaussian_integration(e, 33)
            #e_expr = sp.lambdify(xi, e, "numpy")
            #e_expr_array = e_expr(dummy_array)
            #error_el = simps(y=e_expr_array, x=dummy_array)
            self.error[iteration] += error_el
            #print('for time = ', time)
            #print('error = ', self.error)

            if (self._verbosity > 0):
                #print("u_approx, Element", str(i+1), ":\n", disp, '\n')
                #print("u_analyt, Element", str(i+1), ":\n", u_an, '\n')
                pass

            self.x_plot[i] = np.linspace(float(self._Le*i),float(self._Le*(i+1)),self._plot_accuracy)
                
            mapping = -1 + 2*(X-self._Le*(i))/self._Le

            disp_expr_tmp = sp.lambdify(xi, disp, "numpy")
            disp_expr = sp.lambdify(X, disp_expr_tmp(mapping), "numpy")
            
            self.u_plot[i] = disp_expr(self.x_plot[i])

           
        
        if(self._verbosity > 0):
            #print('integrated error in space = ', self.error[int(time/self._dt)-1])
            pass
        
        #strain_energy = sp.sqrt(strain_energy[0])
        #strain_energy_exact = 0.06544088433080137260487994866374172246756572795350655893067784512107572226722403942992936169523014756802289277390
        #self.error = (strain_energy_exact - strain_energy) / strain_energy_exact
        #if (self._verbosity > 0):
        #    #print("Strain Energy (Approx.)= ", strain_energy)
        #    #print("Strain Energy ( Exact )= ", strain_energy_exact)
        #    #print("Error in the Energy Norm = ", self.error * 100, "%")



    def _plot_results(self, time, figure):
        """
        Plot the displacement field    
        """
        #print("\n... Plotting Results")

        # Plot Displacements
        
        u_plot = list()
        x_plot = list()
        for i in range(self._num_elements):
            u_plot.extend(self.u_plot[i])
            x_plot.extend(self.x_plot[i])
        FE, = plt.plot(x_plot, u_plot, 'b', lw=1.5, marker='o', markevery=(self._plot_accuracy), markerfacecolor='None')
        FE.set_label('FE Solution')
        plt.legend(loc='best')
        
        x_plot = np.linspace(0,self._L, 50)
        u_analytical = sp.sin(3*sp.pi/2 * X) * (time+self._dt)



        plt.title('Temperature (Polynomial Degree:'+str(self._polynomial_degree)+' Elements:'+str(self._num_elements)+'), t = '+str(time+self._dt))
        plt.xlabel('X',fontsize=15)
        #plt.ylim(-0.05,0.05)
        figure.canvas.set_window_title('Temperature (Polynomial Degree:'+str(self._polynomial_degree)+' Elements:'+str(self._num_elements)+')')
        plt.grid(True)
        plt.hold(False)
        plt.pause(0.05)
        #plt.savefig('Displacement_p'+str(self._polynomial_degree)+'_e'+str(self._num_elements),format='jpeg', dpi=300)
        