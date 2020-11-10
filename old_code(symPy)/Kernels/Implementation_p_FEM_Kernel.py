#=============================================
# Implementation p-FEM: Kernel
#=============================================

# importing required modules
import numpy as np
from numpy import sin
import sympy as sp
from matplotlib import pyplot as plt
import Auxilary.welcome as welcome

# Declaration of necessary symbols
xi, X = sp.symbols('xi X')
# F = sp.symbols('F')


class Truss:
    """
    Class that creates an instance to solve and does
    1D hp version of FEM
    """

    def __init__(self,num_elements, polynomial_degree, num_GP, E, A, L, F, f_x, verbosity):
        """
        Constructor
        """

        # Print a welcome Message
        welcome.print_welcome_message()
        print('An instance to be solved has been created!')
        print('------------------------------------------')

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
        self._f_x = f_x
        self._verbosity = verbosity
        
        # Declaration of required variables
        self._Le = self._L / self._num_elements
        self._dof_node = 1
        self._dof_element = self._polynomial_degree + 1
        self._location_matrix = np.zeros([self._dof_element, self._num_elements], 'int')
        self._N = [0] * self._dof_element
        self._plot_accuracy = 50
        
        # Print the instance's info
        if (self._verbosity > 0):                     
            self._print_info()


    def _print_info(self):
        """
        Print info of the system before solving
        """
        print('\nInstance info:')
        print("\tNumber of Elements: ", self._num_elements)
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
        if(show_disp == True):
            self._plot_results()


    def _print_shape_functions(self):
        """
        Print the shape functions
        """

        for i in range(len(self._N)):
            print("N[", i+1, "] = ", self._N[i])
    

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
        print("\n... Setting Shape Functions ...")

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
            for i in range(self._polynomial_degree + 1):
                self._plot_shape_functions(i)
         
        if (self._verbosity > 1):
            print("Location Matrix:")
            print(self._location_matrix, "\n")
            print("Nodal Positions X-Direction:")
            print(self._x_vector, "\n")
            print("Number of Degrees of freedom:")
            print(self._num_dof, "\n")
            print("Lenght of N:")
            print(len(self._N), "\n")


    def _perform_numerical_integration(self, x, B, J, e):
        """
        Gaussian Integration for RHS and LHS
        """

        if (self._num_GP == 8):
            xg = [0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774, 0.5255324099, -0.5255324099, 0.1834346425, -0.1834346425]
            wg = [0.1012285363, 0.1012285363, 0.2223810345, 0.2223810345, 0.3137066459, 0.3137066459, 0.3626837834, 0.3626837834]
        elif (self._num_GP == 7):
            xg = [0.9491079123, -0.9491079123, 0.7415311856, -0.7415311856, 0.4058451514, -0.4058451514, 0.0]
            wg = [0.1294849662, 0.1294849662, 0.2797053915, 0.2797053915, 0.3818300505, 0.3818300505, 0.4179591837]
        elif (self._num_GP == 6):
            xg = [0.9324695142, -0.9324695142, 0.6612093865, -0.6612093865, 0.2386191861, -0.2386191861]
            wg = [0.1713244924, 0.1713244924, 0.3607615730, 0.3607615730, 0.4679139346, 0.4679139346]
        elif (self._num_GP == 5):
            xg = [ 0.0 , (1/3) * np.sqrt(5 - 2*np.sqrt(10/7)) , -(1/3) * np.sqrt(5 - 2*np.sqrt(10/7)) , (1/3) * np.sqrt(5 + 2*np.sqrt(10/7)) , -(1/3) * np.sqrt(5 + 2*np.sqrt(10/7)) ]
            wg = [ 128/225 , (322 + 13*np.sqrt(70))/900 , (322 + 13*np.sqrt(70))/900 , (322 - 13*np.sqrt(70))/900 , (322 - 13*np.sqrt(70))/900 ]
        elif (self._num_GP == 4):
            xg = [ np.sqrt((3 - 2*np.sqrt(6/5))/7), -np.sqrt((3 - 2*np.sqrt(6/5))/7) , np.sqrt((3 + 2*np.sqrt(6/5))/7) , -np.sqrt((3 + 2*np.sqrt(6/5))/7)]
            wg = [ (18 + np.sqrt(30))/36 , (18 + np.sqrt(30))/36 , (18 - np.sqrt(30))/36 , (18 - np.sqrt(30))/36 ]
        elif (self._num_GP == 3):
            xg = [ -np.sqrt(3/5) , 0 , np.sqrt(3/5) ]
            wg = [ 5/9 , 8/9 , 5/9 ]
        elif (self._num_GP == 2):
            xg = [ -np.sqrt(1/3) , np.sqrt(1/3) ]
            wg = [ 1 , 1 ]
        elif (self._num_GP == 1):
            xg = [ 0 ]
            wg = [ 2 ]


        Ke = sp.zeros(self._dof_element, self._dof_element)
        Fe = sp.zeros(self._dof_element,1)

        # f(x(xi)) .. Global to local mapping
        f_x_xi = self._f_x.subs({X:x})
        
        K_expr = sp.lambdify(xi, sp.Matrix(np.multiply(B*np.array([B]).T, (2/self._Le)* self._E*self._A)), "numpy")
        fe_expr = sp.lambdify(xi, sp.Matrix(np.multiply(self._N, (self._Le/2) * f_x_xi)), "numpy")
                
        for i in range(self._num_GP):
            Ke += wg[i] * K_expr(xg[i])
            Fe += wg[i] * fe_expr(xg[i])
        
        if (self._verbosity > 1):
            e += 1
            print("K_e_" , e , ":\n", Ke , "\n")
            print("f_e_" , e , ":\n", Fe , "\n")
        
        return Ke, Fe


    def _perform_analytical_integration(self, x, B, J, index):
        """
        Analytical Integration using the symbolic library in Python
        """

        Ke = sp.integrate(sp.Matrix(np.multiply(B*np.array([B]).T, (2/self._Le) * self._E * self._A)),(xi, -1, 1))
        
        # f(x(xi)) .. Global to local mapping
        f_x_xi = self._f_x.subs({X:x})
        Fe = sp.integrate(sp.Matrix(np.multiply(self._N, (self._Le/2) * f_x_xi)),(xi, -1, 1))              
        
        if (self._verbosity > 1):   
            index += 1
            print("K_e_" + str(index), ":\n", Ke, "\n")
            print("f_e_" + str(index), ":\n", Fe, "\n")
        
        return Ke, Fe
        

    def _assemble_stiffness_matrix_and_force_vector(self):
        """
        Assemble Element Stiffness Matrix and Force Vector based on the Location Matrix
        """
        print("\n... Assembling Stiffness Matrix and Force Vector ...")
        
        self._K = sp.zeros(self._num_dof,self._num_dof)
        self._f_body = sp.zeros(1, self._num_dof)
        
        for e in range(self._num_elements): # loop over elements
            x = 0
            for i in range(2): # defines the mapping of x to the parametric coordinate xi, in our case should be 2, as x = N1*X1 + N2*X2
                active_nodes = self._location_matrix[i,e]
                x += self._N[i] * self._x_vector[active_nodes - 1]
            
            J = sp.diff(x, xi)
            
            if (self._verbosity > 0):
                print("x =", x)
                print("J =", J)
                
            B = [0] * self._dof_element # B-Matrix, Shape Function Derivatives
            for i in range(self._dof_element):
                for j in range(self._dof_node):
                    B[self._dof_node*i + j] = sp.diff(self._N[i], xi) 
            
            if (self._numerical_integration): # numerical Integration
                Ke, Fe = self._perform_numerical_integration(x, B, J, e)
            else: # analytical Integration
                Ke, Fe = self._perform_analytical_integration(x, B, J, e)
            
            # Assemble Stiffness Matrix and Force Vector
            for i in range(self._dof_element):
                for j in range(self._dof_element):
                    ii = self._location_matrix[i,e]
                    jj = self._location_matrix[j,e]
                    self._K[ii-1, jj-1] = self._K[ii-1, jj-1] + Ke[i,j]
                self._f_body[ii-1] = self._f_body[ii-1] + Fe[i]
        
        if (self._verbosity > 0):
            print("K:\n", self._K, "\n")
            print("f_body:\n", self._f_body, "\n")
        
    
    def _apply_boundary_conditions_and_solve_the_system(self):
        """
        Reduce the system of Equations assuming u[0]=0 and solving using numpy
        """
        print("\n... Applying Boundary Conditions and Solving the System ...")

        self.K_red = self._K[1:,1:]
        self.f_red = sp.zeros(self._num_dof - 1, 1)
        self.f_red[-1] = self._F
        
        for i in range(self._num_dof - 1):
            self.f_red[i] = self.f_red[i] + self._f_body[i+1]
        
        if (self._verbosity > 0):
            print("K_red:\n", self.K_red, "\n")
            print("f_red:\n", self.f_red, "\n")
            
        u_red = self.K_red.LUsolve(self.f_red)
        
        self.u = sp.zeros(1, 1).col_join(u_red) # Full Displacement Vector
                
        if (self._verbosity > 0):
            print("Displacement Vector u:\n", self.u, "\n")
            
            
    def _post_processing(self):
        """
        Necessary calculations to plot the displacement field
        """
        print("\n... Post-Processing ...")

        self.u_plot = [0] * self._num_elements
        self.x_plot = [0] * self._num_elements
        
        for i in range(self._num_elements):
            disp=0
            for j in range(self._dof_element):
                disp += self._N[j] * self.u[self._location_matrix[j,i]-1]
            
            if (self._verbosity > 0):
                print("u, Element", str(i+1), ":\n", disp, '\n')

            self.x_plot[i] = np.linspace(float(self._Le*i),float(self._Le*(i+1)),self._plot_accuracy)
                
            mapping = -1 + 2*(X-self._Le*(i))/self._Le

            disp_expr_tmp = sp.lambdify(xi, disp, "numpy")
            disp_expr = sp.lambdify(X, disp_expr_tmp(mapping), "numpy")
            
            self.u_plot[i] = disp_expr(self.x_plot[i])

        strain_energy = 0.5 * self.u.T * self._K * self.u
        strain_energy = sp.sqrt(strain_energy[0])
        strain_energy_exact = 0.06544088433080137260487994866374172246756572795350655893067784512107572226722403942992936169523014756802289277390
        self.error = (strain_energy_exact - strain_energy) / strain_energy_exact
        if (self._verbosity > 0):
            print("Strain Energy (Approx.)= ", strain_energy)
            print("Strain Energy ( Exact )= ", strain_energy_exact)
            print("Error in the Energy Norm = ", self.error * 100, "%")



    def _plot_results(self):
        """
        Plot the displacement field    
        """
        print("\n... Plotting Results")

        x_plot = np.linspace(0,self._L,self._plot_accuracy)
        u_exact_plot = np.array([  3.19189120e-16,  -2.92473198e-03,  -5.75846263e-03,
        -8.45459163e-03,  -1.09601761e-02,  -1.32210840e-02,
        -1.51861590e-02,  -1.68104849e-02,  -1.80578393e-02,
        -1.89024177e-02,  -1.93299083e-02,  -1.93379957e-02,
        -1.89363636e-02,  -1.81462695e-02,  -1.69997556e-02,
        -1.55385583e-02,  -1.38127781e-02,  -1.18793637e-02,
        -9.80046304e-03,  -7.64169191e-03,  -5.47036423e-03,
        -3.35372706e-03,  -1.35723904e-03,   4.57072353e-04,
         2.03214271e-03,   3.31726314e-03,   4.26930821e-03,
         4.85377755e-03,   5.04563293e-03,   4.82991581e-03,
         4.20213409e-03,   3.16840983e-03,   1.74538356e-03,
        -4.01261465e-05,  -2.15170549e-03,  -4.54416503e-03,
        -7.16458683e-03,  -9.95358679e-03,  -1.28467760e-02,
        -1.57764019e-02,  -1.86731464e-02,  -2.14680546e-02,
        -2.40945646e-02,  -2.64906051e-02,  -2.86007248e-02,
        -3.03782126e-02,  -3.17871669e-02,  -3.28044651e-02,
        -3.34215850e-02,  -3.36462234e-02])

        # Plot Displacements
        fig_disp = plt.figure()
        
        for i in range(self._num_elements):
            FE, = plt.plot(self.x_plot[i], self.u_plot[i], 'b', lw=1.5, marker='o', markevery=(self._plot_accuracy-1), markerfacecolor='None')
        FE.set_label('FE Solution')
        plt.legend
        plt.plot(x_plot, u_exact_plot, 'k--', lw=1.5, label='Exact Solution')
        plt.legend(loc='best')
        plt.title('Displacement (Polynomial Degree:'+str(self._polynomial_degree)+' Elements:'+str(self._num_elements)+')')
        plt.xlabel('X',fontsize=15)
        fig_disp.canvas.set_window_title('Displacement (Polynomial Degree:'+str(self._polynomial_degree)+' Elements:'+str(self._num_elements)+')')
        plt.grid(True)
        #plt.savefig('Displacement_p'+str(self._polynomial_degree)+'_e'+str(self._num_elements),format='jpeg', dpi=300)
        plt.show()
