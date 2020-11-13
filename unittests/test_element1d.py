from unittest import TestCase
import numpy as np

from FE_code.element_1d import Element1D
from FE_code.node import Node

pi = np.pi
sin = np.sin
cos = np.cos


class TestElement1D(TestCase):
    """
    Testing the CombinationSum class
    """

    def test_element1D_instance_is_created(self):

        nodes = [Node(1, 0, 0), Node(2, 0.05, 0)]
        dofs = np.array([0, 1])
        element_1d = Element1D(1, nodes, 1, dofs)

        self.assertEqual(element_1d.id, 1)
        self.assertEqual(element_1d.nodes[0].id, 1)
        self.assertEqual(element_1d.nodes[0]._x, 0)
        self.assertEqual(element_1d.nodes[0]._y, 0)
        self.assertEqual(element_1d.nodes[1].id, 2)
        self.assertEqual(element_1d.nodes[1]._x, 0.05)
        self.assertEqual(element_1d.nodes[1]._y, 0)

    def test_coords(self):

        nodes = [Node(1, 0, 0), Node(2, 0.05, 0)]
        dofs = np.array([0, 1])
        element_1d = Element1D(1, nodes, 1, dofs)

        self.assertEqual(element_1d.nodes[0].coordinates, [0, 0])
        self.assertEqual(element_1d.nodes[1].coordinates, [0.05, 0])

    def test_k_matrix(self):

        nodes = [Node(1, 0, 0), Node(2, 0.05, 0)]
        dofs = np.array([0, 1])
        element_1d = Element1D(1, nodes, 1, dofs)
        k_e = element_1d.k_matrix()

        self.assertEqual(k_e[0][0], 20.0)
        self.assertEqual(k_e[0][1], -20.0)
        self.assertEqual(k_e[1][0], -20.0)
        self.assertEqual(k_e[1][1], 20.0)

    def test_m_matrix(self):

        nodes = [Node(1, 0, 0), Node(2, 0.05, 0)]
        dofs = np.array([0, 1])
        element_1d = Element1D(1, nodes, 1, dofs)
        m_e = element_1d.m_matrix()

        self.assertAlmostEqual(m_e[0][0], 0.01666667)
        self.assertAlmostEqual(m_e[0][1], 0.00833333)
        self.assertAlmostEqual(m_e[1][0], 0.00833333)
        self.assertAlmostEqual(m_e[1][1], 0.01666667)

    def test_f_vector(self):

        nodes = [Node(1, 0, 0), Node(2, 0.05, 0)]
        dofs = np.array([0, 1])
        element_1d = Element1D(1, nodes, 1, dofs)

        def force_global_function(x, t):
            return 2*pi*cos(2*pi*t)*sin(2*pi*x) + 4*pi*pi*sin(2*pi*t)*sin(2*pi*x)

        f_e = element_1d.f_vector(force_global_function, 0)

        self.assertAlmostEqual(f_e[0], 0.01635941)
        self.assertAlmostEqual(f_e[1], 0.03258397)

    def test_integrate_uAnlytical_minus_uFEM_squared(self):

        nodes = [Node(1, 0, 0), Node(2, 0.05, 0)]
        dofs = np.array([0, 1])
        element_1d = Element1D(1, nodes, 1, dofs)
        local_u_FEM = [3.0, 0.01]

        def u_an(x, t):
            return sin(2*pi*x)*sin(2*pi*t)

        integral = element_1d.integrate_uAnlytical_minus_uFEM_squared(u_an, local_u_FEM, 1)

        self.assertAlmostEqual(integral, 240.80266666666665)
