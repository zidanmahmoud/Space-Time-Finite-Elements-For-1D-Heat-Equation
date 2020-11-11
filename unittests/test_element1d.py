from unittest import TestCase
from FE_code.element_1d import Element1D
from FE_code.node import Node

import numpy as np


class TestCombinationSum(TestCase):
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
