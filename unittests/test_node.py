from unittest import TestCase
import numpy as np

from FE_code.node import Node

pi = np.pi
sin = np.sin
cos = np.cos


class TestNode(TestCase):
    """
    Testing the Node class
    """

    def test_coords(self):

        nodes = Node(1, 2, 30)
        coord = nodes.coordinates

        self.assertEqual(coord, [2, 30])
