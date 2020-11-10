
import numpy as np

def shape_functions_1d(xi, degree):
    """
    Calculate the hierarchic shape functions for 1 dimension

    Parameters
    ----------
    xi : float
        the local position to calculate the shape functions upon
    degree : int
        the degree of the polynomial of the shape functions

    Returns
    -------
    N : ndarray
        shape functions values
    """
    if degree != int(degree):
        raise RuntimeError('The degree must be an integer')

    N = list()
    N.append(1/2 * (1 - xi))
    N.append(1/2 * (1 + xi))
    if degree > 1:
        N.append(1/4 * np.sqrt(6) * (xi**2 - 1))
    if degree > 2:
        N.append(1/4 * np.sqrt(10) * (xi**2 - 1) * xi)
    if degree > 3:
        N.append(1/16 * np.sqrt(14) * (5*xi**4 - 6*xi**2 + 1))
    if degree > 4:
        N.append(3/16 * np.sqrt(2) * xi * (7*xi**4 - 10*xi**2 + 3))
    if degree > 5:
        N.append(1/32 * np.sqrt(22) * (21*xi**6 - 35*xi**4 + 15*xi**2 - 1))
    if degree > 6:
        N.append(1/32 * np.sqrt(26) * xi * (33*xi**6 - 63*xi**4 + 35*xi**2 - 5))
    if degree > 7:
        N.append(1/256 * np.sqrt(30) * (-140*xi**2 - 924*xi**6 + 630*xi**4 + 5 + 429*xi**8))
    if degree > 8:
        raise RuntimeError('Not yet implemented ;p you will need to wait for the next version :3')
    return np.array(N)


def shape_functions_derivatives_1d(xi, degree):
    """
    Calculate the hierarchic shape functions derivatives for 1 dimension

    Parameters
    ----------
    xi : float
        the local position to calculate the shape functions upon
    degree : int
        the degree of the polynomial of the shape functions

    Returns
    -------
    dN : ndarray
        shape functions derivatives values
    """
    if degree != int(degree):
        raise RuntimeError('The degree must be an integer')

    dN = list()
    dN.append(-1/2)
    dN.append(1/2)
    if degree > 1:
        dN.append(1/2 * np.sqrt(6) * xi)
    if degree > 2:
        dN.append(1/4 * np.sqrt(10) * (3*xi**2 - 1))
    if degree > 3:
        dN.append(1/16 * np.sqrt(14) * (20*xi**3 - 12*xi))
    if degree > 4:
        dN.append(3/16 * np.sqrt(2) * (35*xi**4 - 30*xi**2 + 3))
    if degree > 5:
        dN.append(1/32 * np.sqrt(22) * (126*xi**5 - 140*xi**3 + 30*xi))
    if degree > 6:
        dN.append(1/32 * np.sqrt(26) * (231*xi**6 - 315*xi**4 + 105*xi**2 - 5))
    if degree > 7:
        dN.append(1/256 * np.sqrt(30) * (-280*xi - 5544*xi**5 + 2520*xi**3 + 3432*xi**7))
    if degree > 8:
        raise RuntimeError('Not yet implemented ;p you will need to wait for the next version :3')
    return np.array(dN)


def shape_functions_derivatives_2d(xi, eta, degree):
    """
    Calculate the hierarchic shape functions for 2 dimensions

    Parameters
    ----------
    xi : float
        the local position in the first dimension to calculate the shape functions upon
    eta : float
        the local position in the second dimension to calculate the shape functions upon
    degree : int
        the degree of the polynomial of the shape functions

    Returns
    -------
    dN : ndarray
        shape functions derivatives values
    """
    if degree != int(degree):
        raise RuntimeError('The degree must be an integer')

    Nxi = shape_functions_1d(xi, degree)
    Neta = shape_functions_1d(eta, degree)
    dNdxi = shape_functions_derivatives_1d(xi, degree)
    dNdeta = shape_functions_derivatives_1d(eta, degree)
    dNdxi2D = np.outer(dNdxi, Neta)
    dNdeta2D = np.outer(Nxi, dNdeta)

    dNdxi2D = np.concatenate(
        (
            np.append(dNdxi2D[0:2, 0], dNdxi2D[0:2, 1][::-1]), # nodal
            dNdxi2D[2:, 0], # edge 1
            dNdxi2D[1, 2:], # edge 2
            dNdxi2D[2:, 1], # edge 3
            dNdxi2D[0, 2:], # edge 4
            dNdxi2D[2:, 2:].flatten() # internal
        )
    )

    dNdeta2D = np.concatenate(
        (
            np.append(dNdeta2D[0:2, 0], dNdeta2D[0:2, 1][::-1]), # nodal
            dNdeta2D[2:, 0], # edge 1
            dNdeta2D[1, 2:], # edge 2
            dNdeta2D[2:, 1], # edge 3
            dNdeta2D[0, 2:], # edge 4
            dNdeta2D[2:, 2:].flatten() # internal
        )
    )
    dN = np.array([dNdxi2D, dNdeta2D])
    return dN


def shape_functions_2d(xi, eta, degree):
    """
    Calculate the hierarchic shape functions for 2 dimensions

    Parameters
    ----------
    xi : float
        the local position in the first dimension to calculate the shape functions upon
    eta : float
        the local position in the second dimension to calculate the shape functions upon
    degree : int
        the degree of the polynomial of the shape functions

    Returns
    -------
    N : ndarray
        shape functions values
    """
    if degree != int(degree):
        raise RuntimeError('The degree must be an integer')

    Nxi = shape_functions_1d(xi, degree)
    Neta = shape_functions_1d(eta, degree)
    N2D = np.outer(Nxi, Neta)

    N = np.concatenate(
        (
            np.append(N2D[0:2, 0], N2D[0:2, 1][::-1]), # nodal
            N2D[2:, 0], # edge 1
            N2D[1, 2:], # edge 2
            N2D[2:, 1], # edge 3
            N2D[0, 2:], # edge 4
            N2D[2:, 2:].flatten() # internal
        )
    )

    return N
