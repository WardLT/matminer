"""Utilities related to computing spherical-harmonic-based features"""

from sympy.physics.quantum.cg import clebsch_gordan, cg_simp
from sympy.physics.wigner import wigner_3j
from scipy.special import lpmv, factorial
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=32)
def get_wigner_coeffs(l):
    """Get the list of non-zero Wigner 3j triplets

    Args:
        l (int): Desired l
    Returns:
        List of tuples that contain:
            - ((int)) m coordinates of the triplet
            - (float) Wigner coefficient
    """

    return [((m1, m2, m3), float(wigner_3j(l, l, l, m1, m2, m3)))
            for m1, m2, m3 in _iterate_wigner_3j(l)]


def _iterate_wigner_3j(l):
    """Iterator over all non-zero Wigner 3j triplets

    Args:
        l (int) - Desired l
    Generates:
        pairs of acceptable l's
    """

    for m1 in range(-l, l+1):
        for m2 in range(-l, l+1):
            m3 = -1 * (m1 + m2)
            if -l <= m3 <= l:
                yield m1, m2, m3


def compute_hyperspherical_harmonic(j, m, mp, theta_0, theta, phi):
    """Compute 4D  hyperspherical harmonics

    Args:
        j (float): First coefficient of harmonic
        m (int): Second coefficient
        mp (int): Third component
        theta_0 (ndarray): Theta_0 (distance therm in SNAP)
        theta (ndarray): Polar angles
        phi (ndarray): Azimuthal angles
    Return:
         (ndarray) Hyperspherical harmonics
    """

    return np.exp(np.multiply(j * np.i, theta_0)) / np.sqrt(2 * np.pi) \
           * _hyper_sphere_legendre(2, -j, m, theta) * _hyper_sphere_legendre(3, -m, mp, phi)


def _hyper_sphere_legendre(j, l, L, theta):
    """Compute the Legendre portion of the general spherical harmonic

    Args:
        j (float): Dimension of spherical harmonic
        l (int): Component 1 of Legendre term
        L (int): Component 2 of Legendre term
        theta ([float]): Angles for this dimension
    Returns:
         [float]: Evaluating the function of each theta
    """

    jp = (2 - j) / 2
    return np.sqrt((2 * L + j - 1) / 2 * factorial(L + l + j - 2) / factorial(L - l)) \
        * np.power(np.sin(theta), jp) * lpmv(-l - jp, L + jp, np.cos(theta))


def compute_h_coupling_coefficients(j, m, mp):
    """Compute the transfer coefficient (H) for the bispectrum

    Args:
        j (tuple[int]): j values (3)
        m (tuple[int]): m values (3)
        mp (tuple[int]): m' values (3)
    Returns:
        Clebsch-Gordan coupling coefficients
    """

    return float(cg_simp(clebsch_gordan(j[0], j[1], j[2], m[0], m[1], m[2]) *
                         clebsch_gordan(j[0], j[1], j[2], mp[0], mp[1], mp[2])))
