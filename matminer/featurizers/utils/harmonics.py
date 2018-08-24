"""Utilities related to computing spherical-harmonic-based features"""

from sympy.physics.wigner import wigner_3j
from functools import lru_cache


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
            if -l <= m3 <= 1:
                yield m1, m2, m3
