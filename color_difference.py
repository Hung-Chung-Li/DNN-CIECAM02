import numpy as np
from colour.utilities import tsplit

def deltaE_94(Lab_1, Lab_2, textiles=False):
    """
    Returns the difference :math:`\Delta E_{ab}` between two given
    *CIE L\*a\*b\** colourspace arrays using *CIE 1994* recommendation.

    Parameters
    ----------
    Lab_1 : array_like
        *CIE L\*a\*b\** colourspace array 1.
    Lab_2 : array_like
        *CIE L\*a\*b\** colourspace array 2.
    textiles : bool, optional
        Textiles application specific parametric factors
        :math:`k_L=2,\ k_C=k_H=1,\ k_1=0.048,\ k_2=0.014` weights are used
        instead of :math:`k_L=k_C=k_H=1,\ k_1=0.045,\ k_2=0.015`.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\Delta E_{ab}`.

    Notes
    -----
    -   *CIE 1994* colour differences are not symmetrical: difference between
        ``Lab_1`` and ``Lab_2`` may not be the same as difference between
        ``Lab_2`` and ``Lab_1`` thus one colour must be understood to be the
        reference against which a sample colour is compared.

    References
    ----------
    -   :cite:`Lindbloom2011a`

    Examples
    --------
    >>> Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
    >>> Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
    >>> delta_E_CIE1994(Lab_1, Lab_2)  # doctest: +ELLIPSIS
    83.7792255...
    >>> delta_E_CIE1994(Lab_1, Lab_2, textiles=True)  # doctest: +ELLIPSIS
    88.3355530...
    """

    k_1 = 0.048 if textiles else 0.045
    k_2 = 0.014 if textiles else 0.015
    k_L = 2 if textiles else 1
    k_C = 1
    k_H = 1

    L_1, a_1, b_1 = tsplit(Lab_1)
    L_2, a_2, b_2 = tsplit(Lab_2)

    C_1 = np.hypot(a_1, b_1)
    C_2 = np.hypot(a_2, b_2)

    s_L = 1
    s_C = 1 + k_1 * C_1
    s_H = 1 + k_2 * C_1

    delta_L = L_1 - L_2
    delta_C = C_1 - C_2
    delta_A = a_1 - a_2
    delta_B = b_1 - b_2

    delta_H = np.real(np.sqrt(delta_A ** 2 + delta_B ** 2 - delta_C ** 2, dtype = complex))

    L = (delta_L / (k_L * s_L)) ** 2
    C = (delta_C / (k_C * s_C)) ** 2
    H = (delta_H / (k_H * s_H)) ** 2

    d_E = np.sqrt(L + C + H)

    return d_E