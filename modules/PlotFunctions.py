import numpy as np

def scientific_format(num, precision = 1, min_pow = 0):
    """Converts float to LaTeX scientific notation.

    Parameters
    ----------
    num : float
        Number to be converted.
    precision : int, optional
        Number of decimals, by default 1
    min_pow : int, optional
        Only converts numbers with power greater than min_pow, by default 0
    Returns
    -------
    string
        Returns number in LaTeX scientific notation.
    """
    power = np.floor(np.log10(num))
    factor = round(num*10**(-power), precision)

    if np.abs(power) < min_pow:
        return "{}".format(num)
    else: return "{0} \\times 10^{{{1}}}".format(factor, int(power))



if __name__ == "__main__":
    # testing functions
    a = scientific_format(0.5, 3, 2)
    print(a)