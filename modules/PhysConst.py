"""
Physical constants in SI units and in seconds
"""
class Constants:
    # Gravitational constant
    Gconst = 6.67259e-11  # m^3/(kg*s^2)
    # speed of light
    clight = 2.99792458e8  # m/s
    # Solar mass
    MSun = 1.989e30  # kg
    # Parsec
    parsec = 3.08568025e16  # m

class UnitsToSeconds:
    """
    Convert desired unit to seconds
    """
    const = Constants()
    tSun = const.MSun*const.Gconst / const.clight**3  # from Solar mass to seconds
    Dist = 1.0e6*const.parsec / const.clight  # from Mpc to seconds
