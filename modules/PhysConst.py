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
    # const = Constants()
    tSun = Constants.MSun*Constants.Gconst / Constants.clight**3  # from Solar mass to seconds
    Dist = 1.0e6*Constants.parsec / Constants.clight  # from Mpc to seconds


if __name__ == "__main__":
    # const = Constants()
    print(UnitsToSeconds.tSun)


