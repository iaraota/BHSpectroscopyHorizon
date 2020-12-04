import numpy as np
from scipy import integrate
from . import PhysConst


def inner_product(frequency, h1, h2, Sh):
    """Compute gravitational wave noise-weigthed inner product

    Parameters
    ----------
        frequency: real array_like
            Sample frequencies corresponding to h1, h2 and Sh for integration
        h1: complex array_like
            first argument
        h2: complex array_like
            second argument
        Sh: real array_like
            Square root of noise Power Spectral Density [1/sqrt(Hz)]

    Returns
    -------
        scalar: Returns inner product between h1 and h2 weigthed by detector noise Sh.
    """
    frequency, h1, h2, Sh = np.asanyarray(frequency), np.asanyarray(h1), np.asanyarray(h2), np.asanyarray(Sh)
    return 4 * np.real(np.trapz((h1 * np.conj(h2)) / Sh**2, x = frequency))

def luminosity_distance(redshift):
    """ 
    Compute luminosity distance as function of the redshift

    Parameters
    ----------
        redshift: scalar
            Cosmological redshift value

    Returns
    -------
        scalar: Returns luminosity distance relative to given redshift
    """
    
    # cosmological constants
    # values from https://arxiv.org/pdf/1807.06209.pdf
    h = 0.6796
    H_0 = h*100*1e+3 # Huble constant m s**-1 Mpc**-1
    clight = 2.99792458e8 # speed of lightm s**-1
    Dist_H = clight/H_0 # Huble distance

    Omega_M = 0.315
    Omega_Λ = 1-Omega_M
    Omega_K = 0.0

    Ez = lambda z: 1/np.sqrt(Omega_M*(1+z)**3 + Omega_K*(1+z)**2 + Omega_Λ)
    Dist_C = Dist_H*integrate.quad(Ez, 0, redshift)[0]
    Dist_L = (1 + redshift)*Dist_C
    """#= If Omega_K was not 0
    if Omega_K > 0
        Dist_M = Dist_H*sinh(sqrt(Omega_K)*Dist_C/Dist_H)/sqrt(Omega_K)
    elif Omega_K == 0.0
        Dist_M = Dist_C
    elif Omega_K < 0
        Dist_M = Dist_H*np.sin(sqrt(Omega_K)*Dist_C/Dist_H)/sqrt(Omega_K)
    Dist_A = Dist_M/(1+redshift)
    Dist_L = (1+redshift)*Dist_M
    """
    return Dist_L

def convert_units(Mass_QNM, redshift, mass_f = 1):
    """ Compute factors that converts times and frequencies of the QNM
        according to the BH mass and redshift and waveform amplitude factor

    Parameters
    ----------
    Mass_QNM : scalar
        Final black hole mass in source frame
    redshift : scalar
        Redshift of the source
    mass_f : scalar, optional
        Factor of the final black hole mass relative to the total mass 
        of the binary (it is given by the BBH simulation), by default 1

    Returns
    -------
    array_like
        Returns time and amplitude conversion factors
    """
    # Source parameters
    M_final = (1+redshift)*Mass_QNM  # QNM mass in detector frame
    M_total = M_final / mass_f # Binary total mass (m1+m2)
    d_L = luminosity_distance(redshift)
    time_unit = (M_total)*PhysConst.UnitsToSeconds.tSun
    strain_unit = ((M_final)*PhysConst.UnitsToSeconds.tSun) / (d_L*PhysConst.UnitsToSeconds.Dist)

    return time_unit, strain_unit

def compute_qnm_time(t, A, phi, omega_r = None, omega_i = None, freq = None, tau = None, part = None):
    """ Compute quasinormal mode waveform

    Parameters
    ----------
    t : array_like
        time domain for the waveform to
    A : scalar
        Amplitude of the QNM
    phi : scalar
        Phase shift of the mode 
    omega_r : scalar, optional
        Real frequency of QNM in code units (solar mass), by default None
    omega_i : scalar, optional
        Imaginary frequency of QNM in code units, by default None
    freq : scalar, optional
        Frequency of QNM in Hertz, by default None
    tau : scalar, optional
        Decay time of QNM in seconds, by default None
    part: string, optional
        choose real or imaginary part, if it is set to None returns complex QNM

    Returns
    -------
    array_like
        Returns QNM array
    """

    t = np.asanyarray(t)
    waveform = np.empty(t.shape)
    if omega_r is not None and omega_i is not None:
        waveform = A*np.exp(-t*np.abs(omega_i))*np.exp(1j *(np.abs(omega_r)*t - phi))
    elif freq is not None and tau is not None:
        waveform = A*np.exp(-t/np.abs(tau))*np.exp(1j *(2*np.pi*np.abs(freq)*t - phi))
    else:
        raise ValueError("A value must be given for (omega_r and omega_i) OR (freq and tau)")

    if part == "real":
        return np.real(waveform)
    elif part == "imag":
        return np.imag(waveform)
    else: return waveform

def compute_qnm_fourier(f, A, phi, omega_r = None, omega_i = None, freq = None, tau = None, part = "real", convention = "FH"):
    """ Computes the Fourier transform of a single QNM.

    Parameters
    ----------
    f : array_like
        Frequency domain to compute the Fourier transform
    A : scalar
        Amplitude of the QNM 
    phi : scalar
        Phase of the QNM
    omega_r : scalar, optional
        Real part of the QNM frequency in code units, by default None
    omega_i : scalar, optional
        Imaginary part of the QNM frequency in code units, by default None
    freq : scalar, optional
        Frequency of the QNM in Hertz, by default None
    tau : scalar, optional
        Decay time of the QNM in seconds, by default None
    part : str, optional
        Select real or imaginary part of the QNM, by default "real"
    convention : str, optional
        Select convetion, FH reflects the waveform and EF cuts the waveform at time zero
        see https://arxiv.org/abs/gr-qc/0512160 for definition of the conventions. This
        affects high frequency content due to spectral leakage, by default "FH"

    Returns
    -------
    array_like
        Returns the fourier transform of the QNM

    """
    f = np.asanyarray(f)
    if freq is not None and tau is not None:
        omega_r = 2*np.pi*freq
        omega_i = 1/tau

    if convention == "EF":
        if part == "real":
            return A*((1j*2*np.pi*f + omega_i)*np.cos(phi) + omega_r*np.sin(phi)) / (omega_r**2 - (2*np.pi*f - 1j*omega_i)**2)
        elif part == "imag":
            return -A*(-(1j*2*np.pi*f + omega_i)*np.sin(phi) + omega_r*np.cos(phi)) / (-omega_r**2 + (2*np.pi*f - 1j*omega_i)**2)
        elif part == "psd":
            return A**2 / (omega_i**2 + (omega_r - 2*np.pi*f)**2) 
        else:
            raise ValueError("\"part\" argument of compute_qnm_fourier must be set to \"real\", \"imag\" or \"psd\".")

    elif convention == "FH":
        if part == "real":
            return A*omega_i*(np.exp(1j*phi)/(omega_i**2 + (-2*np.pi*f + omega_r)**2) + np.exp(-1j*phi)/(omega_i**2 + (2*np.pi*f + omega_r)**2))/2
        elif part == "imag":
            return 1j*A*omega_i*(np.exp(1j*phi)/(omega_i**2 + (-2*np.pi*f + omega_r)**2) - np.exp(-1j*phi)/(omega_i**2 + (2*np.pi*f + omega_r)**2))/2
        else:
            raise ValueError("\"part\" argument of compute_qnm_fourier must be set to \"real\" or \"imag\".")

    else:
        raise ValueError("convention argument must be set to \"FH\" or \"EF\".")

class QuasinormalMode:
    """ Compute QNM in time and frequency domain.
    """
    def __init__(self, amplitude, phase, omega_r, omega_i, mass = None, redshift = None, mass_f = 1):
        """Define QNM parameter.

        Parameters
        ----------
        amplitude : scalar
            amplitude of the QNM.
        phase : scalar
            Phase shift of the QNM.
        omega_r : scalar
            Real part of the QNM frequency in code units
        omega_i : scalar
            Imaginary part of the QNM frequency in code units, by default None
        mass : scalar, optional
            Mass of the BH that emmited the QNM, by default None
        redshift : scalar, optional
            Cosmological redshift of the BH, by default None
        mass_f : scalar, optional
            Final mass factor relative to the binary total mass (mass = mass_f*(m1 + m2)), by default 1
        """
        self.amplitude = amplitude
        self.phase = phase
        self.omega_r = omega_r 
        self.omega_i = omega_i

        self.frequency = None

        if mass is not None and redshift is not None:
            time_unit, strain_unit = convert_units(mass, redshift, mass_f)
            self.frequency = omega_r/2/np.pi/time_unit
            self.decay_time = time_unit/omega_i
            self.amplitude_scale = strain_unit
            self.time_convert = time_unit
    
    def qnm_time(self, t, part = None, units = "NR"):
        """ Computed QNM in time domain.

        Parameters
        ----------
        t : array_like
            Time array to compute QNM;
        part : string, optional
            Choose real or imaginary part of the QNM, by default None
        units : str, optional
            Choose "NR" for code units (M_odot) or "SI" (seconds), by default "NR"

        Returns
        -------
        array_like
            Returns QNM in time domain.

        Raises
        ------
        ValueError
            "SI" units can only be choosen if mass and redshift were given.
        """
        if units == "NR":
            return compute_qnm_time(t, self.amplitude, self.phase, omega_r = self.omega_r, omega_i = self.omega_i, part = part)
        elif units == "SI":
            if self.frequency is None:
                raise ValueError("Mass and redshift must be given!")
            else: 
                return self.amplitude_scale*compute_qnm_time(t, self.amplitude, self.phase, freq = self.frequency, tau = self.decay_time, part = part)
        else: raise ValueError("Units must be set to \"NR\" or \"SI\"!")
    def qnm_fourier(self, f, part = "real", convention = "FH", freqs_unit = "SI"):
        """Compute QNM in frequency domain.

        Parameters
        ----------
        f : array_like
            Frequency array to comput QNM.
        part : str, optional
            Choose "real" or "imaginary" part (polarization) of the QNM, by default "real"
        convention : str, optional
            Select convetion, FH reflects the waveform and EF cuts the waveform at time zero
            see https://arxiv.org/abs/gr-qc/0512160 for definition of the conventions. This 
            affects high frequency content due to spectral leakage, by default "FH"
        freqs_unit : str, optional
            Information about frequency array unit, set "NR" for code units (1/M_odot). This
            also defines the units of the returned QNM, by default "SI"

        Returns
        -------
        array_like
            Returns QNM in frequency domain.
        """
        f = np.asanyarray(f)
        if freqs_unit == "SI":
            if self.frequency is None:
                raise ValueError("Mass and redshift must be given!")
            else:
                return self.amplitude_scale*self.time_convert*compute_qnm_fourier(f*self.time_convert, self.amplitude, self.phase, omega_r = self.omega_r, omega_i = self.omega_i, part = part, convention = convention)
        elif freqs_unit == "NR":      
            return compute_qnm_fourier(f, self.amplitude, self.phase, omega_r = self.omega_r, omega_i = self.omega_i, part = part, convention = convention)

