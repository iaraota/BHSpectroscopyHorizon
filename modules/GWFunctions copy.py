import numpy as np
import os
from scipy import interpolate, integrate
from . import PhysConst


def InnerProduct(frequency, h1, h2, Sh):
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

def LuminosityDistance(redshift):
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

def ImportDetectorStrain(detector, interpolation = False):
    """Import detector noise power specrtal density

    Parameters
    ----------
        detector: string 
            Detector name 'LIGO', 'LISA', 'CE' = 'CE2silicon', 'CE2silica' or 'ET'
        interpolation: bool
            Interpolate noise PSD

    Returns
    -------
        Dictionary: Returns detector frequency array relative to detector psd and detector label. If interpolation = true, also returns interpolated function.
    """
    # choose noise
    noise = {}
    i_freq = 0
    i_psd = 1
    if detector == "LIGO":
        file_name = "aLIGODesign.txt"
        noise["label"] = "LIGO - Design sensitivity"
    elif detector == "LISA":
        file_name = "LISA_Strain_Sensitivity_range.txt"
        noise["label"] = "LISA sensitivity"
    elif detector == "ET":
        i_psd = 3
        file_name = "ET/ETDSensitivityCurve.txt"
        noise["label"] = "ET_D sum sensitivity"
    elif detector == "CE" or detector == "CE2silicon":
        file_name = "CE/CE2silicon.txt"
        noise["label"] = "CE silicon sensitivity"
    elif detector == "CE2silica":
        file_name = "CE/CE2silica.txt"
        noise["label"] = "CE silica sensitivity"
    else:
        raise ValueError("Wrong detector option! Choose \"LIGO\", \"LISA\", \"CE\" = \"CE2silicon\", \"CE2silica\" or \"ET\"")
    
    file_path = os.path.join(os.getcwd(), "../detectors", file_name)
    noise_file = np.genfromtxt(file_path)
    noise["freq"], noise["psd"] = noise_file[:,i_freq], noise_file[:,i_psd]
    
    if interpolation == False:
        return noise
    else:
        itp = interpolate.interp1d(noise["freq"], noise["psd"], "cubic")
        return noise, itp

def CodeUnitsToSI(Mass_QNM, redshift, mass_f = 1):
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
    d_L = LuminosityDistance(redshift)
    time_unit = (M_total)*PhysConst.UnitsToSeconds.tSun
    strain_unit = ((M_final)*PhysConst.UnitsToSeconds.tSun) / (d_L*PhysConst.UnitsToSeconds.Dist)

    return time_unit, strain_unit

def QNM(t, A, phi, omega_r = None, omega_i = None, freq = None, tau = None, part = None):
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

def FourierQNM(f, A, phi, omega_r = None, omega_i = None, freq = None, tau = None, part = "real", convention = "FH"):
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
        see https://arxiv.org/abs/gr-qc/0512160 for definition of the conventions, by default "FH"

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
            raise ValueError("\"part\" argument of FourierQNM must be set to \"real\", \"imag\" or \"psd\".")

    elif convention == "FH":
        if part == "real":
            return A*omega_i*(np.exp(1j*phi)/(omega_i**2 + (-2*np.pi*f + omega_r)**2) + np.exp(-1j*phi)/(omega_i**2 + (2*np.pi*f + omega_r)**2))/2
        elif part == "imag":
            return 1j*A*omega_i*(np.exp(1j*phi)/(omega_i**2 + (-2*np.pi*f + omega_r)**2) - np.exp(-1j*phi)/(omega_i**2 + (2*np.pi*f + omega_r)**2))/2
        else:
            raise ValueError("\"part\" argument of FourierQNM must be set to \"real\" or \"imag\".")

    else:
        raise ValueError("convention argument must be set to \"FH\" or \"EF\".")
