import numpy as np
from modules import GWFunctions, MCMCFunctions, ImportData, PlotFunctions

class SourceData:
    """Generate waveforms, noise and injected data
    from choosen QNMs and detector in frequency domain.

    Parameters
    ----------
    detector : str
        Detector used to compute noise, can be set to
        "LIGO", "LISA", "ET", "CE = CE2silicon" and "CE2silica".
    modes : list
        Quasinormal modes considered.
    final_mass : float
        Final mass in the source frame.
    redshift : float
        SOurce redshift.
    q_mass : float
        Mass ratio of the binary.
    convention : str, optional
        Convention used to compute de Fourier transform of the QNMs,
        can be set do "FH" or "EF", by default "FH"
    """

    def __init__(self,
        detector:str,
        modes:list,
        final_mass:float,
        redshift:float,
        q_mass:float,
        convention:str="FH"):

        self.modes = tuple(modes)
        self.final_mass = final_mass
        self.redshift = redshift
        self.q_mass = q_mass
        self.ft_convention = convention

        # import detector strain
        self.detector = ImportData.import_detector(detector, False)

        # get QNM parameters from simulation
        self.qnm_pars, self.mass_f = ImportData.import_simulation_qnm_parameters(self.q_mass)

        # get convertion factor for time and amplitude
        self.time_convert, self.amplitude_scale = GWFunctions.convert_units(
            self.final_mass, self.redshift, self.mass_f)

        # compute QNMs waveforms
        self._compute_qnm_modes()

        # compute noise
        self._random_noise()

    def _compute_qnm_modes(self):
        """Comptute QNM waveforms in frequency domain.
        """
        qnm_modes = dict()
        for (k,v) in self.qnm_pars.items():
            qnm_modes[k] = GWFunctions.QuasinormalMode(v["amplitudes"], v["phases"], v["omega"][0], 
                            v["omega"][1], self.final_mass, self.redshift, self.mass_f)
            qnm_modes[k].qnm_f = {
                "real": qnm_modes[k].qnm_fourier(self.detector["freq"],
                        "real", self.ft_convention, "SI"),
                "imag": qnm_modes[k].qnm_fourier(self.detector["freq"],
                        "imag", self.ft_convention, "SI")
                }
            # qnm_modes[k].qnm_t = strain_unit*qnm_modes[k].qnm_time(times/time_unit, part, "NR")
        self.qnm_modes = qnm_modes

    def _random_noise(self):
        """Generate noise in frequency domain.
        """
        N_data = len(self.detector["psd"]) 
        # generate random noise from PSD
        self.noise = self.detector["psd"]*np.exp(1j*np.random.rand(N_data)*2*np.pi)
        # make noise an immutable array
        self.noise.flags.writeable = False

    def _inject_data(self, modes_data):
        """Generate data from noise and QNM waveform.
        """
        # d = n
        self.data = np.copy(self.noise)
        # d = n + modes
        for mode in modes_data:
            self.data += self.qnm_modes[mode].qnm_f["real"]



if __name__ == '__main__':

    m_f = 142
    z = 0.8
    q = 1.0
    detector = "LIGO"
    modes = ("(2,2,0)", "(2,2,1) I")
    teste = SourceData(detector, modes, m_f, z, q, "FH")