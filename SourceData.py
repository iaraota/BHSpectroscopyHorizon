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
        final_mass:float,
        redshift:float,
        q_mass:float,
        convention:str="FH"):

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

    def inject_data(self, modes_data):
        """Generate data from noise and QNM waveform.
        """
        # d = n
        self.data = np.copy(self.noise)
        # d = n + modes
        for mode in modes_data:
            self.data += self.qnm_modes[mode].qnm_f["real"]

    def __str__(self):
        return ('Create QNMs for a binary with:\n\t' 
            +f'mass ratio {self.q_mass},\n\t'
            +f'final mass {self.final_mass},\n\t'
            +f'redshift {self.redshift}\n\t'
            +f'and {self.detector["label"]} detector.\n\n'
            +'The method inject_data(modes_data) injects the selected QNMs in the detector noise.'
            )

    def __repr__(self):
        return (f'Create QNMs for a binary with:\n\t' 
            + 'mass ratio {self.q_mass},\n\t'
            +'final mass {self.final_mass},\n\t'
            +'redshift {self.redshift}\n\t'
            +'and {self.detector["label"]} detector.\n\n'
            +'The method inject_data(modes_data) injects the selected QNMs in the detector noise.'
            )



if __name__ == '__main__':

    m_f = 142
    z = 0.8
    q = 1.0
    detector = "LIGO"
    teste = SourceData(detector, m_f, z, q, "FH")
    print(teste)