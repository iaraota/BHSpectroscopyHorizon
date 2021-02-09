import numpy as np

from SourceData import SourceData
from modules import GWFunctions, MCMCFunctions

class Models(SourceData):
    """Generate waveform model function of QNMs."""
    
    def __init__(
        self,
        modes_model:str,
        *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.modes_model = modes_model

    def choose_model(self,
        model:str,
        ratio:bool,
        ):
        """Choose QNM model.

        Parameters
        ----------
        model : str
            QNM model. Can be set to {"kerr", "mass_spin",
            "freq_tau", "omegas"}

        ratio : bool
            Choose true if model has amplitude ratios
            and False if model fits all amplitudes
        """

        models = {
            "kerr": {False: self.kerr_amplitude, True: self.kerr_ratio},
            "mass_spin": {False: self.mass_spin_amplitude, True: self.mass_spin_ratio},
            "freq_tau": {False: self.freq_tau_amplitude, True: self.freq_tau_ratio},
            "omegas": {False: self.omegas_amplitude, True: self.omegas_ratio},
            }

        try:
            self.model = models[model][ratio]

        except:
            if not isinstance(ratio, bool):
                raise ValueError("ratio should be set to True or False")
            else:
                raise ValueError('model shoul be {"kerr", "mass_spin", "freq_tau", "omegas"}')


    def kerr_amplitude(self, theta:list):
        """QNM model with mass and spin as parameters.
        Assuming the no-hair theorem, frequencies of
        all modes are computed from the same mass and
        spin.

        Parameters
        ----------
        theta : list
            [M,a] + [A, phi]*num_mode

        Returns
        -------
        function
            QNM model as a function of theta.
        """
        return self._model_function(theta, self._parameters_kerr_mass_spin)

    def kerr_ratio(self, theta:list):
        """QNM model with mass and spin as parameters.
        Assuming the no-hair theorem, frequencies of
        all modes are computed from the same mass and
        spin.

        Parameters
        ----------
        theta : list
            [M,a] + [R, phi]*num_mode
            R = A_i/A_0 and A_0 for i = 0

        Returns
        -------
        function
            QNM model as a function of theta.
        """
        theta = self._convert_ratio(theta)
        return self._model_function(theta, self._parameters_kerr_mass_spin)

    def mass_spin_amplitude(self, theta:list):
        """QNM model with mass and spin as parameters.
        It is not assuming the no-hair theorem, has a
        mass and a spin for each mode.

        Parameters
        ----------
        theta : list
            [A, phi, M, a]*num_mode

        Returns
        -------
        function
            QNM model as a function of theta.
        """
        return self._model_function(theta, self._parameters_mass_spin)

    def mass_spin_ratio(self, theta:list):
        """QNM model with mass and spin as parameters.
        It is not assuming the no-hair theorem, has a
        mass and a spin for each mode.

        Parameters
        ----------
        theta : list
            [R, phi, M, a]*num_mode
            R = A_i/A_0 and A_0 for i = 0

        Returns
        -------
        function
            QNM model as a function of theta.
        """
        theta = self._convert_ratio(theta)
        return self._model_function(theta, self._parameters_mass_spin)

    def freq_tau_amplitude(self, theta:list):
        """QNM model with frequency and decay time as parameters.

        Parameters
        ----------
        theta : list
            [A, phi, f, tau]*num_mode

        Returns
        -------
        function
            QNM model as a function of theta.
        """
        return self._model_function(theta, self._parameter_freq_tau)

    def freq_tau_ratio(self, theta:list):
        """QNM model with frequency and decay time as parameters.

        Parameters
        ----------
        theta : list
            [R, phi, f, tau]*num_mode
            R = A_i/A_0 and A_0 for i = 0

        Returns
        -------
        function
            QNM model as a function of theta.
        """
        theta = self._convert_ratio(theta)
        return self._model_function(theta, self._parameter_freq_tau)

    def omegas_amplitude(self, theta:list):
        """QNM model with numerical units (time in untis of initial mass) 
        frequencies as parameters.

        Parameters
        ----------
        theta : list
            [A, phi, omega_r, omega_i]*num_mode

        Returns
        -------
        function
            QNM model as a function of theta.
        """
        return self._model_function(theta, self._parameters_omegas)

    def omegas_ratio(self, theta:list):
        """QNM model with numerical units (time in untis of initial mass) 
        frequencies as parameters.

        Parameters
        ----------
        theta : list
            [R, phi, omega_r, omega_i]*num_mode
            R = A_i/A_0 and A_0 for i = 0

        Returns
        -------
        function
            QNM model as a function of theta.
        """
        theta = self._convert_ratio(theta)
        return self._model_function(theta, self._parameters_omegas)

    def _parameters_omegas(
        self,
        theta:list,
        ):
        return theta

    def _parameters_kerr_mass_spin(
        self,
        theta:list,
        ):
        """Compute QNM parameters (A, phi, omega_r, omega_i) given [M, a] + [A, phi]*num_modes
        create self.theta_model list

        Parameters
        ----------
        theta : list
            injected parameters to the model [M,a] + [A, phi]*num_modes
        """
        theta_model = []
        M, a = theta[:2]
        M = M/self.mass_initial
        for i in range(len(self.modes_model)):
            A, phi = theta[2 + 2*i: 4 + 2*i]
            omega_r, omega_i = self.transform_mass_spin_to_omegas(
                M,
                a,
                self.df_a_omegas[self.modes_model[i]],
                )
            theta_model.extend([A, phi, omega_r, omega_i])
        return theta_model

    def _parameters_mass_spin(
        self,
        theta:list,
        ):
        """Compute QNM parameters (A, phi, omega_r, omega_i) given [A, phi, M, A]*num_modes
        create self.theta_model list

        Parameters
        ----------
        theta : list
            injected parameters to the model [A, phi, M, A]*num_modes
        """
        theta_model = []
        
        for i in range(len(self.modes_model)):
            A, phi, M, a = theta[0 + 4*i: 4 + 4*i]
            M = M/self.mass_initial
            omega_r, omega_i = self.transform_mass_spin_to_omegas(
                M,
                a,
                self.df_a_omegas[self.modes_model[i]],
                )
            theta_model.extend([A, phi, omega_r, omega_i])
        return theta_model

    def _parameter_freq_tau(
        self,
        theta:list,
        ):
        """Compute QNM parameters (A, phi, omega_r, omega_i) given [A, phi, freq, tau]*num_modes
        create self.theta_model list

        Parameters
        ----------
        theta : list
            injected parameters to the model [A, phi, freq, tau]*num_modes
        """        
        
        theta_model = []
        for i in range(len(self.modes_model)):
            A, phi, freq, tau = theta[0 + 4*i: 4 + 4*i]
            tau *= 1e-3
            omega_r = freq*2*np.pi*self.time_convert
            omega_i = self.time_convert/tau

            theta_model.extend([A, phi, omega_r, omega_i])   

        return theta_model

    def _convert_ratio(
        self,
        theta:list,
        ):
        """Transform amplitude of sub-dominant modes in amplitude
        ratio A_i/A_0

        Parameters
        ----------
        theta : list
            QNM model parameters
        parameter_function : function
            Function that converts model parameters to
            (A, phi, omega_r, omega_i)*num_modes

        Returns
        -------
        list
            Model parameters with amplitude ratios.
        """

        for i in range(1,len(self.modes_model)):
            theta[4*i] *= theta[0]

        return theta

    def _model_function(self,
        theta:list,
        parameter_function,
        ):
        """Generate waveform model function of QNMs.

        Parameters
        ----------
        theta : array_like
            Model parameters.
        parameter_function : function
            Function that converts model parameters to
            (A, phi, omega_r, omega_i)*num_modes

        Returns
        -------
        function
            Waveform model as a function of parameters theta.
        """
        h_model = 0
        theta_model = parameter_function(theta)
        for i in range(len(self.modes_model)):
            A, phi, omega_r, omega_i = theta_model[0 + 4*i: 4 + 4*i]
            h_model += self._h_model_qnm(A, phi, omega_r, omega_i)

        return h_model

    def _h_model_qnm(self,
        A:float,
        phi:float,
        omega_r:float,
        omega_i:float,
        ):
        """Quasinormal mode model function.

        Parameters
        ----------
        A : float
            Amplitude
        phi : float
            phase
        omega_r : float
            real frequency in code units (initial mass units)
        omega_i : float
            imaginary frequency in code units

        Returns
        -------
        array
            Returns QNM in frequency domain and SI units.
        """
        return self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
                self.detector["freq"]*self.time_convert, A, phi, omega_r, omega_i, 
                part = "real", convention = self.ft_convention)


class TrueParameters(SourceData):
    """Generate true parameters of injected QNMs."""
    
    def __init__(
        self,
        modes_model:str,
        *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.modes_model = modes_model

    def choose_theta_true(
        self,
        model:str,
        ratio,
        ):
        """Generate a list of the true injected parameters.

        Parameters
        ----------
        model : str
            QNM model. Can be set to {"kerr", "mass_spin",
            "freq_tau", "omegas"}

        ratio : bool
            Choose true if model has amplitude ratios
            and False if model fits all amplitudes
        """
        if not isinstance(ratio, bool):
            raise ValueError("ratio should be set to True or False")

        models = {
            "kerr": self._true_kerr,
            "mass_spin": self._true_mass_spin,
            "freq_tau": self._true_freq_tau,
            "omegas": self._true_omegas,
            }

        try:
            models[model](ratio)
        except:
            raise ValueError('model shoul be {"kerr", "mass_spin", "freq_tau", "omegas"}')

    def _true_kerr(self, ratio:bool):
        self.theta_true = [self.final_mass, self.final_spin]
        self.theta_labels = [r"$M_f [M_\odot]$", r"$a_f [M_f]$"]
        
        for mode in self.modes_model:
            if ratio and (mode != self.modes_model[0]):
                self.theta_true.extend([
                    self.qnm_modes[mode].amplitude/self.qnm_modes[self.modes_model[0]].amplitude,
                    float(self.qnm_modes[mode].phase)
                    ])
                self.theta_labels.extend([
                    r"$R_{{{0}}}$".format(mode),
                    r"$\phi_{{{0}}}$".format(mode),
                    ])
            else:
                self.theta_true.extend([
                    self.qnm_modes[mode].amplitude,
                    float(self.qnm_modes[mode].phase),
                    ])

                self.theta_labels.extend([
                    r"$A_{{{0}}}$".format(mode),
                    r"$\phi_{{{0}}}$".format(mode),
                    ])

    def _true_mass_spin(self, ratio:bool):
        self.theta_true = []
        self.theta_labels = []
        
        for mode in self.modes_model:
            if ratio and (mode != self.modes_model[0]):
                self.theta_true.extend([
                    self.qnm_modes[mode].amplitude/self.qnm_modes[self.modes_model[0]].amplitude,
                    float(self.qnm_modes[mode].phase),
                    self.final_mass,
                    self.final_spin,
                    ])
                self.theta_labels.extend([
                    r"$R_{{{0}}}$".format(mode),
                    r"$\phi_{{{0}}}$".format(mode),
                    r"$M_{{{0}}}$".format(mode),
                    r"$a_{{{0}}}$".format(mode),
                    ])
            else:
                self.theta_true.extend([
                    self.qnm_modes[mode].amplitude,
                    float(self.qnm_modes[mode].phase),
                    self.final_mass,
                    self.final_spin,
                    ])

                self.theta_labels.extend([
                    r"$A_{{{0}}}$".format(mode),
                    r"$\phi_{{{0}}}$".format(mode),
                    r"$M_{{{0}}}$".format(mode),
                    r"$a_{{{0}}}$".format(mode),
                    ])

    def _true_freq_tau(self, ratio:bool):
        self.theta_true = []
        self.theta_labels = []
        
        for mode in self.modes_model:
            if ratio and (mode != self.modes_model[0]):
                self.theta_true.extend([
                    self.qnm_modes[mode].amplitude/self.qnm_modes[self.modes_model[0]].amplitude,
                    float(self.qnm_modes[mode].phase),
                    self.qnm_modes[mode].frequency,
                    self.qnm_modes[mode].decay_time*1e3,
                    ])
                self.theta_labels.extend([
                    r"$R_{{{0}}}$".format(mode),
                    r"$\phi_{{{0}}}$".format(mode),
                    r"$f_{{{0}}} [Hz]$".format(mode),
                    r"$\tau_{{{0}}} [ms]$".format(mode),
                    ])
            else:
                self.theta_true.extend([
                    self.qnm_modes[mode].amplitude,
                    float(self.qnm_modes[mode].phase),
                    self.qnm_modes[mode].frequency,
                    self.qnm_modes[mode].decay_time*1e3,
                    ])

                self.theta_labels.extend([
                    r"$A_{{{0}}}$".format(mode),
                    r"$\phi_{{{0}}}$".format(mode),
                    r"$f_{{{0}}} [Hz]$".format(mode),
                    r"$\tau_{{{0}}} [ms]$".format(mode),
                    ])

    def _true_omegas(self, ratio:bool):
        self.theta_true = []
        self.theta_labels = []
        
        for mode in self.modes_model:
            if ratio and (mode != self.modes_model[0]):
                self.theta_true.extend([
                    self.qnm_modes[mode].amplitude/self.qnm_modes[self.modes_model[0]].amplitude,
                    float(self.qnm_modes[mode].phase),
                    self.qnm_modes[mode].omega_r,
                    self.qnm_modes[mode].omega_i,
                    ])
                self.theta_labels.extend([
                    r"$R_{{{0}}}$".format(mode),
                    r"$\phi_{{{0}}}$".format(mode),
                    r"$\omega^r_{{{0}}}$".format(mode),
                    r"$\omega^i_{{{0}}}$".format(mode),
                    ])
            else:
                self.theta_true.extend([
                    self.qnm_modes[mode].amplitude,
                    float(self.qnm_modes[mode].phase),
                    self.qnm_modes[mode].omega_r,
                    self.qnm_modes[mode].omega_i,
                    ])

                self.theta_labels.extend([
                    r"$A_{{{0}}}$".format(mode),
                    r"$\phi_{{{0}}}$".format(mode),
                    r"$\omega^r_{{{0}}}$".format(mode),
                    r"$\omega^i_{{{0}}}$".format(mode),
                    ])


class Priors(SourceData):
    """Generate priors for QNMs waveform models."""
    
    def __init__(
        self,
        modes_model:str,
        *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.modes_model = modes_model

    def uniform_prior(
        self,
        model:str,
        ratio,
        ):
        """Generate uniform priors parameters.

        Parameters
        ----------
        model : str
            QNM model. Can be set to {"kerr", "mass_spin",
            "freq_tau", "omegas"}

        ratio : bool
            Choose true if model has amplitude ratios
            and False if model fits all amplitudes
        """
        if not isinstance(ratio, bool):
            raise ValueError("ratio should be set to True or False")

        models = {
            "kerr": self._true_kerr,
            "mass_spin": self._true_mass_spin,
            "freq_tau": self._true_freq_tau,
            "omegas": self._true_omegas,
            }

        try:
            models[model](ratio)
            self.prior_function = lambda theta: MCMCFunctions.noninfor_log_prior(theta, self.prior_min, self.prior_max)

        except:
            raise ValueError('model shoul be {"kerr", "mass_spin", "freq_tau", "omegas"}')

    def _true_kerr(self, ratio:bool):
        self.prior_min = [1, 0]
        self.prior_max = [self.final_mass*10, 0.9999]

        for mode in self.modes_model:
            self.prior_min.extend([
                0,
                0,
            ])
            if ratio and (mode != self.modes_model[0]):
                A_max = 1
            else:
                A_max = 10

            self.prior_max.extend([
                A_max,
                2*np.pi,
            ])

    def _true_mass_spin(self, ratio:bool):
        self.prior_min = []
        self.prior_max = []

        for mode in self.modes_model:
            self.prior_min.extend([
                0,
                0,
                1,
                0,
            ])

            if ratio and (mode != self.modes_model[0]):
                A_max = 1
            else:
                A_max = 10

            self.prior_max.extend([
                A_max,
                2*np.pi,
                self.final_mass*10,
                0.9999,
            ])
        
    def _true_freq_tau(self, ratio:bool):
        self.prior_min = []
        self.prior_max = []

        for mode in self.modes_model:
            self.prior_min.extend([
                0,
                0,
                self.qnm_modes[mode].frequency/10,
                self.qnm_modes[mode].decay_time*1e3/10,
            ])

            if ratio and (mode != self.modes_model[0]):
                A_max = 1
            else:
                A_max = 10

            self.prior_max.extend([
                A_max,
                2*np.pi,
                self.qnm_modes[mode].frequency*10,
                self.qnm_modes[mode].decay_time*1e3*10,
            ])

    def _true_omegas(self, ratio:bool):
        self.prior_min = []
        self.prior_max = []

        for mode in self.modes_model:
            self.prior_min.extend([
                0,
                0,
                self.qnm_modes[mode].omega_r/10,
                self.qnm_modes[mode].omega_i/10,
            ])
            if ratio and (mode != self.modes_model[0]):
                A_max = 1
            else:
                A_max = 10

            self.prior_max.extend([
                A_max,
                2*np.pi,
                self.qnm_modes[mode].omega_r*10,
                self.qnm_modes[mode].omega_i*10,
            ])

if __name__ == '__main__':
    np.random.seed(1234)
    m_f = 500
    z = 0.1
    q = 1.5

    detector = "LIGO"
    modes = ["(2,2,0)", "(2,2,1) I"]
    teste = Models(modes, detector, m_f, z, q, "FH")
    print(teste._parameters_kerr_mass_spin([m_f, teste.final_spin, 1, 2, 3, 4]))

    # import matplotlib.pyplot as plt
    # plt.loglog(teste.freq_tau_ratio([1, 1, 0.5, 0.1, 0.5, 1, 0.6, 0.7]))
    # plt.show()
    # teste.plot()
