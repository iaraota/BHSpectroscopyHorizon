import numpy as np

from SourceData import SourceData
from modules import GWFunctions, MCMCFunctions
from modules.PhysConst import UnitsToSeconds


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
            "df_dtau", "df_dtau_sub", "freq_tau", "omegas"}

        ratio : bool
            Choose true if model has amplitude ratios
            and False if model fits all amplitudes
        """

        models = {
            "kerr": {False: self.kerr_amplitude, True: self.kerr_ratio},
            "mass_spin": {False: self.mass_spin_amplitude, True: self.mass_spin_ratio},
            "df_dtau": {False: self.df_dtau_amplitude},
            "df_dtau_sub": {False: self.df_dtau_subdominant_amplitude},
            "freq_tau": {False: self.freq_tau_amplitude, True: self.freq_tau_ratio},
            "omegas": {False: self.omegas_amplitude, True: self.omegas_ratio},
            }

        try:
            self.model = models[model][ratio]

        except:
            if not isinstance(ratio, bool):
                raise ValueError("ratio should be set to True or False")
            else:
                raise ValueError('model should be {"kerr", "mass_spin", "df_dtau", "df_dtau_sub", "freq_tau", "omegas"}')

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

    def df_dtau_amplitude(self, theta:list):
        """QNM model with frequency and decay time as parameters.

        Parameters
        ----------
        theta : list
            [M, a] + [A, phi, df, dtau]*num_mode

        Returns
        -------
        function
            QNM model as a function of theta.
        """
        return self._model_function(theta, self._parameter_df_dtau)

    def df_dtau_subdominant_amplitude(self, theta:list):
        """QNM model with frequency and decay time as parameters.

        Parameters
        ----------
        theta : list
            [M, a] + [A, phi, df, dtau]*num_mode

        Returns
        -------
        function
            QNM model as a function of theta.
        """
        return self._model_function(theta, self._parameter_df_dtau_subdominant)

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
        """Compute QNM parameters (A, phi, freq, tau) given [M, a] + [A0, phi0] +
        [R, phi]*num_modes
        create self.theta_model list. M is the final mass in the detector frame.

        Parameters
        ----------
        theta : list
            injected parameters to the model [M,a] + [A0, phi0] + [R, phi]*num_modes
        """
        theta_model = []
        M, a = theta[:2]
        convert_freqs = M*UnitsToSeconds.tSun
        for i in range(len(self.modes_model)):
            R, phi = theta[2 + 2*i: 4 + 2*i]
            omega_r, omega_i = self.transform_mass_spin_to_omegas(
                1,
                a,
                self.df_a_omegas[self.modes_model[i]],
                )
            freq = omega_r/2/np.pi/convert_freqs
            tau = convert_freqs/omega_i
            theta_model.extend([R, phi, freq, tau])
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

    def _parameter_df_dtau_subdominant(
        self,
        theta:list,
        ):
        """Compute QNM parameters (A, phi, omega_r, omega_i) given [M, a] + [A, phi, dfreq, dtau]*num_modes
        create self.theta_model list

        Parameters
        ----------
        theta : list
            injected parameters to the model [M, a] + [A, phi, dfreq, dtau]*num_modes
        """        
        
        theta_model = []
        M, a = theta[:2]
        M = M/self.mass_initial
        delta_omega_r, delta_omega_i = {}, {}
        delta_omega_r[self.modes_model[0]] = 0
        delta_omega_i[self.modes_model[0]] = 0
        delta_omega_r[self.modes_model[1]] = theta[2]
        delta_omega_i[self.modes_model[1]] = theta[3]
        for i in range(len(self.modes_model)):
            A, phi = theta[4 + 2*i: 6 + 2*i]
            omega_r_GR, omega_i_GR = self.transform_mass_spin_to_omegas(
                M,
                a,
                self.df_a_omegas[self.modes_model[i]],
                )
            omega_r = omega_r_GR*(1 + delta_omega_r[self.modes_model[i]])
            omega_i = omega_i_GR*(1 + delta_omega_i[self.modes_model[i]])
            theta_model.extend([A, phi, omega_r, omega_i])
        return theta_model

    def _parameter_df_dtau(
        self,
        theta:list,
        ):
        """Compute QNM parameters (A, phi, omega_r, omega_i) given [M, a] + [A, phi, dfreq, dtau]*num_modes
        create self.theta_model list

        Parameters
        ----------
        theta : list
            injected parameters to the model [M, a] + [A, phi, dfreq, dtau]*num_modes
        """        
        
        theta_model = []
        M, a = theta[:2]
        M = M/self.mass_initial
        for i in range(len(self.modes_model)):
            A, phi, delta_omega_r, delta_omega_i = theta[2 + 4*i: 6 + 4*i]
            omega_r_GR, omega_i_GR = self.transform_mass_spin_to_omegas(
                M,
                a,
                self.df_a_omegas[self.modes_model[i]],
                )
            omega_r = omega_r_GR*(1 + delta_omega_r)
            omega_i = omega_i_GR*(1 + delta_omega_i)
            theta_model.extend([A, phi, omega_r, omega_i])
        return theta_model

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
        # theta_model should have the form [A0, phi0, freq0, tau0, A1, phi1, ...]

        A0, phi0, freq0, tau0 = theta_model[0:4]
        
        # Fitted A0 will be A0*[Solar mass in seconds]/[Gpc in seconds]
        amplitude = A0*UnitsToSeconds.tSun/(UnitsToSeconds.Dist*1e3)
        
        # amplitude ratio between first model and dominant = 1 
        h_model = self._h_model_qnm(1, phi0, freq0, tau0)

        for i in range(1,len(self.modes_model)):
            R, phi, freq, tau = theta_model[0 + 4*i: 4 + 4*i]
            h_model += self._h_model_qnm(R, phi, freq, tau)

        h_model *= amplitude
        return h_model

    def _h_model_qnm(self,
        R:float,
        phi:float,
        freq:float,
        tau:float,
        ):
        """Quasinormal mode model function.

        Parameters
        ----------
        R : float
            Amplitude ratio between mode and dominant mode.
        phi : float
            QNM phase.
        freq : float
            frequency of oscilation in Herz
        tau : float
            decay time in seconds.

        Returns
        -------
        array
            Returns QNM in frequency domain and SI units.
        """
        angular_mean = np.sqrt(1/5/4/np.pi)
        # angular_mean = 1

        h_real=  GWFunctions.compute_qnm_fourier(
            self.detector["freq"],
            R,
            phi,
            freq=freq,
            tau=tau,
            part = "real",
            convention = self.ft_convention
            )
        h_imag=  GWFunctions.compute_qnm_fourier(
            self.detector["freq"],
            R,
            phi,
            freq=freq,
            tau=tau,
            part = "imag",
            convention = self.ft_convention
            )
        
        return angular_mean*(h_real+ h_imag)


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
            "df_dtau", "df_dtau_sub", "freq_tau", "omegas"}

        ratio : bool
            Choose true if model has amplitude ratios
            and False if model fits all amplitudes
        """
        if not isinstance(ratio, bool):
            raise ValueError("ratio should be set to True or False")

        models = {
            "kerr": self._true_kerr,
            "mass_spin": self._true_mass_spin,
            "df_dtau": self._true_df_dtau,
            "df_dtau_sub": self._true_df_dtau_subdominant,
            "freq_tau": self._true_freq_tau,
            "omegas": self._true_omegas,
            }

        try:
            models[model](ratio)
        except:
            raise ValueError('model should be {"kerr", "mass_spin", "df_dtau", "df_dtau_sub", "freq_tau", "omegas"}')

    def _true_kerr(self, ratio:bool):
        self.theta_true = [self.final_mass*(1+self.redshift), self.final_spin]
        self.theta_labels = [r"$M_f(1+z)$", r"$a_f$"]
        
        for mode in self.modes_model:
            if mode == self.modes_model[0]:
                self.theta_true.extend([
                    (
                        self.qnm_modes[mode].amplitude*
                        self.final_mass*(1+self.redshift)/
                        self.dist_Gpc
                    ),
                    float(self.qnm_modes[mode].phase),
                    ])

                self.theta_labels.extend([
                    r"$A_{{{0}}}M_f(1+z)/D_L$".format(mode),
                    r"$\phi_{{{0}}}$".format(mode),
                    ])
            
            else:
                self.theta_true.extend([
                    self.qnm_modes[mode].amplitude/self.qnm_modes[self.modes_model[0]].amplitude,
                    float(self.qnm_modes[mode].phase),
                    ])

                self.theta_labels.extend([
                    r"$R_{{{0}}}$".format(mode),
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

    def _true_df_dtau(self, ratio:bool):
        self.theta_true = [self.final_mass, self.final_spin]
        self.theta_labels = [r"$M_f$", r"$a_f$"]
        
        for mode in self.modes_model:
            if ratio and (mode != self.modes_model[0]):
                self.theta_true.extend([
                    self.qnm_modes[mode].amplitude/self.qnm_modes[self.modes_model[0]].amplitude,
                    float(self.qnm_modes[mode].phase),
                    0.,
                    0.,
                    ])
                self.theta_labels.extend([
                    r"$R_{{{0}}}$".format(mode),
                    r"$\phi_{{{0}}}$".format(mode),
                    r"$\delta f_{{{0}}}/f_{{{0}}}$".format(mode),
                    r"$\delta \tau_{{{0}}}/\tau_{{{0}}}$".format(mode),
                    ])
            else:
                self.theta_true.extend([
                    self.qnm_modes[mode].amplitude,
                    float(self.qnm_modes[mode].phase),
                    0.,
                    0.,
                    ])

                self.theta_labels.extend([
                    r"$A_{{{0}}}$".format(mode),
                    r"$\phi_{{{0}}}$".format(mode),
                    r"$\delta f_{{{0}}}/f_{{{0}}}$".format(mode),
                    r"$\delta \tau_{{{0}}}/\tau_{{{0}}}$".format(mode),
                    ])

    def _true_df_dtau_subdominant(self, ratio:bool):
        self.theta_true = [self.final_mass, self.final_spin, 0, 0]
        self.theta_labels = [
            r"$M_f$", r"$a_f$",
            r"$\delta f_{{{0}}}/f_{{{0}}}$".format(self.modes_model[1]),
            r"$\delta \tau_{{{0}}}/\tau_{{{0}}}$".format(self.modes_model[1]),
            ]
        
        for mode in self.modes_model:
            if ratio and (mode != self.modes_model[0]):
                self.theta_true.extend([
                    self.qnm_modes[mode].amplitude/self.qnm_modes[self.modes_model[0]].amplitude,
                    float(self.qnm_modes[mode].phase),
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
            "df_dtau", "df_dtau_sub", "freq_tau", "omegas"}

        ratio : bool
            Choose true if model has amplitude ratios
            and False if model fits all amplitudes
        """
        if not isinstance(ratio, bool):
            raise ValueError("ratio should be set to True or False")

        models = {
            "kerr": self._prior_kerr,
            "mass_spin": self._prior_mass_spin,
            "df_dtau": self._prior_df_dtau,
            "df_dtau_sub": self._prior_df_dtau_subdominant,
            "freq_tau": self._prior_freq_tau,
            "omegas": self._prior_omegas,
            }

        try:
            models[model](ratio)
            self.prior_function = lambda theta: MCMCFunctions.noninfor_log_prior(theta, self.prior_min, self.prior_max)

        except:
            raise ValueError('model should be {"kerr", "mass_spin", "df_dtau", "df_dtau_sub", "freq_tau", "omegas"}')

    def cube_uniform_prior(
        self,
        model:str,
        ratio,
        ):
        """Generate uniform priors parameters. And transform the
        unit cube 'hypercube ~ Unif[0., 1.)' to real values priors
        for MultiNest sampling.

        Parameters
        ----------
        model : str
            QNM model. Can be set to {"kerr", "mass_spin",
            "df_dtau", "df_dtau_sub", "freq_tau", "omegas"}

        ratio : bool
            Choose true if model has amplitude ratios
            and False if model fits all amplitudes
        """

        if not isinstance(ratio, bool):
            raise ValueError("ratio should be set to True or False")

        models = {
            "kerr": self._prior_kerr,
            "mass_spin": self._prior_mass_spin,
            "df_dtau": self._prior_df_dtau,
            "df_dtau_sub": self._prior_df_dtau_subdominant,
            "freq_tau": self._prior_freq_tau,
            "omegas": self._prior_omegas,
            }

        try:
            models[model](ratio)
            self.prior_function = lambda hypercube: self._hypercube_transform(hypercube, self.prior_min, self.prior_max)

        except:
            raise ValueError('model should be {"kerr", "mass_spin", "df_dtau", "df_dtau_sub", "freq_tau", "omegas"}')

    def _hypercube_transform(
        self,
        hypercube,
        prior_min:list,
        prior_max:list,
        ):
        """Transfor prior to cube unit cube 'hypercube ~ Unif[0., 1.)' 
        to the parameter of interest for MultiNest sampling .

        Parameters
        ----------
        hypercube : array_like
            Unit cube to be transformed to real values priors.
        prior_min : list
            Minimum values in the prior.
        prior_max : list
            Maximum values in the prior.

        Returns
        -------
        array_like
            Returns transformed cube from Unif[0,1] to [min, max].
        """
        transform = lambda a, b, x: a + (b - a) * x

        cube = np.array(hypercube)
        for i in range(len(self.prior_min)):
            cube[i] = transform(self.prior_min[i], self.prior_max[i], cube[i])
        
        return cube

    def _prior_kerr(self, ratio:bool):
        self.prior_min = [1, 0]
        self.prior_max = [self.final_mass*(1+self.redshift)*10, 0.9999]

        for mode in self.modes_model:
            self.prior_min.extend([
                0,
                0,
            ])
            if mode == self.modes_model[0]:
                A_max = 10*(
                        self.final_mass*(1+self.redshift)/
                        self.dist_Gpc
                        )
            else:
                A_max = 10

            self.prior_max.extend([
                A_max,
                2*np.pi,
            ])

    def _prior_mass_spin(self, ratio:bool):
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

    def _prior_df_dtau(self, ratio:bool):
        self.prior_min = [1, 0]
        self.prior_max = [self.final_mass*10, 0.9999]

        for mode in self.modes_model:
            self.prior_min.extend([
                0,
                0,
                -1,
                -1,
            ])

            if ratio and (mode != self.modes_model[0]):
                A_max = 1
            else:
                A_max = 10

            self.prior_max.extend([
                A_max,
                2*np.pi,
                1,
                1,
            ])

    def _prior_df_dtau_subdominant(self, ratio:bool):
        self.prior_min = [1, 0, -1, -1]
        self.prior_max = [self.final_mass*10, 0.9999, 1, 1]

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

    def _prior_freq_tau(self, ratio:bool):
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

    def _prior_omegas(self, ratio:bool):
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
    modes = ["(2,2,0)"]
    teste = Models(modes, detector, m_f, z, q, "FH")
    print(teste._parameters_kerr_mass_spin([m_f, teste.final_spin, 1, 2, 3, 4]))

    wave = teste._model_function([m_f, teste.final_spin,0.4*(m_f*(1+z))/teste.dist_Gpc, 2], teste._parameters_kerr_mass_spin)
    print(wave)
    import matplotlib.pyplot as plt
    plt.loglog(teste.detector["freq"],teste.detector["psd"],)
    plt.loglog(teste.detector["freq"], np.abs(wave))
    plt.show()
    # teste.plot()
