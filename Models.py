import numpy as np

from SourceData import SourceData
from modules import GWFunctions, MCMCFunctions
from modules.PhysConst import UnitsToSeconds


class Models(SourceData):
    """Generate waveform model function of QNMs."""

    def __init__(
        self,
        modes_model: str,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.modes_model = modes_model

    def choose_model(self,
                     model: str,
                     ):
        """Choose QNM model.

        Parameters
        ----------
        model : str
            QNM model. Can be set to {"freq_tau", "kerr",
            "mass_spin", "df_dtau", "df_dtau_sub"}
        """

        models = {
            "freq_tau": self.freq_tau_model,
            "freq_tau_multi": self.freq_tau_model,
            "kerr": self.kerr_model,
            "mass_spin": self.mass_spin_model,
            "df_dtau": self.df_dtau_model,
            "df_dtau_sub": self.df_dtau_subdominant_model,
        }

        try:
            self.model = models[model]

        except:
            raise ValueError(
                'model should be {"freq_tau", "kerr", "mass_spin", "df_dtau", "df_dtau_sub"}')

    def freq_tau_model(self, theta: list):
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
        return self._model_function(theta, self._parameters_freq_tau)

    def kerr_model(self, theta: list):
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

    def mass_spin_model(self, theta: list):
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

    def df_dtau_model(self, theta: list):
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

    def df_dtau_subdominant_model(self, theta: list):
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

    def _parameters_freq_tau(
        self,
        theta: list,
    ):
        """Compute QNM parameters (A, phi, freq, tau) given [A, phi, freq, tau]*num_modes
        create self.theta_model list

        Parameters
        ----------
        theta : list
            injected parameters to the model [A, phi, freq, tau]*num_modes
        """
        return theta

    def _parameters_kerr_mass_spin(
        self,
        theta: list,
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
        convert_freqs = M * UnitsToSeconds.tSun
        for i in range(len(self.modes_model)):
            R, phi = theta[2 + 2 * i: 4 + 2 * i]
            omega_r, omega_i = self.transform_mass_spin_to_omegas(
                1,
                a,
                self.df_a_omegas[self.modes_model[i]],
            )
            freq = omega_r / 2 / np.pi / convert_freqs
            tau = 1e3 * convert_freqs / omega_i
            theta_model.extend([R, phi, freq, tau])
        return theta_model

    def _parameters_mass_spin(
        self,
        theta: list,
    ):
        """Compute QNM parameters (A or R, phi, omega_r, omega_i) given [A or R, phi, M, A]*num_modes
        create self.theta_model list

        Parameters
        ----------
        theta : list
            injected parameters to the model [A or R, phi, M, A]*num_modes
        """
        theta_model = []

        for i in range(len(self.modes_model)):
            A, phi, M, a = theta[0 + 4 * i: 4 + 4 * i]
            convert_freqs = M * UnitsToSeconds.tSun
            omega_r, omega_i = self.transform_mass_spin_to_omegas(
                1,
                a,
                self.df_a_omegas[self.modes_model[i]],
            )
            freq = omega_r / 2 / np.pi / convert_freqs
            tau = 1e3 * convert_freqs / omega_i
            theta_model.extend([A, phi, freq, tau])
        return theta_model

    def _parameter_df_dtau(
        self,
        theta: list,
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
        convert_freqs = M * UnitsToSeconds.tSun
        for i in range(len(self.modes_model)):
            R, phi, delta_omega_r, delta_omega_i = theta[2 + 4 * i: 6 + 4 * i]
            omega_r_GR, omega_i_GR = self.transform_mass_spin_to_omegas(
                M,
                a,
                self.df_a_omegas[self.modes_model[i]],
            )
            omega_r = omega_r_GR * (1 + delta_omega_r)
            omega_i = omega_i_GR * (1 + delta_omega_i)

            freq = omega_r / 2 / np.pi / convert_freqs
            tau = 1e3 * convert_freqs / omega_i
            theta_model.extend([R, phi, freq, tau])
        return theta_model

    def _parameter_df_dtau_subdominant(
        self,
        theta: list,
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
        convert_freqs = M * UnitsToSeconds.tSun
        delta_omega_r, delta_omega_i = {}, {}
        delta_omega_r[self.modes_model[0]] = 0
        delta_omega_i[self.modes_model[0]] = 0
        delta_omega_r[self.modes_model[1]] = theta[2]
        delta_omega_i[self.modes_model[1]] = theta[3]
        for i in range(len(self.modes_model)):
            R, phi = theta[4 + 2 * i: 6 + 2 * i]
            omega_r_GR, omega_i_GR = self.transform_mass_spin_to_omegas(
                1,
                a,
                self.df_a_omegas[self.modes_model[i]],
            )
            omega_r = omega_r_GR * (1 + delta_omega_r[self.modes_model[i]])
            omega_i = omega_i_GR * (1 + delta_omega_i[self.modes_model[i]])

            freq = omega_r / 2 / np.pi / convert_freqs
            tau = 1e3 * convert_freqs / omega_i
            theta_model.extend([R, phi, freq, tau])
        return theta_model

    def _model_function(self,
                        theta: list,
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

        theta_model = parameter_function(theta)
        # theta_model should have the form [A0, phi0, freq0, tau0, A1, phi1, ...]

        A0, phi0, freq0, tau0 = theta_model[0:4]

        # Fitted A0 will be A_mode*final_mass[solar masses]/(luminosity distance [Gpc])
        amplitude = A0 * UnitsToSeconds.tSun / (UnitsToSeconds.Dist * 1e3)
        # amplitude = A0
        # amplitude ratio between first model and dominant = 1
        # tau is fitted in [ms]
        h_model = self._h_model_qnm(1, phi0, freq0, tau0 * 1e-3)

        # add more modes to data
        for i in range(1, len(self.modes_model)):
            R, phi, freq, tau = theta_model[0 + 4 * i: 4 + 4 * i]
            h_model += self._h_model_qnm(R, phi, freq, tau * 1e-3)

        h_model *= amplitude
        return h_model

    def _h_model_qnm(self,
                     R: float,
                     phi: float,
                     freq: float,
                     tau: float,
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
        angular_mean = np.sqrt(1 / 5 / 4 / np.pi)
        # angular_mean = 1

        h_real = GWFunctions.compute_qnm_fourier(
            self.detector["freq"],
            R,
            phi,
            freq=freq,
            tau=tau,
            part="real",
            convention=self.ft_convention
        )
        h_imag = GWFunctions.compute_qnm_fourier(
            self.detector["freq"],
            R,
            phi,
            freq=freq,
            tau=tau,
            part="imag",
            convention=self.ft_convention
        )

        return angular_mean * (h_real + h_imag)


class TrueParameters(SourceData):
    """Generate true parameters of injected QNMs."""

    def __init__(
        self,
        modes_model: str,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.modes_model = modes_model

    def choose_theta_true(
        self,
        model: str,
    ):
        """Generate a list of the true injected parameters.

        Parameters
        ----------
        model : str
            QNM model. Can be set to {"kerr", "mass_spin",
            "df_dtau", "df_dtau_sub", "freq_tau", "omegas"}
        """
        models = {
            "kerr": self._true_kerr,
            "mass_spin": self._true_mass_spin,
            "df_dtau": self._true_df_dtau,
            "freq_tau": self._true_freq_tau,
            "freq_tau_multi": self._true_freq_tau_multi,
            # "df_dtau_sub": self._true_df_dtau_subdominant(),
        }

        try:
            models[model]()
        except:
            raise ValueError(
                'model should be {"freq_tau", "kerr", "mass_spin", "df_dtau", "df_dtau_sub"}')

    def _true_freq_tau(self):
        self.theta_true = []
        self.theta_labels = []
        self.theta_labels_plain = []

        for mode in self.modes_model:
            if mode != self.modes_model[0]:
                R = self.qnm_modes[mode].amplitude / \
                    self.qnm_modes[self.modes_model[0]].amplitude
                label_R = r"$R_{{{0}}}$".format(mode)
                label_R_plain = f"R_{mode[1]+mode[3]+mode[5]}"

            else:
                R = (
                    self.qnm_modes[mode].amplitude *
                    self.final_mass * (1 + self.redshift) /
                    self.dist_Gpc
                )
                label_R = r"$A_{{{0}}}M_f(1+z)/D_L$".format(mode)
                label_R_plain = f"A_{mode[1]+mode[3]+mode[5]}"

            self.theta_true.extend([
                R,
                float(self.qnm_modes[mode].phase),
                self.qnm_modes[mode].frequency,
                self.qnm_modes[mode].decay_time * 1e3,
            ])

            self.theta_labels.extend([
                label_R,
                r"$\phi_{{{0}}}$".format(mode),
                r"$f_{{{0}}} [Hz]$".format(mode),
                r"$\tau_{{{0}}} [ms]$".format(mode),
            ])
            self.theta_labels_plain.extend([
                label_R_plain,
                f"phi_{mode[1]+mode[3]+mode[5]}",
                f"freq_{mode[1]+mode[3]+mode[5]}",
                f"tau_{mode[1]+mode[3]+mode[5]}",
            ])

    def _true_freq_tau_multi(self):
        self.theta_true = []
        self.theta_labels = []
        self.theta_labels_plain = []
        i = 1
        for mode in self.modes_model:
            if mode != self.modes_model[0]:
                label_R = r"$R_{{{0}}}$".format(mode)
                label_R_plain = f"R_{i}modes"
                label_phi_plain = f'phi_{i}modes'
                label_freq_plain = f'freq_{i}modes'
                label_tau_plain = f'tau_{i}modes'
                R, phi, freq, tau = {}, {}, {}, {},
                for (qnm_mode, qnm) in self.qnm_modes.items():
                    R[qnm_mode] = qnm.amplitude / \
                        self.qnm_modes[self.modes_model[0]].amplitude
                    phi[qnm_mode] = float(qnm.phase)
                    freq[qnm_mode] = qnm.frequency
                    tau[qnm_mode] = qnm.decay_time * 1e3

            else:

                label_R = r"$A_{{{0}}}M_f(1+z)/D_L$".format(mode)
                label_R_plain = f"A_{mode[1]+mode[3]+mode[5]}"
                label_phi_plain = f"phi_{mode[1]+mode[3]+mode[5]}"
                label_freq_plain = f"freq_{mode[1]+mode[3]+mode[5]}"
                label_tau_plain = f"tau_{mode[1]+mode[3]+mode[5]}"

                R = (
                    self.qnm_modes[mode].amplitude *
                    self.final_mass * (1 + self.redshift) /
                    self.dist_Gpc
                )
                phi = float(self.qnm_modes[mode].phase)
                freq = self.qnm_modes[mode].frequency
                tau = self.qnm_modes[mode].decay_time * 1e3

            self.theta_true.extend([
                R,
                phi,
                freq,
                tau,
            ])

            self.theta_labels.extend([
                label_R,
                r"$\phi_{{{0}}}$".format(mode),
                r"$f_{{{0}}} [Hz]$".format(mode),
                r"$\tau_{{{0}}} [ms]$".format(mode),
            ])
            self.theta_labels_plain.extend([
                label_R_plain,
                label_phi_plain,
                label_freq_plain,
                label_tau_plain,
            ])
            i += 1

    def _true_kerr(self):
        self.theta_true = [self.final_mass *
                           (1 + self.redshift), self.final_spin]
        self.theta_labels = [r"$M_f(1+z)$", r"$a_f$"]

        for mode in self.modes_model:
            if mode == self.modes_model[0]:
                self.theta_true.extend([
                    (
                        self.qnm_modes[mode].amplitude *
                        self.final_mass * (1 + self.redshift) /
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
                    self.qnm_modes[mode].amplitude /
                    self.qnm_modes[self.modes_model[0]].amplitude,
                    float(self.qnm_modes[mode].phase),
                ])

                self.theta_labels.extend([
                    r"$R_{{{0}}}$".format(mode),
                    r"$\phi_{{{0}}}$".format(mode),
                ])

    def _true_mass_spin(self):
        self.theta_true = []
        self.theta_labels = []

        for mode in self.modes_model:
            if mode != self.modes_model[0]:
                R = self.qnm_modes[mode].amplitude / \
                    self.qnm_modes[self.modes_model[0]].amplitude
                label_R = r"$R_{{{0}}}$".format(mode)
            else:
                R = (
                    self.qnm_modes[mode].amplitude *
                    self.final_mass * (1 + self.redshift) /
                    self.dist_Gpc
                )
                label_R = r"$A_{{{0}}}M_f(1+z)/D_L$".format(mode)

            self.theta_true.extend([
                R,
                float(self.qnm_modes[mode].phase),
                self.final_mass * (1 + self.redshift),
                self.final_spin,
            ])
            self.theta_labels.extend([
                label_R,
                r"$\phi_{{{0}}}$".format(mode),
                r"$(1+z)M_{{{0}}}$".format(mode),
                r"$a_{{{0}}}$".format(mode),
            ])

    def _true_df_dtau(self):
        self.theta_true = [self.final_mass *
                           (1 + self.redshift), self.final_spin]
        self.theta_labels = [r"$M_f(1+z)$", r"$a_f$"]

        for mode in self.modes_model:
            if mode != self.modes_model[0]:
                R = self.qnm_modes[mode].amplitude / \
                    self.qnm_modes[self.modes_model[0]].amplitude
                label_R = r"$R_{{{0}}}$".format(mode)
            else:
                R = (
                    self.qnm_modes[mode].amplitude *
                    self.final_mass * (1 + self.redshift) /
                    self.dist_Gpc
                )
                label_R = r"$A_{{{0}}}M_f(1+z)/D_L$".format(mode)

            self.theta_true.extend([
                R,
                float(self.qnm_modes[mode].phase),
                0,
                0,
            ])

            self.theta_labels.extend([
                label_R,
                r"$\phi_{{{0}}}$".format(mode),
                r"$\delta f_{{{0}}}/f_{{{0}}}$".format(mode),
                r"$\delta \tau_{{{0}}}/\tau_{{{0}}}$".format(mode),
            ])

    def _true_df_dtau_subdominant(self):
        self.theta_true = [self.final_mass *
                           (1 + self.redshift), self.final_spin, 0, 0]
        self.theta_labels = [
            r"$M_f(1+z)$", r"$a_f$",
            r"$\delta f_{{{0}}}/f_{{{0}}}$".format(self.modes_model[1]),
            r"$\delta \tau_{{{0}}}/\tau_{{{0}}}$".format(self.modes_model[1]),
        ]

        for mode in self.modes_model:
            if mode == self.modes_model[0]:
                self.theta_true.extend([
                    (
                        self.qnm_modes[mode].amplitude *
                        self.final_mass * (1 + self.redshift) /
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
                    self.qnm_modes[mode].amplitude /
                    self.qnm_modes[self.modes_model[0]].amplitude,
                    float(self.qnm_modes[mode].phase),
                ])

                self.theta_labels.extend([
                    r"$R_{{{0}}}$".format(mode),
                    r"$\phi_{{{0}}}$".format(mode),
                ])


class Priors(SourceData):
    """Generate priors for QNMs waveform models."""

    def __init__(
        self,
        modes_model: str,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.modes_model = modes_model

    def uniform_prior(
        self,
        model: str,
    ):
        """Generate uniform priors parameters.

        Parameters
        ----------
        model : str
            QNM model. Can be set to {"freq_tau", "kerr",
            "mass_spin", "df_dtau", "df_dtau_sub"}

        """
        models = {
            "kerr": self._prior_kerr,
            "mass_spin": self._prior_mass_spin,
            "df_dtau": self._prior_df_dtau,
            "freq_tau": self._prior_freq_tau,
            # "df_dtau_sub": self._prior_df_dtau_subdominant(),
        }

        try:
            models[model]()
            # self.prior_function = lambda theta: MCMCFunctions.noninfor_log_prior(theta, self.prior_min, self.prior_max)
            self.prior_function = self._prior_mcmc

        except:
            raise ValueError(
                'model should be {"freq_tau", "kerr", "mass_spin", "df_dtau", "df_dtau_sub"}')

    def _prior_mcmc(self, theta):
        return MCMCFunctions.noninfor_log_prior(theta, self.prior_min, self.prior_max)

    def cube_uniform_prior(
        self,
        model: str,
    ):
        """Generate uniform priors parameters. And transform the
        unit cube 'hypercube ~ Unif[0., 1.)' to real values priors
        for MultiNest sampling.

        Parameters
        ----------
        model : str
            QNM model. Can be set to {"freq_tau", "kerr",
            "mass_spin", "df_dtau", "df_dtau_sub"}
        """

        models = {
            "kerr": self._prior_kerr,
            "mass_spin": self._prior_mass_spin,
            "df_dtau": self._prior_df_dtau,
            "freq_tau": self._prior_freq_tau,
            "freq_tau_multi": self._prior_freq_tau_multimodes,
            # "df_dtau_sub": self._prior_df_dtau_subdominant(),
        }

        # try:
        models[model]()
        self.prior_function = self._prior_function
        # self.prior_function = lambda hypercube: self._hypercube_transform(
        #     hypercube,
        #     self.prior_min,
        #     self.prior_max,
        #     self.prior_scale,
        #     )

        # except:
        #     raise ValueError('model should be {"freq_tau", "kerr", "mass_spin", "df_dtau", "df_dtau_sub"}')

    def _prior_function(self, hypercube):
        return self._hypercube_transform(
            hypercube,
            self.prior_min,
            self.prior_max,
            self.prior_scale,
        )

    def _hypercube_transform(
        self,
        hypercube,
        prior_min: list,
        prior_max: list,
        transforms=None,
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
        transforms: list
            List that choose the prior to be 'linear' or 'log'.
            Must have the same length as prior_min and prior_max.
            If set to 'None' or 'linear', all parameters will be linear.
            If set to 'log' all parameters will be in log scale.

        Returns
        -------
        array_like
            Returns transformed cube from Unif[0,1] to [min, max].
        """

        if transforms == None or transforms == 'linear':
            transforms = ['linear'] * len(prior_min)
        elif transforms == 'log':
            transforms = ['linear'] * len(prior_min)

        transform = {
            'linear': lambda a, b, x: a + (b - a) * x,
            'log': lambda a, b, x: a * (b / a)**x,
        }

        cube = np.array(hypercube)
        for i in range(len(prior_min)):
            cube[i] = transform[transforms[i]](
                prior_min[i], prior_max[i], cube[i])

        return cube

    def _prior_freq_tau_multimodes(self):
        self.prior_scale = []
        self.prior_min = []
        self.prior_max = []
        percent = 0.5
        M_min = self.final_mass * (1 - percent)
        M_max = self.final_mass * (1 + percent)

        z_min = self.redshift * (1 - percent)
        z_max = self.redshift * (1 + percent)

        time_scale_min = (1 + z_min) * \
            (M_min / self.mass_f) * UnitsToSeconds.tSun
        time_scale_max = (1 + z_max) * \
            (M_max / self.mass_f) * UnitsToSeconds.tSun

        omegas_r = []
        omegas_i = []
        for (k, v) in self.qnm_modes.items():
            if k == self.modes_model[0]:
                pass
            else:
                omegas_r.append(v.omega_r)
                omegas_i.append(v.omega_i)

        for mode in self.modes_model:
            if mode == self.modes_model[0]:
                A_max = M_max * (1 + z_min) / \
                    (self.luminosity_distance(z_min) * 1e-3) * 10
                A_min = M_min * (1 + z_max) / \
                    (self.luminosity_distance(z_max) * 1e-3) / 10
                self.prior_scale.extend(['log', 'linear', 'log', 'linear'])
                self.prior_min.extend([
                    A_min,
                    0,
                    self.qnm_modes[mode].omega_r / 2 / np.pi / time_scale_max,
                    (time_scale_min / self.qnm_modes[mode].omega_i) * 1e3,
                ])

                self.prior_max.extend([
                    A_max,
                    2 * np.pi,
                    self.qnm_modes[mode].omega_r / 2 / np.pi / time_scale_min,
                    (time_scale_max / self.qnm_modes[mode].omega_i) * 1e3,
                ])
            else:
                A_min = 0
                A_max = 0.9
                self.prior_scale.extend(['linear', 'linear', 'log', 'linear'])

                self.prior_min.extend([
                    A_min,
                    0,
                    min(omegas_r) / 2 / np.pi / time_scale_max,
                    (time_scale_min / max(omegas_i)) * 1e3,
                ])

                self.prior_max.extend([
                    A_max,
                    2 * np.pi,
                    max(omegas_r) / 2 / np.pi / time_scale_min,
                    (time_scale_max / min(omegas_i)) * 1e3,
                ])

    def _prior_freq_tau(self):
        self.prior_scale = []
        self.prior_min = []
        self.prior_max = []
        percent = 0.5
        M_min = self.final_mass * (1 - percent)
        M_max = self.final_mass * (1 + percent)

        z_min = self.redshift * (1 - percent)
        z_max = self.redshift * (1 + percent)

        time_scale_min = (1 + z_min) * \
            (M_min / self.mass_f) * UnitsToSeconds.tSun
        time_scale_max = (1 + z_max) * \
            (M_max / self.mass_f) * UnitsToSeconds.tSun
        for mode in self.modes_model:

            if mode == self.modes_model[0]:
                A_max = M_max * (1 + z_min) / \
                    (self.luminosity_distance(z_min) * 1e-3) * 10
                A_min = M_min * (1 + z_max) / \
                    (self.luminosity_distance(z_max) * 1e-3) / 10
                # A_max = 10*self.final_mass*(1 + self.redshift)/(self.luminosity_distance(self.redshift)*1e-3)
                # A_min = 0.01*self.final_mass*(1 + self.redshift)/(self.luminosity_distance(self.redshift)*1e-3)
                # A_max = 30318
                # self.prior_scale.extend(['log', 'linear', 'log', 'log'])
                self.prior_scale.extend(['log', 'linear', 'log', 'linear'])
            else:
                A_min = 0
                A_max = 0.9
                self.prior_scale.extend(['linear', 'linear', 'log', 'linear'])

            self.prior_min.extend([
                A_min,
                0,
                self.qnm_modes[mode].omega_r / 2 / np.pi / time_scale_max,
                (time_scale_min / self.qnm_modes[mode].omega_i) * 1e3,
                # 5,
                # 4.03e-05*1e3,
            ])

            self.prior_max.extend([
                A_max,
                2 * np.pi,
                self.qnm_modes[mode].omega_r / 2 / np.pi / time_scale_min,
                (time_scale_max / self.qnm_modes[mode].omega_i) * 1e3,
                # 5000,
                # 18.12*1e3,
            ])

    def _prior_kerr(self):
        self.prior_scale = ['linear', 'linear']
        self.prior_min = [1, 0]
        self.prior_max = [5e3, 0.9999]
        percent = 0.5
        M_min = self.final_mass * (1 - percent)
        M_max = self.final_mass * (1 + percent)
        z_min = self.redshift * (1 - percent)
        z_max = self.redshift * (1 + percent)
        time_scale_min = (M_min / self.mass_f) * \
            (1 + z_min) * UnitsToSeconds.tSun
        time_scale_max = (M_max / self.mass_f) * \
            (1 + z_max) * UnitsToSeconds.tSun

        for mode in self.modes_model:

            if mode == self.modes_model[0]:
                A_max = M_max * (1 + z_min) / \
                    (self.luminosity_distance(z_min) * 1e-3) * 10
                A_min = M_min * (1 + z_max) / \
                    (self.luminosity_distance(z_max) * 1e-3) / 10
                # A_max = 30318
                self.prior_scale.extend(['log', 'linear'])
            else:
                A_max = 0.9
                A_min = 0
                self.prior_scale.extend(['linear', 'linear'])

            self.prior_min.extend([
                A_min,
                0,
            ])
            self.prior_max.extend([
                A_max,
                2 * np.pi,
            ])

    def _prior_mass_spin(self):
        self.prior_min = []
        self.prior_max = []

        for mode in self.modes_model:
            self.prior_min.extend([
                0,
                0,
                1,
                0,
            ])

            if mode == self.modes_model[0]:
                A_max = 10 * (
                    self.final_mass * (1 + self.redshift) /
                    self.dist_Gpc
                )
            else:
                A_max = 10

            self.prior_max.extend([
                A_max,
                2 * np.pi,
                self.final_mass * (1 + self.redshift) * 10,
                0.9999,
            ])

    def _prior_df_dtau(self):
        self.prior_min = [1, 0]
        self.prior_max = [self.final_mass * (1 + self.redshift) * 10, 0.9999]

        for mode in self.modes_model:
            self.prior_min.extend([
                0,
                0,
                -1,
                -1,
            ])

            if mode == self.modes_model[0]:
                A_max = 10 * (
                    self.final_mass * (1 + self.redshift) /
                    self.dist_Gpc
                )
            else:
                A_max = 10

            self.prior_max.extend([
                A_max,
                2 * np.pi,
                1,
                1,
            ])

    def _prior_df_dtau_subdominant(self):
        self.prior_min = [1, 0, -1, -1]
        self.prior_max = [self.final_mass * 10, 0.9999, 1, 1]

        for mode in self.modes_model:
            self.prior_min.extend([
                0,
                0,
            ])

            if mode == self.modes_model[0]:
                A_max = 10 * (
                    self.final_mass * (1 + self.redshift) /
                    self.dist_Gpc
                )
            else:
                A_max = 10

            self.prior_max.extend([
                A_max,
                2 * np.pi,
            ])

    def luminosity_distance(self, redshift):
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
        from scipy import integrate

        # cosmological constants
        # values from https://arxiv.org/pdf/1807.06209.pdf
        h = 0.6796
        H_0 = h * 100 * 1e+3  # Huble constant m s**-1 Mpc**-1
        clight = 2.99792458e8  # speed of lightm s**-1
        Dist_H = clight / H_0  # Huble distance

        Omega_M = 0.315
        Omega_Λ = 1 - Omega_M
        Omega_K = 0.0

        def Ez(z): return 1 / np.sqrt(Omega_M * (1 + z)
                                      ** 3 + Omega_K * (1 + z)**2 + Omega_Λ)
        Dist_C = Dist_H * integrate.quad(Ez, 0, redshift)[0]
        Dist_L = (1 + redshift) * Dist_C

        return Dist_L


if __name__ == '__main__':
    np.random.seed(123450)
    m_f = 63
    z = 0.093
    q = 10

    # np.random.seed(4652)
    # m_f, z = 17.257445345175107, 9.883089941558583e-05

    detector = "LIGO"
    modes = ["(2,2,0)"]
    # modes = ["(2,2,0)", "(2,2,1) I"]
    # modes = ["(2,2,1) I"]
    teste = Models(modes, detector, m_f, z, q, "FH")

    print(teste.time_convert)

    print(m_f * (1 + z) * UnitsToSeconds.tSun)
