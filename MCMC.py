import time
import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import corner
from scipy.optimize import minimize
from scipy import stats
import scipy.integrate as integrate
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

    def _inject_data(self):
        """Generate data from noise and QNM waveform.
        """
        # d = n
        self.data = np.copy(self.noise)
        # d = n + modes
        for mode in self.modes_data:
            self.data += self.qnm_modes[mode].qnm_f["real"]

class MCMC(SourceData):
    """Compute posterior probability densities for
    quasinormal modes in the frequency domain."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nwalkers = 100
        self.nsteps = 2000
        self.thin = 30

        # self.nwalkers = 32
        # self.nsteps = 500
        # self.thin = 10

    def _compute_pdf(self):
        """Compute posterior density function of the parameters.
        """
        ndim = len(self.initial)
        # TODO: escolher inicio de caminhada em pontos aleatórios.
        pos = (self.theta_true + 1e-4 * np.random.randn(self.nwalkers, ndim))
        # pos = (self.max_loglike + 1e-4 * np.random.randn(self.nwalkers, ndim))

        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.log_pdf)
        sampler.run_mcmc(pos, self.nsteps, progress=True)
        self.samples = sampler.get_chain()
        self.flat_samples = sampler.get_chain(discard=int(self.nsteps/2), thin=self.thin, flat=True)

    def _theta_true(self):
        """Injected parameters.
        """
        self.theta_true = []
        for mode in self.modes_model:
            self.theta_true.extend([self.qnm_modes[mode].amplitude,
                float(self.qnm_modes[mode].phase),
                self.qnm_modes[mode].frequency*1e-2,
                self.qnm_modes[mode].decay_time*1e3])
        self.theta_true = tuple(self.theta_true)

    def model_function(self, theta:list):
        """Generate waveform model function of QNMs.

        Parameters
        ----------
        theta : array_like
            Model parameters.

        Returns
        -------
        function
            Waveform model as a function of parameters theta.
        """
        h_model = 0
        for i in range(len(self.modes_model)):
            A, phi, freq, tau = theta[0 + 4*i: 4 + 4*i]
            freq *= 1e2
            tau *= 1e-3
            omega_r = freq*2*np.pi*self.time_convert
            omega_i = self.time_convert/tau
            h_model += self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
                self.detector["freq"]*self.time_convert, A, phi, omega_r, omega_i, 
                part = "real", convention = self.ft_convention)
        return h_model

    def likelihood_ratio_test(self):
        # TODO: arrumar esse método, talvez separar a maximização da likelihood e esse em outra classe.
        self.modes_data = self.modes
        self.modes_model = self.modes
        self._inject_data()
        self._theta_true()
        self._maximize_loglike()

        likelihood = {"2data2model": MCMCFunctions.log_likelihood_qnm(self.max_loglike,
            self.model_function, self.data, self.detector["freq"], self.detector["psd"]
            )}

        self.modes_model = [self.modes[0]]
        
        self._theta_true()
        self._maximize_loglike()

        likelihood["2data1model"] = MCMCFunctions.log_likelihood_qnm(self.max_loglike,
            self.model_function, self.data, self.detector["freq"], self.detector["psd"]
            )

        self.modes_data = [self.modes[0]]

        self.modes_model = self.modes
        self._inject_data()
        self._theta_true()
        self._maximize_loglike()

        # likelihood = {"1data2model": MCMCFunctions.log_likelihood_qnm(self.max_loglike,
        #     self.model_function, self.data, self.detector["freq"], self.detector["psd"]
        #     )}
        likelihood["1data2model"] = MCMCFunctions.log_likelihood_qnm(self.max_loglike,
            self.model_function, self.data, self.detector["freq"], self.detector["psd"]
            )

        self.modes_model = [self.modes[0]]
        self._theta_true()
        self._maximize_loglike()

        likelihood["1data1model"] = MCMCFunctions.log_likelihood_qnm(self.max_loglike,
            self.model_function, self.data, self.detector["freq"], self.detector["psd"]
            )


        log_likelihood_ratio = {
            "2 modes": 2*(likelihood["2data2model"] - likelihood["2data1model"]),
            "1 mode": 2*(likelihood["1data2model"] - likelihood["1data1model"])}
        print(log_likelihood_ratio)
        print("data: 1 mode")
        print("2modes/1mode: " + str(stats.chi2.cdf(log_likelihood_ratio["1 mode"], 4)))
        print("1mode/2modes: " +str(stats.chi2.sf(log_likelihood_ratio["1 mode"], 4)))
        print("data: 2 modes")
        print("2modes/1mode: " + str(stats.chi2.cdf(log_likelihood_ratio["2 modes"], 4)))
        print("1mode/2modes: " + str(stats.chi2.sf(log_likelihood_ratio["2 modes"], 4)))

    def _maximize_loglike(self):
        """Compute the maximum of the likelihood for a given model.
        """
        self.initial = (self.theta_true +
                    np.random.randn(len(self.theta_true))*np.floor(np.log10(self.theta_true))*1e-4)
        negative_loglike = lambda theta: - MCMCFunctions.log_likelihood_qnm(theta,
            self.model_function, self.data, self.detector["freq"], self.detector["psd"]
            )
        self.max_loglike = minimize(negative_loglike, self.initial, method = "L-BFGS-B").x

    def _prior_and_logpdf(self):
        """Define priors for the model
        """
        # TODO: escolher prior do modo sub-dominante para pegar info do modo dominante, gaussiana em torno?
        limit_min = []
        limit_max = []
        for i in range(len(self.modes_model)):
            # limit_min.extend([0., 0., self.theta_true[2 + 4*i]/100, self.theta_true[3 + 4*i]/100])
            limit_min.extend([0., 0., 0., 0.])
            limit_max.extend([100., 2*np.pi, self.theta_true[2 + 4*i]*100, self.theta_true[3 + 4*i]*100])

        self.prior_function = lambda theta: MCMCFunctions.noninfor_log_prior(theta, limit_min, limit_max)
        self.log_pdf = lambda theta: MCMCFunctions.log_probability_qnm(
            theta, self.prior_function, self.model_function, self.data,
            self.detector["freq"], self.detector["psd"])

class PlotPDF(MCMC):
    """Plots for MCMC.
    """
    # TODO: generalizar funções de plot para valer para qualquer MCMC. Talvez criar outra classe para fazer a conta com varios modos.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pdf_two_models(self):
        self.modes_data = self.modes
        self.modes_model = [self.modes[0]]
        self._inject_data()
        self._theta_true()
        self._maximize_loglike()
        self._prior_and_logpdf()


        self.plot_seaborn()
        
        # self.modes_model = [self.modes[0]]
        # self._theta_true()
        # self._maximize_loglike()
        # self._prior_and_logpdf()
        # self.plot_seaborn()

        # self.modes_data = [self.modes[0]]
        # self.modes_model = self.modes
        # self._inject_data()
        # self._theta_true()
        # self._maximize_loglike()
        # self._prior_and_logpdf()


        # self.plot_seaborn()

        # self.modes_model = [self.modes[0]]
        # self._theta_true()
        # self._maximize_loglike()
        # self._prior_and_logpdf()
        # self.plot_seaborn()
        
    def plot_walks(self):
        self._plot_labels()
        fig, axes = plt.subplots(len(self.theta_true), figsize=(10, 7), sharex=True)
        for i in range(len(self.theta_true)):
            ax = axes[i]
            ax.plot(self.samples[:, :, i], alpha=0.3)
            ax.set_xlim(0, len(self.samples))
            ax.set_ylabel(self.labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.show()

    def plot_seaborn(self):
        self._compute_pdf()
        self._plot_parameters()
        self._plot_labels()
        df = pd.DataFrame(self.flat_samples, columns=self.labels)
        theta_true = list(self.theta_true)
        for i in range(len(self.modes_model)):
            df[self.labels[2 + 4*i]] *= 1e2
            theta_true[2 + 4*i] *= 1e2
            # df[self.labels[3 + 4*i]] *= 1e-3
        hpds = {}
        perctiles = {}
        for column in df:
            hpds[column] = hpd(df[column], 0.90)
            perctiles[column] = np.percentile(df[column], [5, 50, 95])
        fig = sns.pairplot(df, corner=True, diag_kind="kde", kind="hist", plot_kws=dict(rasterized = True))
        # fig = sns.pairplot(df, corner=True, diag_kind="kde", kind="kde", plot_kws=dict(levels = 3))
        # fig.map_lower(sns.kdeplot, levels = 3)

        for i in range(len(theta_true)):
            # highest probability density region
            # fig.axes[i][i].axvline(x = hpds[self.labels[i]][0], color="tab:blue", ls = "--")
            # fig.axes[i][i].axvline(x = hpds[self.labels[i]][1], color="tab:blue", ls = "--")

            # 90% credible region
            fig.axes[i][i].axvline(x = df[self.labels[i]].quantile(0.05), color="tab:blue", ls="-")
            # fig.axes[i][i].axvline(x = df[self.labels[i]].quantile(.5), color="tab:blue", ls=":")
            fig.axes[i][i].axvline(x = df[self.labels[i]].quantile(.95), color="tab:blue", ls="-")

            # true values
            fig.axes[i][i].axvline(x = theta_true[i], color="tab:red", ls="--", lw=3)

            # Show mean and errors
            n_round = 3
            if self.labels[i][1] == "f" or self.labels[i][2] == "t":
                n_round = 2

            fig.axes[i][i].title.set_text(r"${{{0}}}^{{+{1}}}_{{{2}}}$".format(
                np.round(df[self.labels[i]].quantile(.5),n_round),
                np.round(df[self.labels[i]].quantile(.95) - df[self.labels[i]].quantile(.5),n_round),
                np.round(df[self.labels[i]].quantile(.05) - df[self.labels[i]].quantile(.5),n_round)))

        # fig.axes[-1][-1].set_xlim(0, 2*df[self.labels[i]].quantile(.95))

        label_modes = ""
        for mode in self.modes:
            label_modes +="_" + mode[1]+mode[3]+mode[5]
        fig.tight_layout()
        plt.savefig("figs/"+str(len(self.modes_data))+"data"+str(len(self.modes_model))+"model"
                    + str(self.final_mass) + "_" + str(self.redshift) + "_"
                    + self.detector["label"] + label_modes + ".pdf", dpi = 360)
        # plt.show()

    def _plot_labels(self):
        self.labels = []
        for mode in self.modes_model:
            mode = mode[1]+mode[3]+mode[5]
            # if mode == "(2,2,1) I" or mode == "(2,2,1) II":
            #     mode = "(2,2,1)"
            self.labels.extend([r"$A_{{{0}}}$".format(mode), r"$\phi_{{{0}}}$".format(mode),
                            r"$f_{{{0}}}$ [Hz]".format(mode), r"$\tau_{{{0}}}$ [ms]".format(mode)])

    def _plot_parameters(self):
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.family"] = "STIXGeneral"
        # plt.rcParams["figure.figsize"] = [20, 8]  # plot image size

        SMALL_SIZE = 20
        MEDIUM_SIZE = 25
        BIGGER_SIZE = 35

        plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
        plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)


class Bayes(MCMC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        self.modes_data = self.modes
        self.modes_model = [self.modes[0]]
        self._inject_data()
        self._theta_true()
        # self._prior_and_logpdf()
        # prior = self.prior_function

        loglike = lambda A, phi,f, tau: MCMCFunctions.log_likelihood_qnm((A, phi, f, tau),
            self.model_function, self.data, self.detector["freq"], self.detector["psd"]
            )
        start = time.process_time()
        result = integrate.nquad(loglike, 
            [[0.,1],
            [0., 2*np.pi],
            [self.theta_true[2]/2, self.theta_true[2]*2],
            [self.theta_true[3]/2, self.theta_true[3]*2]])
        print(result)

        print(time.process_time() - start)

        # [0., 0., self.theta_true[2]/100, self.theta_true[3]/100], 
        # [100., 2*np.pi, self.theta_true[2]*100, self.theta_true[3]*100])

def hpd(trace, mass_frac) :
    """
    Returns highest probability density region given by
    a set of samples.

    Parameters
    ----------
    trace : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For example, `massfrac` = 0.95 gives a
        95% HPD.

    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD
    """
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)

    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)

    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n-n_samples]

    # Pick out minimal interval
    min_int = np.argmin(int_width)

    # Return interval
    return np.array([d[min_int], d[min_int+n_samples]])

if __name__ == '__main__':
    np.random.seed(123)
    m_f = 142
    z = 0.8
    q = 1.0

    m_f = 150.3
    z = 0.01
    z = 0.72
    z = 0.15
    z = 0.05
    detector = "LIGO"
    modes = ["(2,2,0)", "(2,2,1) I"]
    # print(source)
    # run_mcmc = FreqMCMC(detector, "FH", ("(2,2,0)", "(2,2,1) I"), m_f, z, q)
    # run_mcmc = FreqMCMC(["(3,3,0)"], 63, 0.1, 1.5, "LIGO", "FH")
    # run_mcmc = PlotPDF("LIGO", ["(2,2,0)","(2,2,1) I"], 63, 0.1, 1.5, "FH")
    run_mcmc = PlotPDF(detector, modes, m_f, z, q, "FH")
    run_mcmc.pdf_two_models()
    run_mcmc.likelihood_ratio_test()
    # run_mcmc = FreqMCMC(["(2,2,0)", "(4,4,0)"], m_f, z, q, detector, "FH")
    # # run_mcmc.pdf_two_models()
    # run_mcmc.likelihood_ratio_test()
    # run_mcmc = FreqMCMC(["(2,2,0)", "(3,3,0)"], m_f, z, q, detector, "FH")
    # # run_mcmc.pdf_two_models()
    # run_mcmc.likelihood_ratio_test()
    # run_mcmc = FreqMCMC(["(2,2,0)", "(2,2,1) I"], m_f, z, q, detector, "FH")
    # run_mcmc.likelihood_ratio_test()
    #run_mcmc.pdf_two_models()
    # run_mcmc.plot_seaborn()
    # run_mcmc2 = FreqMCMC(["(2,2,0)","(2,2,1) I"], 5e2, 0.3, 1.5, "LIGO", "FH")
    # run_mcmc2.plot_seaborn()

    # bayes = Bayes(detector, modes, m_f, z, q, "FH")
    # bayes.compute()