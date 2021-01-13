import emcee
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import corner
from scipy.optimize import minimize
from scipy import interpolate, stats
from modules import GWFunctions, MCMCFunctions, ImportData, PlotFunctions
from multiprocessing import cpu_count

def parameter_estimation_qnm(M_f, redshift, q_mass, detector, convention = "FH", model = 1):
    detector_data = ImportData.import_detector(detector, False)

    qnm_pars, mass_f = ImportData.import_simulation_qnm_parameters(q_mass)
    qnm_modes = dict()
    for (k,v) in qnm_pars.items():
        qnm_modes[k] = GWFunctions.QuasinormalMode(v["amplitudes"], v["phases"], v["omega"][0], v["omega"][1], M_f, redshift, mass_f)
        qnm_modes[k].qnm_f = qnm_modes[k].qnm_fourier(detector_data["freq"], "real", convention, "SI")

    mode = "(2,2,0)"
    N_data = len(detector_data["psd"])
    injected_data = detector_data["psd"]*np.exp(1j*np.random.rand(N_data)*2*np.pi) + qnm_modes[mode].qnm_f

    time_unit, strain_unit = GWFunctions.convert_units(M_f, redshift, mass_f)
    def model_function(theta): 
        # A, phi, omega_r, omega_i = theta
        A, phi, freq, tau = theta
        omega_r = freq*2*np.pi*time_unit
        omega_i = time_unit/tau
        return time_unit*strain_unit*GWFunctions.compute_qnm_fourier(detector_data["freq"]*time_unit, A, phi, omega_r, omega_i, part = "real", convention = convention)

    print(qnm_modes[mode].frequency)
    # maximize likelihood
    # theta_true = np.array([qnm_modes[mode].amplitude, qnm_modes[mode].phase, qnm_modes[mode].omega_r, qnm_modes[mode].omega_i])
    theta_true = np.array([qnm_modes[mode].amplitude, qnm_modes[mode].phase, qnm_modes[mode].frequency, qnm_modes[mode].decay_time])
    initial = theta_true + np.random.randn(len(theta_true))*np.floor(np.log10(theta_true))*1e-4
    
    nll = lambda theta: - MCMCFunctions.log_likelihood_qnm(theta, model_function, injected_data, detector_data["freq"], detector_data["psd"])
    soln = minimize(nll, initial, method = "L-BFGS-B")
    print(soln)
    print(theta_true)
    print(nll(theta_true))

    # MCMC parameter estimation
    nwalkers, ndim = 100, len(initial)
    pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim) * np.floor(np.log10(initial))

    # limit_min = np.array([0., 0., 0., 0.])
    # limit_max = np.array([1, 2*np.pi, 1, 1])
    limit_min = np.array([0., 0., theta_true[2]/10, theta_true[3]/10])
    limit_max = np.array([1, 2*np.pi, theta_true[2]*10, theta_true[3]*10])
    prior_f = lambda theta: MCMCFunctions.noninfor_log_prior(theta, limit_min, limit_max)
    log_pdf = lambda theta: MCMCFunctions.log_probability_qnm(theta, prior_f, model_function, injected_data, detector_data["freq"], detector_data["psd"])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_pdf)
    n_step = 1000
    sampler.run_mcmc(pos, n_step, progress=True)
    samples = sampler.get_chain()

    flat_samples = sampler.get_chain(discard=int(n_step/2), thin=15, flat=True)

    print(samples.shape)
    # logprobs = sampler.get_log_prob()
    # plt.plot(logprobs[:,:])

    labels = ["amplitude", "phase", "frequency", "decay time"]
    labels = ["amplitude", "phase", "omega_r", "omega_i"]
    fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    fig = corner.corner(flat_samples, labels=labels, truths=theta_true)
    fig.suptitle(detector+", $M = {0}, z = {1}$".format(PlotFunctions.scientific_format(M_f, precision = 1), redshift))

    sns.jointplot(flat_samples[:,2], flat_samples[:,3]*1e3, x="freq (Hz)",y="tau (ms)", kind="kde", levels = 2)

    plt.show()
    # return sampler, flat_samples, theta_true


def parameter_estimation_qnm_2modes(M_f, redshift, q_mass, detector, convention = "FH", model = 1):
    detector_data = ImportData.import_detector(detector, False)

    qnm_pars, mass_f = ImportData.import_simulation_qnm_parameters(q_mass)
    qnm_modes = dict()
    for (k,v) in qnm_pars.items():
        qnm_modes[k] = GWFunctions.QuasinormalMode(v["amplitudes"], v["phases"], v["omega"][0], v["omega"][1], M_f, redshift, mass_f)
        qnm_modes[k].qnm_f = qnm_modes[k].qnm_fourier(detector_data["freq"], "real", convention, "SI")

    mode = "(2,2,0)"
    mode2 = "(3,3,0)"
    N_data = len(detector_data["psd"])
    injected_data = detector_data["psd"]*np.exp(1j*np.random.rand(N_data)*2*np.pi) + qnm_modes[mode].qnm_f 
    plt.loglog(abs(injected_data))

    injected_data =  qnm_modes[mode].qnm_f + qnm_modes[mode2].qnm_f
    plt.loglog(abs(injected_data))

    plt.show()
    plt.close("all")
    time_unit, strain_unit = GWFunctions.convert_units(M_f, redshift, mass_f)
    def model_function(theta): 
        # A, phi, omega_r, omega_i = theta
        A, phi, freq, tau, A2, phi2, freq2, tau2 = theta
        omega_r = freq*2*np.pi*time_unit
        omega_i = time_unit/tau
        omega_r2 = freq2*2*np.pi*time_unit
        omega_i2 = time_unit/tau2
        h = (time_unit*strain_unit*GWFunctions.compute_qnm_fourier(detector_data["freq"]*time_unit, A, phi, omega_r, omega_i, part = "real", convention = convention) +
            time_unit*strain_unit*GWFunctions.compute_qnm_fourier(detector_data["freq"]*time_unit, A2, phi2, omega_r2, omega_i2, part = "real", convention = convention))
        return time_unit*strain_unit*GWFunctions.compute_qnm_fourier(detector_data["freq"]*time_unit, A, phi, omega_r, omega_i, part = "real", convention = convention)

    print(qnm_modes[mode].frequency)
    # maximize likelihood
    # theta_true = np.array([qnm_modes[mode].amplitude, qnm_modes[mode].phase, qnm_modes[mode].omega_r, qnm_modes[mode].omega_i])
    theta_true = np.array([qnm_modes[mode].amplitude, qnm_modes[mode].phase, qnm_modes[mode].frequency, qnm_modes[mode].decay_time,
                            qnm_modes[mode2].amplitude, qnm_modes[mode2].phase, qnm_modes[mode2].frequency, qnm_modes[mode2].decay_time])
    initial = theta_true + np.random.randn(len(theta_true))*np.floor(np.log10(theta_true))*1e-4
    
    nll = lambda theta: - MCMCFunctions.log_likelihood_qnm(theta, model_function, injected_data, detector_data["freq"], detector_data["psd"])
    soln = minimize(nll, initial, method = "L-BFGS-B")
    print(soln)
    print(theta_true)
    print(nll(theta_true))

    # MCMC parameter estimation
    nwalkers, ndim = 100, len(initial)
    pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim) * np.floor(np.log10(initial))

    # limit_min = np.array([0., 0., 0., 0.])
    # limit_max = np.array([1, 2*np.pi, 1, 1])
    limit_min = np.array([0., 0., theta_true[2]/10, theta_true[3]/10, 0., 0., theta_true[6]/10, theta_true[7]/10])
    limit_max = np.array([1, 2*np.pi, theta_true[2]*10, theta_true[3]*10, 1, 2*np.pi, theta_true[6]*10, theta_true[7]*10])
    prior_f = lambda theta: MCMCFunctions.noninfor_log_prior(theta, limit_min, limit_max)
    log_pdf = lambda theta: MCMCFunctions.log_probability_qnm(theta, prior_f, model_function, injected_data, detector_data["freq"], detector_data["psd"])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_pdf)
    n_step = 500
    sampler.run_mcmc(pos, n_step, progress=True)
    samples = sampler.get_chain()

    flat_samples = sampler.get_chain(discard=int(n_step/2), thin=15, flat=True)

    print(samples.shape)
    # logprobs = sampler.get_log_prob()
    # plt.plot(logprobs[:,:])

    labels = ["amplitude", "phase", "frequency", "decay time", "amplitude2", "phase2", "frequency2", "decay time2"]
    # labels = ["amplitude", "phase", "omega_r", "omega_i"]
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    # fig = corner.corner(flat_samples, labels=labels, truths=theta_true)
    # fig.suptitle(detector+", $M = {0}, z = {1}$".format(PlotFunctions.scientific_format(M_f, precision = 1), redshift))

    sns.jointplot(flat_samples[:,2], flat_samples[:,3]*1e3, x="freq (Hz)",y="tau (ms)", kind="kde", levels = 2)

    plt.show()
    # return sampler, flat_samples, theta_true


# parameter_estimation_qnm_2modes(63, 0.1, 1.5, "LIGO", "FH")
# parameter_estimation_qnm_2modes(120, 0.0001, 10, "LIGO", "FH")
# parameter_estimation_qnm(1e5, 0.01, 1.5, "ET", "FH")


class FreqMCMC:

    def __init__(self, modes, M_f, redshift, q_mass, detector, convention = "FH"):
        np.random.seed(1234)
        self.M_f = M_f
        self.qnm_pars, self.mass_f = ImportData.import_simulation_qnm_parameters(q_mass)
        self.redshift = redshift
        # self.q_mass = q_mass
        self.ft_convention = "FH"
        self.detector = ImportData.import_detector(detector, False)
        self.time_convert, self.amplitude_scale = GWFunctions.convert_units(M_f, redshift, self.mass_f)

        self.modes = modes

        self._compute_qnm_modes()
        self._random_noise()


        self.nwalkers = 100
        self.nsteps = 2000
        self.thin = 30

        # self.nwalkers = 32
        # self.nsteps = 500
        # self.thin = 10


    def pdf_two_models(self):
        self.modes_data = self.modes
        self.modes_model = self.modes
        self._inject_data()
        self._theta_true()
        self._maximize_loglike()
        self._prior_and_logpdf()


        self.plot_seaborn()

        self.modes_model = [self.modes[0]]
        self._theta_true()
        self._maximize_loglike()
        self._prior_and_logpdf()
        self.plot_seaborn()

        self.modes_data = [self.modes[0]]
        self.modes_model = self.modes
        self._inject_data()
        self._theta_true()
        self._maximize_loglike()
        self._prior_and_logpdf()


        self.plot_seaborn()

        self.modes_model = [self.modes[0]]
        self._theta_true()
        self._maximize_loglike()
        self._prior_and_logpdf()
        self.plot_seaborn()


    def _compute_pdf(self):
        ndim = len(self.initial)
        pos = (self.max_loglike + 1e-4 * np.random.randn(self.nwalkers, ndim))

        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.log_pdf)
        sampler.run_mcmc(pos, self.nsteps, progress=True)
        self.samples = sampler.get_chain()
        self.flat_samples = sampler.get_chain(discard=int(self.nsteps/2), thin=self.thin, flat=True)


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
        theta_true = self.theta_true
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

        label_modes = ""
        for mode in self.modes:
            label_modes +="_" + mode[1]+mode[3]+mode[5]
        fig.tight_layout()
        plt.savefig("figs/"+str(len(self.modes_data))+"data"+str(len(self.modes_model))+"model"
                    + str(self.M_f) + "_" + str(self.redshift) + "_"
                    + self.detector["label"] + label_modes + ".pdf", dpi = 360)
        # plt.show()

    def plot_corner(self):
        self._plot_labels()
        fig = corner.corner(self.flat_samples, labels=self.labels, truths=self.theta_true)
        fig.suptitle(self.detector["label"]+", $M = {0}, z = {1}$".format(
            PlotFunctions.scientific_format(self.M_f, precision = 1), self.redshift))
        plt.show()

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

    def _inject_data(self):
        self.data = np.copy(self.noise)
        for mode in self.modes_data:
            self.data += self.qnm_modes[mode].qnm_f["real"]

    def _random_noise(self):
        N_data = len(self.detector["psd"]) 
        self.noise = self.detector["psd"]*np.exp(1j*np.random.rand(N_data)*2*np.pi)

    def _compute_qnm_modes(self):
        qnm_modes = dict()
        for (k,v) in self.qnm_pars.items():
            qnm_modes[k] = GWFunctions.QuasinormalMode(v["amplitudes"], v["phases"], v["omega"][0], 
                            v["omega"][1], self.M_f, self.redshift, self.mass_f)
            qnm_modes[k].qnm_f = {
                "real": qnm_modes[k].qnm_fourier(self.detector["freq"],
                        "real", self.ft_convention, "SI"),
                "imag": qnm_modes[k].qnm_fourier(self.detector["freq"],
                        "imag", self.ft_convention, "SI")
                }
            # qnm_modes[k].qnm_t = strain_unit*qnm_modes[k].qnm_time(times/time_unit, part, "NR")
        self.qnm_modes = qnm_modes

    def _theta_true(self):
        self.theta_true = []
        for mode in self.modes_model:
            self.theta_true.extend([self.qnm_modes[mode].amplitude, 
                float(self.qnm_modes[mode].phase),
                self.qnm_modes[mode].frequency*1e-2,
                self.qnm_modes[mode].decay_time*1e3])

    def model_function(self, theta):
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
        self.initial = (self.theta_true +
                    np.random.randn(len(self.theta_true))*np.floor(np.log10(self.theta_true))*1e-4)
        negative_loglike = lambda theta: - MCMCFunctions.log_likelihood_qnm(theta,
            self.model_function, self.data, self.detector["freq"], self.detector["psd"]
            )
        self.max_loglike = minimize(negative_loglike, self.initial, method = "L-BFGS-B").x

    def _prior_and_logpdf(self):
        limit_min = []
        limit_max = []
        for i in range(len(self.modes_model)):
            limit_min.extend([0., 0., self.theta_true[2 + 4*i]/100, self.theta_true[3 + 4*i]/100])
            limit_max.extend([100., 2*np.pi, self.theta_true[2 + 4*i]*100, self.theta_true[3 + 4*i]*100])

        self.prior_function = lambda theta: MCMCFunctions.noninfor_log_prior(theta, limit_min, limit_max)
        self.log_pdf = lambda theta: MCMCFunctions.log_probability_qnm(
            theta, self.prior_function, self.model_function, self.data,
            self.detector["freq"], self.detector["psd"])


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
    m_f = 5e2
    z = 0.1
    q = 1.5
    detector = "LIGO"
    # run_mcmc = FreqMCMC(["(3,3,0)"], 63, 0.1, 1.5, "LIGO", "FH")
    # run_mcmc = FreqMCMC(["(2,2,0)","(2,2,1) I"], 63, 0.1, 1.5, "LIGO", "FH")
    run_mcmc = FreqMCMC(["(2,2,0)", "(4,4,0)"], m_f, z, q, detector, "FH")
    # run_mcmc.pdf_two_models()
    run_mcmc.likelihood_ratio_test()
    run_mcmc = FreqMCMC(["(2,2,0)", "(3,3,0)"], m_f, z, q, detector, "FH")
    # run_mcmc.pdf_two_models()
    run_mcmc.likelihood_ratio_test()
    run_mcmc = FreqMCMC(["(2,2,0)", "(2,2,1) I"], m_f, z, q, detector, "FH")
    run_mcmc.likelihood_ratio_test()
    #run_mcmc.pdf_two_models()
    # run_mcmc.plot_seaborn()
    # run_mcmc2 = FreqMCMC(["(2,2,0)","(2,2,1) I"], 5e2, 0.3, 1.5, "LIGO", "FH")
    # run_mcmc2.plot_seaborn()
