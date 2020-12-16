import emcee
import h5py
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import corner
from scipy.optimize import minimize
from scipy import interpolate
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

parameter_estimation_qnm(63, 0.1, 1.5, "LIGO", "FH")
# parameter_estimation_qnm(1e5, 0.01, 1.5, "ET", "FH")
