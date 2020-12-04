import emcee
import h5py
import numpy as np
import matplotlib.pyplot as plt 
import corner
from scipy.optimize import minimize
from scipy import interpolate
from modules import GWFunctions, MCMCFunctions, ImportData

def parameter_estimation_qnm(M_f, redshift, q_mass, detector, convention = "FH"):
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
        A, phi, omega_r, omega_i = theta
        return time_unit*strain_unit*GWFunctions.compute_qnm_fourier(detector_data["freq"]*time_unit, A, phi, omega_r, omega_i, part = "real", convention = convention)

    # maximize likelihood
    theta_true = np.array([qnm_modes[mode].amplitude, qnm_modes[mode].phase, qnm_modes[mode].omega_r, qnm_modes[mode].omega_i])
    initial = theta_true + np.random.randn(len(theta_true))*np.floor(np.log10(theta_true))*1e-4
    
    nll = lambda theta: - MCMCFunctions.log_likelihood_qnm(theta, model_function, injected_data, detector_data["freq"], detector_data["psd"])
    soln = minimize(nll, initial, method = "L-BFGS-B")

    # MCMC parameter estimation
    nwalkers, ndim = 32, len(initial)
    pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim) * np.floor(np.log10(initial))

    limit_min = np.array([0., 0., 0., 0.,])
    limit_max = np.array([1, 2*np.pi, 1, 1])
    prior_f = lambda theta: MCMCFunctions.noninfor_log_prior(theta, limit_min, limit_max)
    log_pdf = lambda theta: MCMCFunctions.log_probability_qnm(theta, prior_f, model_function, injected_data, detector_data["freq"], detector_data["psd"])

    print(time_unit)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_pdf)
    sampler.run_mcmc(pos, 2000, progress=True)

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    labels = ["amplitude", "phase", "omega_r", "omega_i"]
    fig = corner.corner(flat_samples, labels=labels, truths=theta_true)
    fig.suptitle(detector+" M = $M_f, z = $redshift")
    plt.show()
parameter_estimation_qnm(63, 0.1, 1.5, "LIGO")