import emcee
import h5py
import numpy as np
import matplotlib.pyplot as plt 
import corner
import seaborn as sns
from scipy.optimize import minimize
from scipy import interpolate, signal
from modules import GWFunctions, MCMCFunctions, ImportData, PlotFunctions
from multiprocessing import cpu_count
from scipy import signal
import pymc3 as pm


def detector_time_domain(detector, corr = False):
    # import detector data and interpolation function
    detector_data, itp_detector = ImportData.import_detector(detector, True)
    
    psd_max = max(detector_data["psd"])
    f_min, f_max = min(detector_data["freq"]), max(detector_data["freq"])
    df = (detector_data["freq"][1] - detector_data["freq"][0])
    
    # create uniform frequency array (needed for ifft)
    freqs = np.arange(f_min, f_max, df)
    detector_psd = itp_detector(freqs)

    # set psd from 0 to f_min to max(psd)
    # freqs = np.arange(0, f_max + f_min, df)
    # psd_max = 0
    # index_fmin = np.where(freqs<f_min)[0]    
    # index_fmax = np.where(freqs>f_max)[0]    
    # detector_psd =  np.concatenate((np.ones(len(index_fmin))*psd_max, 
    #                 itp_detector(freqs[(index_fmin[-1] + 1):index_fmax[0]]),
    #                 np.ones(len(index_fmax))*psd_max))

    # create random phases for psd 
    # detector_psd = detector_psd*np.exp(1j*np.random.randn(len(freqs))*2*np.pi)/2
    if corr == False:
        detector_psd = detector_psd*np.exp(1j*np.random.uniform(low = -np.pi, high = np.pi,size = (len(freqs),)))
    
    else: detector_psd = detector_psd**2/2
    # make the sprectrum conjugate symmetric
    detector_psd = np.concatenate((detector_psd,np.conj(detector_psd[::-1][0:-1])))
    freqs = np.concatenate((-freqs[::-1], freqs))

    # take the inverse fourier transform
    dt = 1/df/len(detector_psd)
    ifft_psd = np.fft.ifft(detector_psd)/dt
    times = np.arange(0, 1/df, dt)
    
    return ifft_psd, times, dt

def parameter_estimation_qnm(M_f, redshift, q_mass, detector, convention = "FH", model = 1):
    detector_data = ImportData.import_detector(detector, False)

    autocorr, times, dt = detector_time_domain(detector, True)
    autocorr = np.real(autocorr)
    noise, times, dt = detector_time_domain(detector)
    noise = np.real(noise)
    # autocorr = signal.convolve(noise, noise[::-1], mode='same')*dt
    # plt.plot(times, noise)
    # plt.plot(times, autocorr)
    # window = signal.tukey(len(autocorr), 0)
    # plt.loglog(np.fft.fftfreq(len(autocorr), dt),np.sqrt(np.abs(np.fft.fft(autocorr*window)*dt)))
    # plt.loglog(detector_data['freq'], detector_data['psd'])
    # plt.show()
    qnm_pars, mass_f = ImportData.import_simulation_qnm_parameters(q_mass)
    time_unit, strain_unit = GWFunctions.convert_units(M_f, redshift, mass_f)
    qnm_modes = dict()
    for (k,v) in qnm_pars.items():
        qnm_modes[k] = GWFunctions.QuasinormalMode(v["amplitudes"], v["phases"], v["omega"][0], v["omega"][1], M_f, redshift, mass_f)
        # qnm_modes[k].qnm_t = qnm_modes[k].qnm_time(times, "real", "SI")
        qnm_modes[k].qnm_t = strain_unit*qnm_modes[k].qnm_time(times/time_unit, "real", "NR")
    
    mode = "(2,2,0)"
    t_100M = qnm_modes[mode].qnm_time_array(100)

    # plt.plot(times,np.real(noise))
    # plt.plot(times,np.real(noise) + qnm_modes[mode].qnm_t)
    # plt.plot(times,qnm_modes[mode].qnm_t)
    # plt.xlim(0, t_100M)
    # plt.show()


    index_100M = np.where(times<t_100M)[0][-1] 

    times = times[:index_100M + 1]
    autocorr = autocorr[:index_100M + 1]
    injected_data = qnm_modes[mode].qnm_t[:index_100M + 1] + np.real(noise)[:index_100M + 1]
    noise = noise[:index_100M + 1]
    # autocorr = signal.convolve(noise, noise[::-1], mode='same', method="direct")*dt
    

    # window = signal.tukey(len(autocorr), 0)
    # plt.loglog(np.fft.fftfreq(len(autocorr), dt),np.sqrt(np.abs(np.fft.fft(autocorr*window)*dt)))
    # plt.loglog(detector_data['freq'], detector_data['psd'])
    # plt.show()
    def model_function(theta, time_array): 
        # A, phi, freq, tau = theta
        # return strain_unit*GWFunctions.compute_qnm_time(time_array, A, phi, freq=freq, tau = tau*1e-3, part = "real")
        A, phi, omega_r, omega_i = theta
        freq = omega_r/2/np.pi/time_unit
        tau = time_unit/omega_i        
        return strain_unit*A*np.exp(-time_array/tau)*np.cos(2*np.pi*freq*time_array - phi)

    # maximize likelihood
    # theta_true = np.array([qnm_modes[mode].amplitude, qnm_modes[mode].phase, qnm_modes[mode].frequency, qnm_modes[mode].decay_time*1e3])
    theta_true = np.array([qnm_modes[mode].amplitude, qnm_modes[mode].phase, qnm_modes[mode].omega_r, qnm_modes[mode].omega_i])
    initial = theta_true + np.random.randn(len(theta_true))*np.floor(np.log10(theta_true))*1e-4

    
    def log_likelihood_time(theta, model_function, data, times, autocorr):
        model = model_function(theta, times)
        dmm  = data - model
        # conv = signal.convolve(1/autocorr, dmm, mode = "same", method="direct")*[times[1] - times[0]]
        # return -0.5*np.trapz(dmm*conv, times)
        # conv = signal.convolve(autocorr, 1/dmm, mode = "same", method="direct")*[times[1] - times[0]]
        # return -0.5*np.trapz(dmm/conv, times)
        # conv = signal.convolve(1/autocorr, dmm**2, mode = "same", method = "direct")*[times[1] - times[0]]
        # return -0.5*np.trapz(conv, times)
        return -0.5*np.trapz(dmm**2, times)
    
    
    nll = lambda theta: -log_likelihood_time(theta, model_function, injected_data, times, autocorr)
    soln = minimize(nll, initial, method = "L-BFGS-B")
    print(soln)
    print(theta_true)
    print(nll(theta_true))


    # MCMC parameter estimation
    nwalkers, ndim = 100, len(initial)
    pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim) * np.floor(np.log10(initial))

    # limit_min = np.array([0., 0., theta_true[2]/10, theta_true[3]/10])
    # limit_max = np.array([1, 2*np.pi, theta_true[2]*10, theta_true[3]*10])
    # prior_f = lambda theta: MCMCFunctions.noninfor_log_prior(theta, limit_min, limit_max)
    def prior_f(theta):
        # A, phi, freq, tau = theta
        # if 0. < A < 1. and 0. < phi <= 2*np.pi and  theta_true[2]/5 < freq < theta_true[2]*5 and theta_true[3]/5 < tau < theta_true[3]*5:
        A, phi, omega_r, omega_i = theta
        if 0. < A < 1. and 0. < phi <= 2*np.pi and 0 < omega_r < 1 and 0.01 < omega_i < 0.5:
            return 0.0
        return -np.inf

    def log_probability_time(theta, log_prior_function, model_function, data, times, autocorr):
        log_prior = log_prior_function(theta)
        if not np.isfinite(log_prior):
            return -np.inf
        return log_prior + log_likelihood_time(theta, model_function, data, times, autocorr)

    log_pdf = lambda theta: log_probability_time(theta, prior_f, model_function, injected_data, times, autocorr)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_pdf)
    n_step = 1000
    sampler.run_mcmc(pos, n_step, progress=True)
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=int(n_step/2), thin=15, flat=True)

    # logprobs = sampler.get_log_prob()
    # plt.plot(logprobs[:,:])

    labels = ["amplitude", "phase", "frequency", "decay time"]
    # labels = ["amplitude", "phase", "omega_r", "omega_i"]
    fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    # fig = corner.corner(flat_samples, labels=labels, truths=theta_true)
    # fig.suptitle(detector+", $M = {0}, z = {1}$".format(PlotFunctions.scientific_format(M_f, precision = 1), redshift))

    # sns.jointplot(flat_samples[:,2], flat_samples[:,3]*1e3, x="freq (Hz)",y="tau (ms)", kind="kde", levels = 2)
    plt.show()
    # return sampler, flat_samples, theta_true

parameter_estimation_qnm(63, 0.1, 1.5, "LIGO")
# parameter_estimation_qnm(1e5, 0.01, 1.5, "CE")
