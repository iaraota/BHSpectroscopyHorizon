from . import GWFunctions
import numpy as np

def log_likelihood_qnm(theta, model_function, data, frequency, noise_psd):
    """ Natural logarithm of the likelihood for gravitational waves

    Parameters
    ----------
    theta: array_like
        model parameters
    model_function : array_like
        waveform model as a function of the parameters
    data : array_like
        Detected data
    frequency : array_like
        Frequency array used to compute de integral
    noise_psd : array_like
        Detector noise spectral density

    Returns
    -------
    scalar
        Returns ln(likelihood) of a GW model relative to some data
    """
    model = model_function(theta)
    return GWFunctions.inner_product(frequency, data, model, noise_psd) - GWFunctions.inner_product(frequency, model, model, noise_psd)/2

def noninfor_log_prior(theta, limits_min, limits_max):
    """Non-informative log prior for parameters theta. If
    limits_min < theta < limits max, prior(theta) = 1
    otherwise, prior(theta) = 0

    Parameters
    ----------
    theta : array_like
        Model parameters.
    limits_min : array_like
        Lower limits for parameters theta. Must have same length as theta.
    limits_max : arrau_like
        Upper limits for parameters theta. Must have same length as theta.

    Returns
    -------
    Scalar
        Returns zero if theta in inside choosen range and - infinity otherwise.
    """
    if len(theta) == len(limits_min) and len(theta) == len(limits_max):
        theta, limits_min, limits_max = np.array(theta), np.array(limits_min), np.array(limits_max)
        if all(limits_min < theta) and all(theta < limits_max):
            return 0.0
        return -np.inf

def log_probability_qnm(theta, log_prior_function, model_function, data, frequency, noise_psd):
    """Compute the logarithm of the probability posterior distribution.

    Parameters
    ----------
    theta : array_like
        Parameters of the model.
    log_prior_function : function
        Function for priors distribuition.
    model_function : function
        Model function.
    data : array_like
        Data array.
    frequency : array_like
        Frequency array relative to data and noise.
    noise_psd : array_like
        Noise power spectral density, used to compute inner product. 

    Returns
    -------
    float
        Returns log of the posteiror probability.
    """
    log_prior = log_prior_function(theta)
    if not np.isfinite(log_prior):
        return -np.inf
    return log_prior + log_likelihood_qnm(theta, model_function, data, frequency, noise_psd)

def log_probability(theta, log_prior_function, log_likelihood, model_function, data, frequency, noise_psd):
    """Compute the logarithm of the probability posterior distribution.

    Parameters
    ----------
    theta : array_like
        Parameters of the model.
    log_prior_function : function
        Function for priors distribuition.
    model_function : function
        Model function.
    data : array_like
        Data array.
    frequency : array_like
        Frequency array relative to data and noise.
    noise_psd : array_like
        Noise power spectral density, used to compute inner product. 

    Returns
    -------
    float
        Returns log of the posteiror probability.
    """
    log_prior = log_prior_function(theta)
    if not np.isfinite(log_prior):
        return -np.inf
    return log_prior + log_likelihood(theta, model_function, data, frequency, noise_psd)
