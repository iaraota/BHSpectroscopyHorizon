import os

import numpy as np
import matplotlib.pyplot as plt

import dynesty
from dynesty import utils as dyfunc

from SourceData import SourceData
from modules import MCMCFunctions, GWFunctions

import corner

class Polychord(SourceData):
    """Compute posterior probability densities for
    quasinormal modes in the frequency domain."""

    def __init__(self, modes_data:list, modes_model:list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modes_data = modes_data
        self.modes_model = modes_model
        self.inject_data(self.modes_data) # construct self.data
        self._theta_true()


    def run_dynesty(self):
        ndim = len(self.theta_true)
        sampler = dynesty.NestedSampler(
            self.loglikelihood, 
            self.prior_transform, 
            ndim,
            bound='multi',
            sample='rwalk',
            maxiter=10000,
            )
        sampler.run_nested()
        results = sampler.results
        samples = results.samples  # samples
        weights = np.exp(results.logwt - results.logz[-1])  # normalized weights
        samples_equal = dyfunc.resample_equal(samples, weights)
        corner.corner(samples_equal, truths=self.theta_true)
        plt.show()
        print(results.summary())
    def prior_transform(self, hypercube):
        """Transforms the uniform random variable 'hypercube ~ Unif[0., 1.)'
        to the parameter of interest 'theta ~ Unif[true/100,true*100]'."""
        transform = lambda a, b, x: a + (b - a) * x
        cube = np.array(hypercube)
        for i in range(len(self.modes_model)):
            cube[0+4*i] = transform(0.0, 10,cube[0 + 4*i])
            cube[1+4*i] = transform(0.0, 2*np.pi,cube[1 + 4*i])
            cube[2+4*i] = transform(self.theta_true[2 + 4*i]/10,
                            self.theta_true[2 + 4*i]*10,cube[2 + 4*i])
            # cube[2+4*i] = transform(0.0, 5000,cube[2 + 4*i])
            cube[3+4*i] = transform(self.theta_true[3 + 4*i]/10,
                            self.theta_true[3 + 4*i]*10,cube[3 + 4*i])
        return cube

    def _theta_true(self):
        """Generate a list of the true injected parameters.
        """
        self.theta_true = []
        for mode in self.modes_model:
            self.theta_true.extend([self.qnm_modes[mode].amplitude,
                float(self.qnm_modes[mode].phase),
                self.qnm_modes[mode].frequency,
                # self.qnm_modes[mode].frequency*1e-2,
                self.qnm_modes[mode].decay_time*1e3])
        self.theta_true = tuple(self.theta_true)

    def loglikelihood(self, theta:list):
        """Generate the likelihood function for QNMs.

        Parameters
        ----------
        theta : array_like
            Model parameters.

        Returns
        -------
        function
            Likelihood for QNMs as a function of parameters theta.
        """

        return MCMCFunctions.log_likelihood_qnm(theta,
            self.model_function, self.data, self.detector["freq"], self.detector["psd"]
            )
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
            # freq *= 1e2
            tau *= 1e-3
            omega_r = freq*2*np.pi*self.time_convert
            omega_i = self.time_convert/tau
            h_model += self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
                self.detector["freq"]*self.time_convert, A, phi, omega_r, omega_i, 
                part = "real", convention = self.ft_convention)
        return h_model


if __name__ == '__main__':
    from datetime import datetime
    np.random.seed(1234)
    start=datetime.now()
    m_f = 500
    z = 0.1
    q = 1.5
    detector = "LIGO"
    modes = ["(2,2,0)", "(2,2,1) I"]
    # modes = ("(2,2,0)", "(4,4,0)")
    modes_model = ["(2,2,0)"]
    teste = Polychord(modes, modes, detector, m_f, z, q, "FH")
    teste.run_dynesty()
    print(datetime.now()-start)
