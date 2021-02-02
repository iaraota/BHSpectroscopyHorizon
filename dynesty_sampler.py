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

    def plot(self):
        plt.loglog(self.data)
        plt.loglog(self.model_function(self.theta_true))

        plt.show()

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
            if i == 0:
                cube[0+4*i] = transform(0.0, 10,cube[0 + 4*i])
            else:
                cube[0+4*i] = transform(0.0, 0.9,cube[0 + 4*i])
            cube[1+4*i] = transform(0.0, 2*np.pi,cube[1 + 4*i])
            cube[2+4*i] = transform(0.0, self.theta_true[2 + 4*i]*10,cube[2 + 4*i])
            # cube[2+4*i] = transform(0.0, 5000,cube[2 + 4*i])
            cube[3+4*i] = transform(0.0, self.theta_true[3 + 4*i]*10,cube[3 + 4*i])
        return cube

    def _theta_true(self):
        """Generate a list of the true injected parameters.
        """
        self.theta_true = []
        for mode in self.modes_model:
            if mode == "(2,2,0)":
                A_0 = 1
            else:
                A_0 = self.qnm_modes["(2,2,0)"].amplitude
            self.theta_true.extend([self.qnm_modes[mode].amplitude/A_0,
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
        A0, phi0, freq0, tau0 = theta[0:4]
        R, phi1, freq1, tau1 = theta[4:]

        tau0 *= 1e-3
        omega_r0 = freq0*2*np.pi*self.time_convert
        omega_i0 = self.time_convert/tau0

        tau1 *= 1e-3
        omega_r1 = freq1*2*np.pi*self.time_convert
        omega_i1 = self.time_convert/tau1

        h_model0 = self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
                self.detector["freq"]*self.time_convert, A0, phi0, omega_r0, omega_i0, 
                part = "real", convention = self.ft_convention)

        h_model1 = self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
                self.detector["freq"]*self.time_convert, A0*R, phi1, omega_r1, omega_i1, 
                part = "real", convention = self.ft_convention)
        return h_model0 + h_model1


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
    # teste.plot()
    print(datetime.now()-start)
