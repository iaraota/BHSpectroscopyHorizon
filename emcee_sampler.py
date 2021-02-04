import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import emcee

from SourceData import SourceData
from modules import MCMCFunctions, GWFunctions

import corner

class EmceeSampler(SourceData):
    """Compute posterior probability densities for
    quasinormal modes in the frequency domain."""

    def __init__(self, modes_data:list, modes_model:list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modes_data = modes_data
        self.modes_model = modes_model
        self.inject_data(self.modes_data) # construct self.data

        self.fit_coeff = {}
        for mode in self.modes_model:
            self.fit_coeff[mode] = self.transf_fit_coeff(mode)

        self.df_a_omegas = {}
        for mode in self.modes_model:
            self.df_a_omegas[mode] = self.create_a_over_M_omegas_dataframe(mode)


    def run_sampler(self):
        """Compute posterior density function of the parameters.
        """
        self._prior_and_logpdf()
        self._theta_true()
        ndim = len(self.theta_true)
        self.nwalkers = 100
        self.nsteps = 1000
        self.thin = 30
        # self.nwalkers = 30
        # self.nsteps = 200
        # self.thin = 15

        # pos = self.prior_transform(np.random.rand(self.nwalkers, ndim))
        pos = (self.theta_true + 1e-4 * np.random.randn(self.nwalkers, ndim))

        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.log_pdf)
        sampler.run_mcmc(pos, self.nsteps, progress=True)
        self.samples = sampler.get_chain()
        self.flat_samples = sampler.get_chain(discard=int(self.nsteps/2), thin=self.thin, flat=True)


        corner.corner(self.flat_samples, truths=self.theta_true, labels = self.theta_labels)
        plt.show()

    def prior_transform(self, hypercube):
        """Transforms the uniform random variable 'hypercube ~ Unif[0., 1.)'
        to the parameter of interest 'theta ~ Unif[true/100,true*100]'."""
        transform = lambda a, b, x: a + (b - a) * x
        cube = np.array(hypercube)
        for i in range(len(self.modes_model)):
            cube[0+4*i] = transform(0.0, 10,cube[0 + 4*i])
            cube[1+4*i] = transform(0.0, 2*np.pi,cube[1 + 4*i])
            cube[2+4*i] = transform(0.9, 1,cube[2 + 4*i])
            cube[3+4*i] = transform(0.0, 0.9999,cube[3 + 4*i])
        return cube
    def _prior_and_logpdf(self):
        """Define priors for the model
        """
        
        limit_min = []
        limit_max = []
        for i in range(len(self.modes_model)):
            limit_min.extend([0., 0., self.final_mass/10, 0])
            limit_max.extend([10., 2*np.pi, self.final_mass*10, 0.9999])

        self.prior_function = lambda theta: MCMCFunctions.noninfor_log_prior(theta, limit_min, limit_max)
        self.log_pdf = lambda theta: MCMCFunctions.log_probability_qnm(
            theta,
            self.prior_function,
            self.model_function,
            self.data,
            self.detector["freq"],
            self.detector["psd"]
            )

    def _theta_true(self):
        """Generate a list of the true injected parameters.
        """
        self.theta_true = []
        self.theta_labels = []

        for mode in self.modes_model:
            self.theta_true.extend(
                [self.qnm_modes[mode].amplitude,
                float(self.qnm_modes[mode].phase),
                self.final_mass,
                self.final_spin])
            self.theta_labels.extend([
                r"$A_{{{0}}}$".format(mode),
                r"$\phi_{{{0}}}$".format(mode),
                r"$M_{{{0}}}$".format(mode),
                r"$a_{{{0}}}$".format(mode),
            ])
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
            A, phi, M, a = theta[0 + 4*i: 4 + 4*i]
            M = M/self.mass_initial
            omega_r, omega_i = self.transform_mass_spin_to_omegas(
                M,
                a,
                self.df_a_omegas[self.modes_model[i]],
                # self.modes_model[i],
                # self.fit_coeff[self.modes_model[i]]
            )
            h_model += self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
                self.detector["freq"]*self.time_convert, A, phi, omega_r, omega_i, 
                part = "real", convention = self.ft_convention)
        return h_model


if __name__ == '__main__':
    np.random.seed(1234)
    m_f = 500
    z = 0.01
    q = 1.5
    detector = "LIGO"
    # modes = ["(2,2,0)"]
    modes = ["(2,2,0)", "(2,2,1) I"]
    # modes = ["(2,2,0)", "(4,4,0)"]
    # modes = ["(2,2,0)", "(3,3,0)"]
    # modes_model = ["(2,2,0)"]
    modes_model = ["(2,2,0)", "(2,2,1) I"]
    # modes_model = ["(2,2,0)", "(4,4,0)"]
    # modes_model = ["(2,2,0)", "(3,3,0)"]
    teste = EmceeSampler(modes, modes_model, detector, m_f, z, q, "FH")
    teste.run_sampler()
    # teste.plot()
