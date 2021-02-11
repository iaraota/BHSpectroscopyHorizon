import os
from datetime import datetime

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

    def run_dynesty(self):
        self._theta_true()
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

        tau0 *= 1e-3
        omega_r0 = freq0*2*np.pi*self.time_convert
        omega_i0 = self.time_convert/tau0

        h_model0 = self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
                self.detector["freq"]*self.time_convert, A0, phi0, omega_r0, omega_i0, 
                part = "real", convention = self.ft_convention)

        R, phi1, freq1, tau1 = theta[4:]
        tau1 *= 1e-3
        omega_r1 = freq1*2*np.pi*self.time_convert
        omega_i1 = self.time_convert/tau1
        h_model1 = self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
                self.detector["freq"]*self.time_convert, A0*R, phi1, omega_r1, omega_i1, 
                part = "real", convention = self.ft_convention)
        return h_model0 + h_model1

    def run_dynesty_amplitude(self):
        self._theta_true_amplitudes()
        ndim = len(self.theta_true)
        sampler = dynesty.NestedSampler(
            self.loglikelihood_amplitudes, 
            self.prior_transform_amplitudes, 
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

    def prior_transform_amplitudes(self, hypercube):
        """Transforms the uniform random variable 'hypercube ~ Unif[0., 1.)'
        to the parameter of interest 'theta ~ Unif[true/100,true*100]'."""
        transform = lambda a, b, x: a + (b - a) * x
        cube = np.array(hypercube)
        for i in range(len(self.modes_model)):
            cube[0+4*i] = transform(0.0, 10,cube[0 + 4*i])
            cube[1+4*i] = transform(0.0, 2*np.pi,cube[1 + 4*i])
            cube[2+4*i] = transform(0.0, self.theta_true[2 + 4*i]*10,cube[2 + 4*i])
            # cube[2+4*i] = transform(0.0, 5000,cube[2 + 4*i])
            cube[3+4*i] = transform(0.0, self.theta_true[3 + 4*i]*10,cube[3 + 4*i])
        return cube

    def loglikelihood_amplitudes(self, theta:list):
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
            self.model_function_amplitudes, self.data, self.detector["freq"], self.detector["psd"]
            )

    def _theta_true_amplitudes(self):
        """Generate a list of the true injected parameters.
        """
        self.theta_true = []
        for mode in self.modes_model:
            self.theta_true.extend([self.qnm_modes[mode].amplitude,
                float(self.qnm_modes[mode].phase),
                self.qnm_modes[mode].frequency,
                self.qnm_modes[mode].decay_time*1e3])
        self.theta_true = tuple(self.theta_true)

    def model_function_amplitudes(self, theta:list):
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
            tau *= 1e-3
            omega_r = freq*2*np.pi*self.time_convert
            omega_i = self.time_convert/tau
            h_model += self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
                self.detector["freq"]*self.time_convert, A, phi, omega_r, omega_i, 
                part = "real", convention = self.ft_convention)
        return h_model

    def run_dynesty_mass_spin(self):
        self.fit_coeff = {}

        for mode in self.modes_model:
            self.fit_coeff[mode] = self.transf_fit_coeff(mode)

        self._theta_true_mass_spin()
        ndim = len(self.theta_true_mass_spin)
        
        # time to run sampler
        start_time=datetime.now()

        sampler = dynesty.NestedSampler(
            self.loglikelihood_mass_spin, 
            self.prior_transform_mass_spin, 
            ndim,
            bound='multi',
            sample='rwalk',
            maxiter=10000,
            )
        sampler.run_nested()
        results = sampler.results
    
        print(datetime.now()-start_time)

        print(results.summary())

        samples = results.samples  # samples
        weights = np.exp(results.logwt - results.logz[-1])  # normalized weights
        samples_equal = dyfunc.resample_equal(samples, weights)
        corner.corner(
            samples_equal,
            truths=self.theta_true_mass_spin,
            labels=self.theta_labels_mass_spin,
            )
        plt.show()

    def prior_transform_mass_spin(self, hypercube):
        """Transforms the uniform random variable 'hypercube ~ Unif[0., 1.)'
        to the parameter of interest 'theta ~ Unif[true/100,true*100]'."""
        transform = lambda a, b, x: a + (b - a) * x
        cube = np.array(hypercube)
        for i in range(len(self.modes_model)):
            cube[0+4*i] = transform(0.0, 10,cube[0 + 4*i])
            cube[1+4*i] = transform(0.0, 2*np.pi,cube[1 + 4*i])
            cube[2+4*i] = transform(0.2, 1,cube[2 + 4*i])
            cube[3+4*i] = transform(0.0, 0.9999,cube[3 + 4*i])
        return cube
        # cube = np.array(hypercube)

        # cube[0] = transform(0.0, 10, cube[0])
        # cube[1] = transform(0.0, 2*np.pi, cube[1])
        # cube[2] = transform(0.0, 1, cube[2])
        # cube[3] = transform(0.0, 0.9999, cube[3])
        # cube[2] = transform(0.0, 1.0, cube[2])
        # cube[3] = transform(0.0, 2*np.pi, cube[3])
        # cube[4] = transform(0.0, 1, cube[4])
        # cube[5] = transform(0.0, 1, cube[5])

        # return cube


    def _theta_true_mass_spin(self):
        """Generate a list of the true injected parameters.
        """
        self.theta_true_mass_spin = []
        self.theta_labels_mass_spin = []

        for mode in self.modes_model:
            self.theta_true_mass_spin.extend(
                [self.qnm_modes[mode].amplitude,
                float(self.qnm_modes[mode].phase),
                self.mass_f,
                self.final_spin])
            self.theta_labels_mass_spin.extend([
                r"$A_{{{0}}}$".format(mode),
                r"$\phi_{{{0}}}$".format(mode),
                r"$M_{{{0}}}$".format(mode),
                r"$a_{{{0}}}$".format(mode),
            ])
        self.theta_true_mass_spin = tuple(self.theta_true_mass_spin)


        # self.theta_true_mass_spin = (
        #     self.qnm_modes[self.modes_model[0]].amplitude,
        #     self.qnm_modes[self.modes_model[0]].phase,
        #     # self.qnm_modes[self.modes_model[1]].amplitude/self.qnm_modes[self.modes_model[0]].amplitude,
        #     # self.qnm_modes[self.modes_model[1]].phase,
        #     self.mass_f,
        #     self.final_spin,
        # )
        # self.theta_labels_mass_spin = (
        #     r"$A_0$",
        #     r"$\phi_0$",
        #     # r"$R$",
        #     # r"$\phi_1$",
        #     r"$M_f$",
        #     r"$a_f$",
        # )

    def loglikelihood_mass_spin(self, theta:list):
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

        return MCMCFunctions.log_likelihood_qnm(
            theta,
            self.model_function_mass_spin,
            self.data,
            self.detector["freq"],
            self.detector["psd"]
            )

    def model_function_mass_spin(self, theta:list):
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
        # A0, phi0, R, phi1, M, a = theta
        # A0, phi0, M0, a0 = theta[0:4]

        # omega_r0, omega_i0 = self.transform_mass_spin_to_omegas(
        #     M0,
        #     a0,
        #     self.fit_coeff[self.modes_model[0]]
        # )

        # h_model0 = self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
        #         self.detector["freq"]*self.time_convert, A0, phi0, omega_r0, omega_i0, 
        #         part = "real", convention = self.ft_convention)

        # h_model = np.copy(h_model0)

        # omega_r1, omega_i1 = self.transform_mass_spin_to_omegas(
        #     M,
        #     a,
        #     self.fit_coeff[self.modes_model[1]]
        # )

        # h_model1 = self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
        #         self.detector["freq"]*self.time_convert, A0*R, phi1, omega_r1, omega_i1, 
        #         part = "real", convention = self.ft_convention)

        # h_model += h_model1

        # return h_model


if __name__ == '__main__':
    np.random.seed(123)
    """GW190521
    final mass = 150.3
    redshift = 0.72
    spectrocopy horizon = 0.148689
    """
    m_f = 500
    z = 0.1
    q = 1.5

    m_f = 150.3
    z = 0.72
    # z = 0.15
    # z = 0.05
    # z = 0.01
    detector = "LIGO"
    modes = ["(2,2,0)"]
    modes = ["(2,2,0)", "(2,2,1) I"]
    # modes = ["(2,2,0)", "(4,4,0)"]
    # modes = ["(2,2,0)", "(3,3,0)"]
    modes_model = ["(2,2,0)"]
    modes_model = ["(2,2,0)", "(2,2,1) I"]
    # modes_model = ["(2,2,0)", "(4,4,0)"]
    # modes_model = ["(2,2,0)", "(3,3,0)"]
    teste = Polychord(modes, modes_model, detector, m_f, z, q, "FH")
    teste.run_dynesty_mass_spin()
    # teste.plot()
