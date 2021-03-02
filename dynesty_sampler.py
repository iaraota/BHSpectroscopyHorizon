import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import dynesty
from dynesty import utils as dyfunc

from SourceData import SourceData
from modules import MCMCFunctions, GWFunctions
from Models import Models, TrueParameters, Priors

import corner

class DynestySampler(SourceData):
    """Compute posterior probability densities for
    quasinormal modes in the frequency domain."""

    def __init__(self, modes_data:list, modes_model:list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modes_data = modes_data
        self.modes_model = modes_model
        self.inject_data(self.modes_data) # construct self.data
        self.models = Models(self.modes_model, *args, **kwargs)
        self.true_pars = TrueParameters(self.modes_model, *args, **kwargs)
        self.priors = Priors(self.modes_model, *args, **kwargs)

    def run_sampler(
        self,
        model:str,
        ratio:bool,
        ):
        self.true_pars.choose_theta_true(model, ratio)
        self.priors.cube_uniform_prior(model, ratio)
        self.models.choose_model(model, ratio)

        ndim = len(self.true_pars.theta_true)
        sampler = dynesty.NestedSampler(
            self.loglikelihood, 
            self.priors.prior_function,
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
        corner.corner(
            samples_equal,
            truths=self.true_pars.theta_true,
            labels = self.true_pars.theta_labels,
            )
        plt.show()
        print(results.summary())

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
            self.models.model, self.data, self.detector["freq"], self.detector["psd"]
            )

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
    z = 0.15
    z = 0.05
    z = 0.01
    m_f = 3e3
    detector = "LIGO"
    modes = ["(2,2,0)"]
    modes = ["(2,2,0)", "(2,2,1) I"]
    # modes = ["(2,2,0)", "(4,4,0)"]
    # modes = ["(2,2,0)", "(3,3,0)"]
    modes_model = ["(2,2,0)"]
    modes_model = ["(2,2,0)", "(2,2,1) I"]
    # modes_model = ["(2,2,0)", "(4,4,0)"]
    # modes_model = ["(2,2,0)", "(3,3,0)"]
    teste = DynestySampler(modes, modes_model, detector, m_f, z, q, "FH")
    teste.run_sampler('kerr', False)
    # teste.plot()
