import os
import json
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
        ):
        self.true_pars.choose_theta_true(model)
        self.priors.cube_uniform_prior(model)
        self.models.choose_model(model)

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
        print(results.logz[-1])

        return results

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

import multiprocessing

def one_mode_bayes_histogram(modes_data, modes_model, detector, num, q):
    manager = multiprocessing.Manager()
    B_factor = manager.dict()
    hist = []
    label_data = 'logZ: '+modes[0]
    label_model = 'logZ:'
    for mode in modes_model:
        label_model += ' '+mode

    masses = np.sort(np.random.choice(np.power(10,np.linspace(np.log10(50), 3, num*10)),num, replace=False))
    redshifts = np.sort(np.random.choice(np.power(10,np.linspace(-2, 0, num*10)), num, replace=False))
    with open(f"data/freq_tau_histogram_{modes_data}_{modes_model}_{num}.dat", "w") as myfile:
        myfile.write(f"#(0)seed(1)mass(2)redshift(3){label_data}(4){label_model}(5)logB\n")
    def compute_log_B(B_fac,i,mass,redshift):#, redshift):
        seed = np.random.randint(1,1e4)
        np.random.seed(seed)
        # B_fac[i] = {'logB': redshift}
        data_mode = DynestySampler(modes, modes, detector, mass, redshift, q, "FH")
        results_data_mode = data_mode.run_sampler('freq_tau')

        model_modes = DynestySampler(modes, modes_model, detector, mass, redshift, q, "FH")
        results_model_modes = model_modes.run_sampler('freq_tau')

        with open(f"data/freq_tau_histogram_{modes_data}_{modes_model}_{num}.dat", "a") as myfile:
            myfile.write(f"{seed}\t{mass}\t{redshift}\t{results_data_mode.logz[-1]}\t{results_model_modes.logz[-1]}\t{results_model_modes.logz[-1]-results_data_mode.logz[-1]}\n")

        B_fac[i] = {
            'mass': mass,
            'redshift': redshift,
            label_data: results_data_mode.logz[-1],
            label_model: results_model_modes.logz[-1],
            'logB': results_model_modes.logz[-1]-results_data_mode.logz[-1],
            }
        del data_mode, model_modes, results_data_mode, results_model_modes

    processes = []
    j = 0
    while j+7 < len(masses):
        for i in range(j,j+8):
            p = multiprocessing.Process(target=compute_log_B, args=(B_factor, i, masses[i], redshifts[i],))
            p.start()
            processes.append(p)
            
        for process in processes:
            process.join()
        j += 8
    if j < len(masses):
        for i in range(j,len(masses)):
            p = multiprocessing.Process(target=compute_log_B, args=(B_factor, i, masses[i], redshifts[i],))
            p.start()
            processes.append(p)
            
        for process in processes:
            process.join()
    
    
    for (key, value) in B_factor.items():
        hist.append(value['logB'])
    # with open(f'hist_test.json', 'w') as fp:
    with open(f'data/freq_tau_histogram_{modes_data}_{modes_model}_{num}.json', 'w') as fp:
        json.dump(B_factor._getvalue(), fp, indent=4)
    
    # plt.hist(hist)
    # plt.show()


if __name__ == '__main__':
    np.random.seed(9503)
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
    m_f = 100
    # m_f = 4e3
    detector = "LIGO"
    modes = ["(2,2,0)"]
    # modes = ["(2,2,0)", "(2,2,1) I"]
    # modes = ["(2,2,0)", "(4,4,0)"]
    # modes = ["(2,2,0)", "(3,3,0)"]
    modes_model = ["(2,2,0)"]
    # modes_model = ["(2,2,0)", "(2,2,1) I"]
    # modes_model = ["(2,2,0)", "(4,4,0)"]
    # modes_model = ["(2,2,0)", "(3,3,0)"]
    # m_f, z = 1435.1783817201451, 0.7583677914997191
    # m_f, z = 649.1219576310824, 0.4442706749606883	
    # m_f = 63
    # z = 0.093
    m_f, z =53.622569730582505,	0.010931819736241643
    teste = DynestySampler(modes, modes_model, detector, m_f, z, q, "FH")
    teste.run_sampler('freq_tau')
    # # teste.plot()
    # one_mode_bayes_histogram(modes, modes_model, detector, 150, q)