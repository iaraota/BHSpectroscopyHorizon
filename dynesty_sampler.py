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

        self.models_data = Models(self.modes_data, *args, **kwargs)
        self.true_pars_data = TrueParameters(self.modes_data, *args, **kwargs)
        self.priors_data = Priors(self.modes_data, *args, **kwargs)

    def compute_bayes_factor(
        self,
        model:str,
        ):
        self.true_pars.choose_theta_true(model)
        self.priors.cube_uniform_prior(model)
        self.models.choose_model(model)

        ndim = len(self.true_pars.theta_true)
        sampler = dynesty.NestedSampler(
            lambda theta: self.loglikelihood(self.models.model, theta), 
            self.priors.prior_function,
            ndim,
            bound='multi',
            sample='rwalk',
            maxiter=10000,
            )
        sampler.run_nested()
        results = sampler.results
        logZ = results.logz[-1]
        # print(results.summary())



        self.true_pars_data.choose_theta_true(model)
        self.priors_data.cube_uniform_prior(model)
        self.models_data.choose_model(model)

        ndim_data = len(self.true_pars_data.theta_true)
        sampler_data = dynesty.NestedSampler(
            lambda theta: self.loglikelihood(self.models_data.model, theta), 
            self.priors_data.prior_function,
            ndim_data,
            bound='multi',
            sample='rwalk',
            maxiter=10000,
            )
        sampler_data.run_nested()
        results_data = sampler_data.results

        logZ_data = results_data.logz[-1]
        logB = logZ - logZ_data
        # print(results_data.summary())
        return logZ, logZ_data,logB

    def run_sampler(
        self,
        model:str,
        ):
        self.true_pars.choose_theta_true(model)
        self.priors.cube_uniform_prior(model)
        self.models.choose_model(model)
        print(self.true_pars.theta_true)
        ndim = len(self.true_pars.theta_true)
        sampler = dynesty.NestedSampler(
            lambda theta: self.loglikelihood(self.models.model, theta), 
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

    def loglikelihood(self, model, theta:list):
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
            model, self.data, self.detector["freq"], self.detector["psd"]
            )

import multiprocessing

def one_mode_bayes_histogram(modes_data, modes_model, detector, num, q, num_procs=8):
    manager = multiprocessing.Manager()
    B_factor = manager.dict()
    hist = []
    label_data = 'logZ: '+modes_data[0]
    label_model = 'logZ:'
    for mode in modes_model:
        label_model += ' '+mode

    masses = np.random.choice(np.power(10,np.linspace(1, 4, num*10)),num, replace=False)
    redshifts = np.random.choice(np.power(10,np.linspace(-2, 0, num*10)), num, replace=False)
    seeds = np.random.randint(1,1e4, num)
    with open(f"data/freq_tau_histogram_{modes_data}_{modes_model}_{num}.dat", "w") as myfile:
        myfile.write(f"#(0)seed(1)mass(2)redshift(3){label_data}(4){label_model}(5)logB\n")
    def compute_log_B(B_fac,i,mass,redshift):#, redshift):
        np.random.seed(seeds[i])
        dy_sampler = DynestySampler(modes_data, modes_model, detector, mass, redshift, q, "FH")
        logZ_model, logZ_data, logB = dy_sampler.compute_bayes_factor('freq_tau')

        with open(f"data/freq_tau_histogram_{modes_data}_{modes_model}_{num}.dat", "a") as myfile:
            myfile.write(f"{seeds[i]}\t{mass}\t{redshift}\t{logZ_data}\t{logZ_model}\t{logB}\n")

        B_fac[i] = {
            'mass': mass,
            'redshift': redshift,
            label_data: logZ_data,
            label_model: logZ_model,
            'logB': logB,
            }
        del data_mode, model_modes, results_data_mode, results_model_modes

    processes = []
    j = 0
    while j+(num_procs-1) < len(masses):
        for i in range(j,j+num_procs):
            p = multiprocessing.Process(target=compute_log_B, args=(B_factor, i, masses[i], redshifts[i],))
            p.start()
            processes.append(p)
            
        for process in processes:
            process.join()
        j += num_procs
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

def secant_method(f1, f2, x1, x2):
    x3  = (x1*f2 - x2*f1)/(f2 - f1)
    return x3



def compute_horizon_masses(masses, modes_data, modes_model, detector, q, num_procs=8):
    manager = multiprocessing.Manager()
    B_factor = manager.dict()
    hist = []
    label_data = 'data'
    for mode in modes_data:
        label_data += '_'+mode[1]+mode[3]+mode[5]
    label_model = 'model'
    for mode in modes_model:
        label_model += '_'+mode[1]+mode[3]+mode[5]


    def spectrocopy_horizon(mass, modes_data, modes_model, detector, q, label_data, label_model):

        def find_logB(redshift):
            dy_sampler = DynestySampler(modes_data, modes_model, detector, mass, redshift, q, "FH")
            logB = -dy_sampler.compute_bayes_factor('freq_tau')[2]
            return logB

        with open(f"data/horizon_{mass}_{label_data}_{label_model}.dat", "a") as myfile:
            myfile.write(f"(0)mass(1)redshift(2)logB\n")
        z1, z2 = [1e-2, 1]
        correct = 8
        error = 0.01


        B_fac = find_logB(z1)
        print(B_fac)
        if B_fac > correct:
            while True:
                z3_ind = np.random.uniform(np.log10(z1), np.log10(z2))
                z3 = 10**z3_ind
                B_fac = find_logB(z3)
                with open(f"data/horizon_{mass}_{label_data}_{label_model}.dat", "a") as myfile:
                    myfile.write(f"\t{mass}\t{z3}\t{B_fac}\n")
                if correct*(1-error) < B_fac < correct*(1+error):
                    break
                elif B_fac < correct*(1-error):
                    z2 = z3
                elif B_fac > correct*(1+error):
                    z1 = z3

            with open(f"data/horizon_{label_data}_{label_model}.dat", "a") as myfile:
                myfile.write(f"\t{mass}\t{z3}\t{B_fac}\n")
        else:
            with open(f"data/horizon_{mass}_{label_data}_{label_model}.dat", "a") as myfile:
                myfile.write(f"\t{mass}\t{z1}\t{B_fac}\n")


    with open(f"data/horizon_{label_data}_{label_model}.dat", "w") as myfile:
        myfile.write(f"#(0)mass(1)redshift(2)logB\n")

    processes = []
    j = 0
    while j+(num_procs-1) < len(masses):
        for i in range(j,j+num_procs):
            p = multiprocessing.Process(target=spectrocopy_horizon, args=(masses[i], modes_data, modes_model, detector, q, label_data, label_model))
            p.start()
            processes.append(p)
            
        for process in processes:
            process.join()
        j += num_procs
    if j < len(masses):
        for i in range(j,len(masses)):
            p = multiprocessing.Process(target=spectrocopy_horizon, args=(masses[i], modes_data, modes_model, detector, q, label_data, label_model))
            p.start()
            processes.append(p)
            
        for process in processes:
            process.join()

if __name__ == '__main__':
    # np.random.seed(1234)
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
    # m_f, z = 1435.1783817201451, 0.7583677914997191
    # m_f, z = 649.1219576310824, 0.4442706749606883	
    # m_f = 63
    # # z = 0.093
    # np.random.seed(3892)
    # m_f, z =56.603517571162335, 0.012034168364629746 
    # np.random.seed(6157)
    # m_f, z =82.76505824347244, 0.02255644501268386 
    # m_f = 10
    z = 1e-2
    m_f, z = 50, 0.015025968556213002
    teste = DynestySampler(modes, modes_model, detector, m_f, z, q, "FH")
    teste.run_sampler('freq_tau')
    # # teste.plot()

    # q = 1.5
    # detector = "LIGO"
    # num_procs = 8
    # modes = ["(2,2,0)"]
    # modes_model = ["(2,2,0)", "(2,2,1) I"]

    # one_mode_bayes_histogram(modes, modes_model, detector, 500, q, num_procs)


    # masses = [50]#np.random.shuffle(10**np.linspace(1, 4, 240))
    # modes_data = ["(2,2,0)", "(2,2,1) I"]
    # modes_model = ["(2,2,0)"]
    # detector = "LIGO"
    # q = 1.5
    # num_procs = 48
    # compute_horizon_masses(masses, modes_data, modes_model, detector, q, num_procs)