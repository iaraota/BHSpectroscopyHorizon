import os
import json
from datetime import datetime
import pathlib

import multiprocessing
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
            sample='unif',
            # sample='rwalk',
            maxiter=10000,
            )
        sampler.run_nested(print_progress=True)

        results = sampler.results
        logZ = results.logz[-1]


        samples = results.samples  # samples
        weights = np.exp(results.logwt - results.logz[-1])  # normalized weights
        samples_equal = dyfunc.resample_equal(samples, weights)

        self.save_estimated_values_and_errors(
        samples_equal,
        self.true_pars.theta_labels_plain,
        self.true_pars.theta_true,
        self.modes_model,
        self.modes_data,
        )
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
            sample='unif',
            # sample='rwalk',
            # maxiter=10000,
            )
        sampler_data.run_nested(print_progress=True)
        # sampler_data.run_nested(print_progress=False)
        results_data = sampler_data.results

        logZ_data = results_data.logz[-1]


        samples_data = results_data.samples  # samples
        weights_data = np.exp(results_data.logwt - results_data.logz[-1])  # normalized weights
        samples_equal_data = dyfunc.resample_equal(samples_data, weights_data)

        self.save_estimated_values_and_errors(
        samples_equal_data,
        self.true_pars_data.theta_labels_plain,
        self.true_pars_data.theta_true,
        self.modes_data,
        self.modes_data,
        )

        logB = logZ - logZ_data
        # print(results_data.summary())
        return logZ, logZ_data,logB

    def _log_likelihood_model(self, theta):
        return self.loglikelihood(self.models.model, theta)

    def _log_likelihood_data(self, theta):
        return self.loglikelihood(self.models_data.model, theta)

    def run_sampler(
        self,
        model:str,
        ):
        self.true_pars.choose_theta_true(model)
        self.priors.cube_uniform_prior(model)
        self.models.choose_model(model)
        print(self.true_pars.theta_true)
        ndim = len(self.true_pars.theta_true)
        from time import time               # use for timing functions
        t0 = time()

        sampler = dynesty.NestedSampler(
            lambda theta: self.loglikelihood(self.models.model, theta), 
            self.priors.prior_function,
            ndim,
            bound='multi',
            sample='rwalk',
            # maxiter=10000,
            )
        t1 = time()

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
        print(f'\ntotal time:{t1-t0}')
        

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


    def save_estimated_values_and_errors(
        self,
        samples,
        label_pars,
        true_pars,
        modes_model,
        modes_data,
        ):

        df_samples = pd.DataFrame(samples, columns = label_pars)

        trues = {}
        for i in range(len(label_pars)):
            trues[label_pars[i]] = true_pars[i]

        path = 'data/samples_pars'
        pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
        
        label_data_modes = ''
        for mode in modes_data:
            label_data_modes += '_'+mode[1]+mode[3]+mode[5]
            
        label_model_modes = ''
        for mode in modes_model:
            label_model_modes += '_'+mode[1]+mode[3]+mode[5]

        for parameter in label_pars:
            file_path = f"{path}/data{label_data_modes}_model{label_model_modes}_par_{parameter}.dat"
            if pathlib.Path(file_path).is_file():
                with open(file_path, "a") as myfile:
                    myfile.write(f"{trues[parameter]}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.95)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.05)-df_samples[parameter].quantile(.5)}\n")
            else:
                with open(file_path, "w") as myfile:
                    myfile.write(f"#(0)true(1)estimated(2)upper(3)lower\n")
                    myfile.write(f"{trues[parameter]}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.95)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.05)-df_samples[parameter].quantile(.5)}\n")



def compute_log_B(B_fac,i,modes_data, modes_model, detector, mass, redshift, q, num, seed):#, redshift):
    np.random.seed(seed)
    dy_sampler = DynestySampler(modes_data, modes_model, detector, mass, redshift, q, "FH")
    logZ_model, logZ_data, logB = dy_sampler.compute_bayes_factor('freq_tau')

    with open(f"data/freq_tau_histogram_{modes_data}_{modes_model}_{num}.dat", "a") as myfile:
        myfile.write(f"{seed}\t{mass}\t{redshift}\t{logZ_data}\t{logZ_model}\t{logB}\n")

    # B_fac[i] = {
    #     'mass': mass,
    #     'redshift': redshift,
    #     label_data: logZ_data,
    #     label_model: logZ_model,
    #     'logB': logB,
    #     }

def one_mode_bayes_histogram(modes_data, modes_model, detector, num, q, num_procs=8):
    manager = multiprocessing.Manager()
    B_factor = manager.dict()
    hist = []
    label_data = 'logZ: '+modes_data[0]
    label_model = 'logZ:'
    for mode in modes_model:
        label_model += ' '+mode

    masses = np.random.choice(np.power(10,np.linspace(1, np.log10(5*10**3), num*10)),num, replace=False)
    redshifts = np.random.choice(np.power(10,np.linspace(-2, 0, num*10)), num, replace=False)
    seeds = np.random.randint(1,1e4, num)

    with open(f"data/freq_tau_histogram_{modes_data}_{modes_model}_{num}.dat", "w") as myfile:
        myfile.write(f"#(0)seed(1)mass(2)redshift(3){label_data}(4){label_model}(5)logB\n")


    processes = []
    j = 0
    while j+(num_procs-1) < len(masses):
        for i in range(j,j+num_procs):
            p = multiprocessing.Process(target=compute_log_B, args=(B_factor, i, modes_data, modes_model, detector, masses[i], redshifts[i], q, num, seeds[i]))
            p.start()
            processes.append(p)
            
        for process in processes:
            process.join()
        j += num_procs
    if j < len(masses):
        for i in range(j,len(masses)):
            p = multiprocessing.Process(target=compute_log_B, args=(B_factor, i, modes_data, modes_model, detector, masses[i], redshifts[i], q, num, seeds[i]))
            p.start()
            processes.append(p)
            
        for process in processes:
            process.join()
    
    
    # for (key, value) in B_factor.items():
    #     hist.append(value['logB'])
    # # with open(f'hist_test.json', 'w') as fp:
    # with open(f'data/freq_tau_histogram_{modes_data}_{modes_model}_{num}.json', 'w') as fp:
    #     json.dump(B_factor._getvalue(), fp, indent=4)
    
    # plt.hist(hist)
    # plt.show()


def find_logB(redshift, modes_data, modes_model, detector, mass, q):
    dy_sampler = DynestySampler(modes_data, modes_model, detector, mass, redshift, q, "FH")
    logB = -dy_sampler.compute_bayes_factor('freq_tau')[2]
    return logB

def spectrocopy_horizon(seed, mass, z0, modes_data, modes_model, detector, q, label_data, label_model):
    np.random.seed(seed)

    with open(f"data/horizon/masses/horizon_{label_data}_{label_model}_{mass}.dat", "w") as myfile:
        myfile.write(f"#(0)seed(1)redshift(2)logB - mass = {mass}\n")

    correct = 8
    error = 0.01
    lower = 8
    upper = 9
    rescale = 0.5
    # z_min, z_max = 0,0
    B_fac = find_logB(z0, modes_data, modes_model, detector, mass, q)

    with open(f"data/horizon/masses/horizon_{label_data}_{label_model}_{mass}.dat", "a") as myfile:
        myfile.write(f"{seed}\t{z0}\t{B_fac}\n")
    
    if B_fac > upper:
        z_min = z0
        while B_fac > upper:
            z_min = z0
            z0 *= 1 + rescale
            B_fac = find_logB(z0, modes_data, modes_model, detector, mass, q)
            with open(f"data/horizon/masses/horizon_{label_data}_{label_model}_{mass}.dat", "a") as myfile:
                myfile.write(f"{seed}\t{z0}\t{B_fac}\n")
        z_max = z0
    elif z0 > 0.01:
        z_max = z0
        while B_fac < lower:
            z_max = z0
            z0 *= 1 - rescale
            B_fac = find_logB(z0, modes_data, modes_model, detector, mass, q)
            with open(f"data/horizon/masses/horizon_{label_data}_{label_model}_{mass}.dat", "a") as myfile:
                myfile.write(f"{seed}\t{z0}\t{B_fac}\n")
        z_min = z0
    if not lower < B_fac < upper:
        while True:
            if z_min <= 0.01:
                break
            z0 = np.random.uniform(z_min, z_max)
            B_fac = find_logB(z0, modes_data, modes_model, detector, mass, q)
            with open(f"data/horizon/masses/horizon_{label_data}_{label_model}_{mass}.dat", "a") as myfile:
                myfile.write(f"{seed}\t{z0}\t{B_fac}\n")

            if lower < B_fac < upper:
                break
            elif B_fac > upper:
                z_min = z0
            elif B_fac < lower:
                z_max = z0
    
    with open(f"data/horizon/horizon_{label_data}_{label_model}.dat", "a") as myfile:
        myfile.write(f"{seed}\t{mass}\t{z0}\t{B_fac}\n")

    return (mass, z0, B_fac)
from multiprocessing import Pool, cpu_count
def compute_horizon_masses(modes_data, modes_model, detector, q):
    label_data = 'data'
    for mode in modes_data:
        label_data += '_'+mode[1]+mode[3]+mode[5]
    label_model = 'model'
    for mode in modes_model:
        label_model += '_'+mode[1]+mode[3]+mode[5]


    with open(f"data/horizon/horizon_{label_data}_{label_model}.dat", "w") as myfile:
        myfile.write(f"#(0)seed(1)mass(2)redshift(3)logB\n")
    mode1 = modes_data[0]
    mode2 = modes_data[1][:7]
    masses, redshifts = np.genfromtxt(f'data/horizon/rayleigh/rayleigh_horizon_{q}_{detector}_{mode1}+{mode2}_detector.dat').T

    seeds = np.random.randint(1,1e4, len(masses))

    # values = tuple((seeds[i], masses[i], redshifts[i], modes_data, modes_model, detector, q, label_data, label_model) for i in range(len(masses)))
    values = [(seeds[i], masses[i], redshifts[i], modes_data, modes_model, detector, q, label_data, label_model) for i in range(len(masses))]
    np.random.shuffle(values)

    i = 1#3#5#7#9#11#13#15#17#19#21#23#25#27#29#31#33#35#37#39#41#43#45#47#49#51#53#55#57#59#61#63#65#67#69#71#73
    i = 1#2#3#4#5#6#7#8#9#10
    i -= 1
    # for i in range(len(masses)):
    spectrocopy_horizon(seeds[i], masses[i], redshifts[i], modes_data, modes_model, detector, q, label_data, label_model)

    # if len(values) > num_procs:
    #     values = values[:num_procs]

    # with Pool() as pool:
    #     res = pool.starmap(spectrocopy_horizon, values)


if __name__ == '__main__':
    """GW190521
    final mass = 150.3
    redshift = 0.72
    spectrocopy horizon = 0.148689
    """
    np.random.seed(9944)
    m_f = 1284
    z = 0.17099759466766967
    q = 1.5

    detector = "LIGO"
    modes = ["(2,2,0)", "(2,2,1) I"]
    # modes = ["(2,2,0)"]
    modes_model = ["(2,2,0)", "(2,2,1) I"]
    modes_model = ["(2,2,0)"]
    teste = DynestySampler(modes, modes_model, detector, m_f, z, q, "FH")
    model = "freq_tau"

    # print(teste.compute_bayes_factor('freq_tau'))
    teste.run_sampler(model)

    # q = 1.5
    # detector = "LIGO"
    # num_procs = 48
    # modes = ["(2,2,0)"]

    # modes_model = ["(2,2,0)", "(4,4,0)"]
    # one_mode_bayes_histogram(modes, modes_model, detector, 500, q, num_procs)
    
    # modes_model = ["(2,2,0)", "(2,2,1) I"]
    # one_mode_bayes_histogram(modes, modes_model, detector, 500, q, num_procs)

    # modes_model = ["(2,2,0)", "(3,3,0)"]
    # one_mode_bayes_histogram(modes, modes_model, detector, 500, q, num_procs)

    # modes_model = ["(2,2,0)", "(2,1,0)"]
    # one_mode_bayes_histogram(modes, modes_model, detector, 500, q, num_procs)


    # modes_data = ["(2,2,0)", "(2,2,1) II"]
    # modes_model = ["(2,2,0)"]
    # detector = "LIGO"
    # q = 1.5
    # compute_horizon_masses(modes_data, modes_model, detector, q)

    # modes_data = ["(2,2,0)", "(4,4,0)"]
    # modes_model = ["(2,2,0)"]
    # detector = "LIGO"
    # q = 1.5
    # compute_horizon_masses(modes_data, modes_model, detector, q)

    # modes_data = ["(2,2,0)", "(3,3,0)"]
    # modes_model = ["(2,2,0)"]
    # detector = "LIGO"
    # q = 1.5
    # compute_horizon_masses(modes_data, modes_model, detector, q)

    # modes_data = ["(2,2,0)", "(2,1,0)"]
    # modes_model = ["(2,2,0)"]
    # detector = "LIGO"
    # q = 1.5
    # compute_horizon_masses(modes_data, modes_model, detector, q)