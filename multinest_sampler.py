import os
import json
from datetime import datetime
import pathlib

import multiprocessing
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas as pd

import dynesty
from dynesty import utils as dyfunc
from pymultinest.solve import solve
import pymultinest
pathlib.Path('data/multinest/chains').mkdir(parents=True, exist_ok=True) 

from SourceData import SourceData
from modules import MCMCFunctions, GWFunctions
from Models import Models, TrueParameters, Priors


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
        mass = self.final_mass
        ndim = len(self.true_pars.theta_true)
        file_path = f'data/multinest/chains/mass_{round(mass,1)}_redshift_{self.redshift}/'
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True) 

        result_model = solve(
            LogLikelihood=lambda theta: self.loglikelihood(self.models.model, theta), 
            Prior=self.priors.prior_function,
            n_dims=ndim,
            n_live_points=500,
            outputfiles_basename=file_path+'model-',
            verbose=False,
            )
        logZ_model = result_model['logZ']
        logZerr_model = result_model['logZerr']
        

        self.true_pars_data.choose_theta_true(model)
        self.priors_data.cube_uniform_prior(model)
        self.models_data.choose_model(model)

        ndim_data = len(self.true_pars_data.theta_true)

        result_data = solve(
            LogLikelihood=lambda theta: self.loglikelihood(self.models_data.model, theta),
            Prior=self.priors_data.prior_function,
            n_dims=ndim_data,
            n_live_points=500,
            outputfiles_basename=file_path+'data-',
            verbose=False,
            )
        logZ_data = result_data['logZ']
        logZerr_data = result_data['logZerr']
        
        logB = logZ_data - logZ_model
        logBerr = np.sqrt(logZerr_data**2 + logZerr_model**2)

        # Save parameters 
        self.save_estimated_values_and_errors(
        result_model['samples'],
        self.true_pars.theta_labels_plain,
        self.true_pars.theta_true,
        self.modes_model,
        self.modes_data,
        logB,
        logBerr,
        )


        self.save_estimated_values_and_errors(
        result_data['samples'],
        self.true_pars_data.theta_labels_plain,
        self.true_pars_data.theta_true,
        self.modes_data,
        self.modes_data,
        logB,
        logBerr,
        )

        return logZ_model, logZ_data, logB, logBerr

    def run_sampler(
        self,
        model:str,
        ):
        from time import time               # use for timing functions
        self.true_pars.choose_theta_true(model)
        self.priors.cube_uniform_prior(model)
        self.models.choose_model(model)
        print(self.true_pars.theta_true)
        ndim = len(self.true_pars.theta_true)
        t0 = time()
        result = solve(
            LogLikelihood=lambda theta: self.loglikelihood(self.models.model, theta), 
            Prior=self.priors.prior_function,
            n_dims=ndim,
            n_live_points=500,
            outputfiles_basename='data/multinest/chains/tw-',
            verbose=True,
            )
        t1 = time()
        print()
        print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
        print()
        print('parameter values:')
        parameters = self.true_pars.theta_labels_plain
        for name, col in zip(parameters, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
            print(f'percentil: {np.percentile(col,50)}+{np.percentile(col,90) - np.percentile(col,50)}-{np.percentile(col,10) - np.percentile(col,50)}')
        print(f'\ntotal time:{t1-t0}')
        # make marginal plots by running:
        # $ python multinest_marginals.py chains/3-
        # For that, we need to store the parameter names:
        corner.corner(result['samples'], quantiles=[0.05, 0.5, 0.95], show_titles=True)
        plt.show()

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
        logB,
        logBerr,
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
                    myfile.write(f"{df_samples[parameter].quantile(.05)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{self.final_mass}\t")
                    myfile.write(f"{self.redshift}\t")
                    myfile.write(f"{logB}\t")
                    myfile.write(f"{logBerr}\n")
            else:
                with open(file_path, "w") as myfile:
                    myfile.write(f"#(0)true(1)estimated(2)upper(3)lower(5)mass(6)redshift(7)logB(8)logBerr\n")
                    myfile.write(f"{trues[parameter]}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.95)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.05)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{self.final_mass}\t")
                    myfile.write(f"{self.redshift}\t")
                    myfile.write(f"{logB}\t")
                    myfile.write(f"{logBerr}\n")



def compute_log_B(B_fac,i,modes_data, modes_model, detector, mass, redshift, q, num, seed):#, redshift):
    np.random.seed(seed)
    dy_sampler = DynestySampler(modes_data, modes_model, detector, mass, redshift, q, "FH")
    logZ_model, logZ_data, logB, logBerr = dy_sampler.compute_bayes_factor('freq_tau')

    with open(f"data/freq_tau_histogram_{modes_data}_{modes_model}_{num}.dat", "a") as myfile:
        myfile.write(f"{seed}\t{mass}\t{redshift}\t{logZ_data}\t{logZ_model}\t{logB}\n")


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
    logZ_model, logZ_data, logB, logBerr = dy_sampler.compute_bayes_factor('freq_tau')
    return logB, logBerr

def spectrocopy_horizon(seed, mass, z0, modes_data, modes_model, detector, q, label_data, label_model):
    np.random.seed(seed)

    file_path_mass = f"data/horizon/masses/horizon_{label_data}_{label_model}_{round(mass,1)}.dat"

    with open(file_path_mass, "w") as myfile:
        myfile.write(f"#(0)seed(1)redshift(2)logB(3)logBerr - mass = {mass}\n")

    correct = 8

    rescale = 0.5
    # z_min, z_max = 0,0
    B_fac, B_fac_err = find_logB(z0, modes_data, modes_model, detector, mass, q)
    with open(file_path_mass, "a") as myfile:
        myfile.write(f"{seed}\t{z0}\t{B_fac}\t{B_fac_err}\n")

    if B_fac - B_fac_err > correct:
        z_min = z0
        while B_fac - B_fac_err > correct:
            z_min = z0
            z0 *= 1 + rescale
            B_fac, B_fac_err = find_logB(z0, modes_data, modes_model, detector, mass, q)
            with open(file_path_mass, "a") as myfile:
                myfile.write(f"{seed}\t{z0}\t{B_fac}\t{B_fac_err}\n")
        z_max = z0
    elif z0 > 0.01:
        z_max = z0
        while B_fac + B_fac_err < correct:
            z_max = z0
            z0 *= 1 - rescale
            B_fac, B_fac_err = find_logB(z0, modes_data, modes_model, detector, mass, q)
            with open(file_path_mass, "a") as myfile:
                myfile.write(f"{seed}\t{z0}\t{B_fac}\t{B_fac_err}\n")
        z_min = z0
    if not B_fac - B_fac_err <= correct <= B_fac + B_fac_err:
        while True:
            if z_min <= 0.01:
                break
            z0 = np.random.uniform(z_min, z_max)
            B_fac, B_fac_err = find_logB(z0, modes_data, modes_model, detector, mass, q)
            with open(file_path_mass, "a") as myfile:
                myfile.write(f"{seed}\t{z0}\t{B_fac}\t{B_fac_err}\n")

            if B_fac - B_fac_err <= correct <= B_fac + B_fac_err:
                break
            elif B_fac - B_fac_err > correct:
                z_min = z0
            elif B_fac + B_fac_err < correct:
                z_max = z0
    
    with open(f"data/horizon/horizon_{label_data}_{label_model}.dat", "a") as myfile:
        myfile.write(f"{seed}\t{mass}\t{z0}\t{B_fac}\t{B_fac_err}\n")

    return (mass, z0, B_fac)
from multiprocessing import Pool, cpu_count
def compute_horizon_masses(modes_data, modes_model, detector, q, num_procs = 8):
    label_data = 'data'
    for mode in modes_data:
        label_data += '_'+mode[1]+mode[3]+mode[5]
    label_model = 'model'
    for mode in modes_model:
        label_model += '_'+mode[1]+mode[3]+mode[5]


    with open(f"data/horizon/horizon_{label_data}_{label_model}.dat", "w") as myfile:
        myfile.write(f"#(0)seed(1)mass(2)redshift(3)logB(4)logBerr\n")
    mode1 = modes_data[0]
    mode2 = modes_data[1][:7]
    masses, redshifts = np.genfromtxt(f'data/horizon/rayleigh/rayleigh_horizon_{q}_{detector}_{mode1}+{mode2}_detector.dat').T

    seeds = np.random.randint(1,1e4, len(masses))

    # values = tuple((seeds[i], masses[i], redshifts[i], modes_data, modes_model, detector, q, label_data, label_model) for i in range(len(masses)))
    values = [(seeds[i], masses[i], redshifts[i], modes_data, modes_model, detector, q, label_data, label_model) for i in range(len(masses))]
    np.random.shuffle(values)

    values_multi = values

    if len(values) > num_procs:
        values_multi = values[:num_procs]

    with Pool() as pool:
        res = pool.starmap(spectrocopy_horizon, values_multi)


    if len(values) - len(values_multi) > num_procs:
        values_multi = values[num_procs:2*num_procs]
    else: values_multi = values[num_procs:]

    with Pool() as pool:
        res = pool.starmap(spectrocopy_horizon, values_multi)


if __name__ == '__main__':
    """GW190521
    final mass = 150.3
    redshift = 0.72
    spectrocopy horizon = 0.148689
    """
    # np.random.seed(9944)
    # m_f = 150.3
    # z = 0.1
    # q = 1.5

    # detector = "LIGO"
    # modes = ["(2,2,0)", "(2,2,1) I"]
    # # modes = ["(2,2,0)"]
    # # modes_model = ["(2,2,0)", "(2,2,1) I"]
    # modes_model = ["(2,2,0)"]
    # teste = DynestySampler(modes, modes_model, detector, m_f, z, q, "FH")
    # model = "freq_tau"

    # print(teste.compute_bayes_factor('freq_tau'))
    # teste.run_sampler(model)

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

    num_procs = 48
    modes_data = ["(2,2,0)", "(2,2,1) II"]
    modes_model = ["(2,2,0)"]
    detector = "LIGO"
    q = 1.5
    compute_horizon_masses(modes_data, modes_model, detector, q, num_procs)

    modes_data = ["(2,2,0)", "(4,4,0)"]
    modes_model = ["(2,2,0)"]
    detector = "LIGO"
    q = 1.5
    compute_horizon_masses(modes_data, modes_model, detector, q, num_procs)

    modes_data = ["(2,2,0)", "(3,3,0)"]
    modes_model = ["(2,2,0)"]
    detector = "LIGO"
    q = 1.5
    compute_horizon_masses(modes_data, modes_model, detector, q, num_procs)

    modes_data = ["(2,2,0)", "(2,1,0)"]
    modes_model = ["(2,2,0)"]
    detector = "LIGO"
    q = 1.5
    compute_horizon_masses(modes_data, modes_model, detector, q, num_procs)