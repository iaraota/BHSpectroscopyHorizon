import os
import json
from datetime import datetime
import pathlib

# from mpi4py import MPI

from multiprocessing import Pool, cpu_count

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


class MultiNestSampler(SourceData):
    """Compute posterior probability densities for
    quasinormal modes in the frequency domain."""

    def __init__(self, modes_data:list, modes_model:list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modes_data = modes_data
        self.modes_model = modes_model
        self.args = args
        self.kwargs = kwargs
        # construct self.data
        self.inject_data(self.modes_data) 
        self.models = Models(self.modes_model, *args, **kwargs)
        self.true_pars = TrueParameters(self.modes_model, *args, **kwargs)
        self.priors = Priors(self.modes_model, *args, **kwargs)

        self.models_data = Models(self.modes_data, *args, **kwargs)
        self.true_pars_data = TrueParameters(self.modes_data, *args, **kwargs)
        self.priors_data = Priors(self.modes_data, *args, **kwargs)

    def compute_bayes_factor(
        self,
        model:str,
        seed=np.random.get_state()[1][0],
        ):
        self.true_pars.choose_theta_true(model)
        self.priors.cube_uniform_prior(model)
        self.models.choose_model(model)
        mass = self.final_mass
        ndim = len(self.true_pars.theta_true)

        label_data = 'data'
        for mode in self.modes_data:
            label_data += '_'+mode[1]+mode[3]+mode[5]
        label_model = 'model'
        for mode in self.modes_model:
            label_model += '_'+mode[1]+mode[3]+mode[5]

        file_path = f'data/multinest/chains/{self.q_mass}_{label_data}_{label_model}_mass_{round(mass,1)}_redshift_{self.redshift}_seed_{seed}/'
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

    def compute_bayes_factor_multi_modes(
        self,
        N_modes:int=3,
        ):
        model = 'freq_tau_multi'
        combinations = [[self.modes_model[0]]]

        
        N_modes = min(len(self.modes_model),N_modes)
        
        for i in range(2,N_modes+1):
            combinations.append(self.modes_model[:i])

        mass = self.final_mass

        label_data = 'data'
        for mode in self.modes_data:
            label_data += '_'+mode[1]+mode[3]+mode[5]

        logZ = {}
        logZerr = {}
        for modes_model in combinations:
            models = Models(modes_model, *self.args, **self.kwargs)
            true_pars = TrueParameters(modes_model, *self.args, **self.kwargs)
            priors = Priors(modes_model, *self.args, **self.kwargs)

            true_pars.choose_theta_true(model)
            priors.cube_uniform_prior(model)
            models.choose_model(model)
            ndim = len(true_pars.theta_true)

            n = len(modes_model)
            label_model = f'model_{n}'

            seed = np.random.get_state()[1][0]
            file_path = f'data/multinest/chains/{self.q_mass}_{label_data}_{label_model}_mass_{round(mass,1)}_redshift_{self.redshift}_seed_{seed}/'
            pathlib.Path(file_path).mkdir(parents=True, exist_ok=True) 

            result = solve(
                LogLikelihood=lambda theta: self.loglikelihood(models.model, theta), 
                Prior=priors.prior_function,
                n_dims=ndim,
                n_live_points=max(500, 50*ndim),
                outputfiles_basename=file_path+'model-',
                verbose=False,
                )
            logZ[n] = result['logZ']
            logZerr[n] = result['logZerr']

        logB = {}
        logBerr = {}
        for i in range(N_modes,1,-1):
            logB[i] = logZ[i] - logZ[i-1]
            logBerr[i] = np.sqrt(logZerr[i]**2 + logZerr[i-1]**2)

        return logZ, logZerr, logB, logBerr

    def run_sampler(
        self,
        model:str,
        label:str,
        ):
        from time import time               # use for timing functions
        self.true_pars.choose_theta_true(model)
        self.priors.cube_uniform_prior(model)
        self.models.choose_model(model)
        print(self.true_pars.theta_true)
        ndim = len(self.true_pars.theta_true)
        t0 = time()
        seed = np.random.get_state()[1][0]
        result = solve(
            LogLikelihood=lambda theta: self.loglikelihood(self.models.model, theta), 
            Prior=self.priors.prior_function,
            n_dims=ndim,
            n_live_points=500,
            outputfiles_basename=f'data/multinest/chains/{label}-{seed}-',
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
        print('trues:', self.true_pars.theta_true)
        # make marginal plots by running:
        # $ python multinest_marginals.py chains/3-
        # For that, we need to store the parameter names:
        corner.corner(result['samples'], quantiles=[0.05, 0.5, 0.95], truths=self.true_pars.theta_true,show_titles=True)
        # plt.show()

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
            file_path = f"{path}/sigma_{self.q_mass}_data{label_data_modes}_model{label_model_modes}_par_{parameter}.dat"
            if pathlib.Path(file_path).is_file():
                with open(file_path, "a") as myfile:
                    myfile.write(f"{self.final_mass}\t")
                    myfile.write(f"{self.redshift}\t")
                    myfile.write(f"{logB}\t")
                    myfile.write(f"{logBerr}\t")
                    myfile.write(f"{trues[parameter]}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.84135)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.15865)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.97725)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.2275)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.99865)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.00135)-df_samples[parameter].quantile(.5)}\n")
            else:
                with open(file_path, "w") as myfile:
                    myfile.write(f"#(0)mass(1)redshift(2)logB(3)logBerr(4)true(5)estimated(50%)")
                    myfile.write(f'(6)upper-1sigma(7)lower-1sigma(8)upper-2sigma(9)lower-2sigma(10)upper-3sigma(11)lower-3sigma\n')
                    myfile.write(f"{self.final_mass}\t")
                    myfile.write(f"{self.redshift}\t")
                    myfile.write(f"{logB}\t")
                    myfile.write(f"{logBerr}\t")
                    myfile.write(f"{trues[parameter]}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.84135)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.15865)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.97725)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.2275)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.99865)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.00135)-df_samples[parameter].quantile(.5)}\n")


def multimodes_logB_redshift(mass, modes_data, modes_model, detector, q, cores = 16, z_min = 1e-2, z_max = 5e-1, N_modes=3):
    N = cores
    pathlib.Path('data/horizon').mkdir(parents=True, exist_ok=True) 
    pathlib.Path('data/horizon/logB_redshift').mkdir(parents=True, exist_ok=True) 

    label_data = 'data'
    for mode in modes_data:
        label_data += '_'+mode[1]+mode[3]+mode[5]
    
    num_modes = 0
    for mode in modes_model:
        num_modes += 1
    label_model = f'model_{num_modes}_modes'

    mode_folder = f'data/horizon/logB_redshift/{q}_{label_data}_{label_model}'
    pathlib.Path(mode_folder).mkdir(parents=True, exist_ok=True) 

    if not os.path.exists(mode_folder+f"/{mass}.dat"):
        with open(mode_folder+f"/{mass}.dat", "w") as myfile:
            myfile.write(f"#(0)seed(1)mass(2)redshift")
            for i in range(2, N_modes+1):
                myfile.write(f"({1+i})logB-{i}modes({2+i})logBerr-{i}modes")
            myfile.write("\n")
    
    if not os.path.exists(mode_folder+f"/logZ-{mass}.dat"):
        with open(mode_folder+f"/logZ-{mass}.dat", "w") as myfile:
            myfile.write(f"#(0)seed(1)mass(2)redshift")
            for i in range(1, N_modes+1):
                myfile.write(f"({2+i})logZ-{i}modes({3+i})logBerr-{i}modes")
            myfile.write("\n")

    redshifts =  np.logspace(np.log10(z_min), np.log10(z_max), N, endpoint=True)
    seeds = np.random.randint(1e3,9e3, N)
    # for i in range(len(redshifts)):
    #     multi_logB_redshift(redshifts[i], modes_data, modes_model, detector, mass, q, seeds[0], N_modes)

    values = [(redshifts[i], modes_data, modes_model, detector, mass, q, seeds[0], N_modes) for i in range(len(redshifts))]

    with Pool(processes=cores) as pool:
        res = pool.starmap(multi_logB_redshift, values)

def multi_logB_redshift(redshift, modes_data, modes_model, detector, mass, q, noise_seed, N_modes):
    label_data = 'data'
    for mode in modes_data:
        label_data += '_'+mode[1]+mode[3]+mode[5]

    num_modes = 0
    for mode in modes_model:
        num_modes += 1
    label_model = f'model_{num_modes}_modes'

    file_path = f'data/horizon/logB_redshift/{q}_{label_data}_{label_model}/{mass}.dat'
    file_path_logZ = f'data/horizon/logB_redshift/{q}_{label_data}_{label_model}/logZ-{mass}.dat'

    sampler = MultiNestSampler(modes_data, modes_model, detector, mass, redshift, q, "FH", noise_seed)
    logZ, logZerr, logB, logBerr = sampler.compute_bayes_factor_multi_modes( N_modes)
    key_logB = sorted(list(logB.keys()))
    with open(file_path, "a") as myfile:
        myfile.write(f"{noise_seed}\t{mass}\t{redshift}\t")
        for i in key_logB:
            myfile.write(f"{logB[i]}\t{logBerr[i]}\t")
        myfile.write("\n")

    key_logZ = sorted(list(logZ.keys()))
    with open(file_path_logZ, "a") as myfile:
        myfile.write(f"{noise_seed}\t{mass}\t{redshift}\t")
        for i in key_logZ:
            myfile.write(f"{logZ[i]}\t{logZerr[i]}\t")
        myfile.write("\n")

    return redshift


def compute_log_B(modes_data, modes_model, detector, mass, redshift, q, num, seed):
    noise_seed = np.random.seed(seed)

    save_seed = np.random.get_state()[1][0]
    dy_sampler = MultiNestSampler(modes_data, modes_model, detector, mass, redshift, q, "FH", noise_seed)
    logZ_model, logZ_data, logB, logBerr = dy_sampler.compute_bayes_factor('freq_tau', save_seed)

    with open(f"data/histogram/freq_tau_histogram_{q}_{modes_data}_{modes_model}_{num}.dat", "a") as myfile:
        myfile.write(f"{seed}\t{mass}\t{redshift}\t{logZ_data}\t{logZ_model}\t{logB}\n")

def one_mode_bayes_histogram(modes_data, modes_model, detector, num, q, cores = 4):
    label_data = 'logZ: '+modes_data[0]
    label_model = 'logZ:'
    for mode in modes_model:
        label_model += ' '+mode

    masses = np.random.choice(np.power(10,np.linspace(1, np.log10(5*10**3), num*10)),num, replace=False)
    redshifts = np.random.choice(np.power(10,np.linspace(-2, 0, num*10)), num, replace=False)
    seeds = np.random.randint(1,1e4, num)

    pathlib.Path('data/histogram').mkdir(parents=True, exist_ok=True) 
    with open(f"data/histogram/freq_tau_histogram_{q}_{modes_data}_{modes_model}_{num}.dat", "w") as myfile:
        myfile.write(f"#(0)seed(1)mass(2)redshift(3){label_data}(4){label_model}(5)logB\n")

    # for i in range(len(masses)):
    #     compute_log_B(modes_data, modes_model, detector, masses[i], redshifts[i], q, num, seeds[i])

    values = [(modes_data, modes_model, detector, masses[i], redshifts[i], q, num, seeds[i]) for i in range(len(redshifts))]
    with Pool(processes=cores) as pool:
        res = pool.starmap(compute_log_B, values)
    # processes = []
    # j = 0
    # while j+(num_procs-1) < len(masses):
    #     for i in range(j,j+num_procs):
    #         p = multiprocessing.Process(target=compute_log_B, args=(modes_data, modes_model, detector, masses[i], redshifts[i], q, num, seeds[i]))
    #         p.start()
    #         processes.append(p)
            
    #     for process in processes:
    #         process.join()
    #     j += num_procs
    # if j < len(masses):
    #     for i in range(j,len(masses)):
    #         p = multiprocessing.Process(target=compute_log_B, args=(B_factor, i, modes_data, modes_model, detector, masses[i], redshifts[i], q, num, seeds[i]))
    #         p.start()
    #         processes.append(p)
            
    #     for process in processes:
    #         process.join()

def find_logB(redshift, modes_data, modes_model, detector, mass, q, noise_seed):
    dy_sampler = MultiNestSampler(modes_data, modes_model, detector, mass, redshift, q, "FH", noise_seed)
    logZ_model, logZ_data, logB, logBerr = dy_sampler.compute_bayes_factor('freq_tau')
    return logB, logBerr

def single_logB_redshift(redshift, modes_data, modes_model, detector, mass, q, noise_seed):

    label_data = 'data'
    for mode in modes_data:
        label_data += '_'+mode[1]+mode[3]+mode[5]
    label_model = 'model'
    for mode in modes_model:
        label_model += '_'+mode[1]+mode[3]+mode[5]

    file_path = f'data/horizon/logB_redshift/{q}_{label_data}_{label_model}/{mass}.dat'

    B_fac, B_fac_err = find_logB(redshift, modes_data, modes_model, detector, mass, q, noise_seed)

    with open(file_path, "a") as myfile:
        myfile.write(f"{noise_seed}\t{mass}\t{redshift}\t{B_fac}\t{B_fac_err}\n")

    return redshift, B_fac, B_fac_err 

def logB_redshift(mass, modes_data, modes_model, detector, q, cores = 4, z_min = 1e-2, z_max = 5e-1):
    N = cores
    pathlib.Path('data/horizon').mkdir(parents=True, exist_ok=True) 
    pathlib.Path('data/horizon/logB_redshift').mkdir(parents=True, exist_ok=True) 

    label_data = 'data'
    for mode in modes_data:
        label_data += '_'+mode[1]+mode[3]+mode[5]
    label_model = 'model'
    for mode in modes_model:
        label_model += '_'+mode[1]+mode[3]+mode[5]

    mode_folder = f'data/horizon/logB_redshift/{q}_{label_data}_{label_model}'
    pathlib.Path(mode_folder).mkdir(parents=True, exist_ok=True) 

    if not os.path.exists(mode_folder+f"/{mass}.dat"):
        with open(mode_folder+f"/{mass}.dat", "w") as myfile:
            myfile.write(f"#(0)seed(1)mass(2)redshift(3)logB(4)logBerr\n")
    num = 25
    # redshifts =  np.logspace(np.log10(z_min), np.log10(z_max), N, endpoint=True)
    redshifts = np.random.choice(np.power(10,np.linspace(-2, 0, num*10)), num, replace=False)
    seeds = np.random.randint(1e3,9e3, N)
    values = [(redshifts[i], modes_data, modes_model, detector, mass, q, seeds[0]) for i in range(len(redshifts))]
    logB, logBerr = [], []

    # processes = []
    # for i in range(len(redshifts)):
    #     single_logB_redshift(redshifts[i], modes_data, modes_model, detector, mass, q, seeds[0])
    #     p = multiprocessing.Process(target=single_logB_redshift, args=(redshifts[i], modes_data, modes_model, detector, mass, q, seeds[0]))
    #     p.start()
    #     processes.append(p)
        
    # for process in processes:
    #     process.join()
    with Pool(processes=cores) as pool:
        res = pool.starmap(single_logB_redshift, values)


def gw190521(x):
    """
        mass detector frame:  258.516
        q = 1.5
        (2,2,1)
        redshift:  0.10732144460553265

        (3,3,0)
        redshift:  0.0632085619493346

        (4,4,0)
        redshift:  0.019970024214604934

        (2,1,0)
        redshift:  0.017732352376406746
        
        q = 10
        (2,2,1)
        redshift:  0.015574396868016179

        (3,3,0)
        redshift:  0.04322156769155426

        (4,4,0)
        redshift:  0.029695357258402746

        (2,1,0)
        redshift:  0.009395222366096346
    """
    M_i = 258.516

    redshifts = {
        '(2,2,1) II': 0.10732144460553265,
        '(3,3,0)': 0.0632085619493346,
        '(4,4,0)': 0.019970024214604934,
        '(2,1,0)': 0.017732352376406746,
        }
    # masses = {
    #     '(2,2,1) II': 708.865994,
    #     '(3,3,0)': 757.284529,
    #     '(4,4,0)': 1131.693143,
    #     '(2,1,0)': 478.296588,
    # }

    # redshifts = {
    #     '(2,2,1) II': 0.191997,
    #     '(3,3,0)': 0.122054,
    #     '(4,4,0)': 0.06077,
    #     '(2,1,0)': 0.020971,
    #     }
    M_f = lambda z: M_i/(1+z)
    detector = 'LIGO'
    q = 1.5
    seed=156123
    model = "freq_tau"
    label = 'gw190521_newnoise'
    for (mode, redshift) in redshifts.items():
        modes_data = ['(2,2,0)', mode]
        m_f = M_f(redshift)
        # m_f = masses[mode]/(1 + redshift)
        sampler = MultiNestSampler(modes_data, modes_data, detector, m_f, redshift, q, "FH", seed)
        sampler.run_sampler(model, label+f'_{mode[1]}{mode[3]}{mode[5]}_')

        sampler.true_pars.choose_theta_true(model)
        print(mode)
        print(sampler.true_pars.theta_true, '\n')
        seed *=7



if __name__ == '__main__':
    """GW190521
    final mass = 150.3
    redshift = 0.72
    spectrocopy horizon = 0.148689
    """
    # np.random.seed(2183193789)
    # gw190521(0)
    # m_f = 610
    # z = 0.15
    # q = 1.5
    # np.random.seed(1234)
    # detector = "LIGO"
    # modes_data = ["(2,2,0)", "(2,2,1) I"]
    # # modes_data = ["(2,2,0)"]
    # # modes_data = ["(2,2,0)", "(2,2,1) II", "(3,3,0)", "(4,4,0)", "(2,1,0)"]
    # modes_model = ["(2,2,0)", "(2,2,1) II"]
    # # modes_model = ["(2,2,0)"]
    # seed = 12345
    # teste = MultiNestSampler(modes_data, modes_model, detector, m_f, z, q, "FH", seed)
    # model = "freq_tau"

    # # print(teste.compute_bayes_factor_multi_modes('freq_tau'))
    # label = 'teste'
    # teste.run_sampler(model, label)

    # modes_model = ["(2,2,0)"]
    # detector = "LIGO"
    # modes_data = ["(2,2,0)", "(2,2,1) II"]
    # modes_data = ["(2,2,0)", "(3,3,0)"]
    # modes_data = ["(2,2,0)", "(4,4,0)"]
    # modes_data = ["(2,2,0)", "(2,1,0)"]
    # q = 10
    # cores = 4
    # z_max = 1e-1
    # z_min = 8e-3
    # masses = np.logspace(np.log10(80), 3.5, 40, endpoint=True)
    # num = 25
    # masses = np.random.choice(np.power(10,np.linspace(1, np.log10(5*10**3), num*10)),num, replace=False)
    # for mass in masses:
    #     logB_redshift(mass, modes_data, modes_model, detector, q, cores, z_min, z_max)


    modes_data = ["(2,2,0)", "(2,2,1) II", "(3,3,0)", "(4,4,0)", "(2,1,0)"]
    detector = "LIGO"
    # modes_model = ["(2,2,0)", "(2,2,1) II"]
    modes_model = ["(2,2,0)", "(2,2,1) II", "(3,3,0)"]#, "(4,4,0)", "(2,1,0)"]
    # modes_model = ["(2,2,0)", "(4,4,0)"]
    # modes_model = ["(2,2,0)", "(2,1,0)"]
    q = 1.5
    cores = 16
    z_max = 5e-1
    z_min = 8e-3
    masses = np.logspace(np.log10(30), 3.5, 50, endpoint=True)
    for mass in masses[:16]:
    # for mass in masses[16:32]:
    # for mass in masses[32:]:
        multimodes_logB_redshift(mass, modes_data, modes_model, detector, q, cores, z_min, z_max,3)



    # # #histogram
    # modes_data = ["(2,2,0)"]
    # modes_models = [["(2,2,0)", "(2,2,1) II"],["(2,2,0)", "(3,3,0)"], ["(2,2,0)", "(4,4,0)"], ["(2,2,0)", "(2,1,0)"]]
    # detector = "LIGO"
    # num = 500
    # qs = [10, 1.5]
    # cores = 48
    # for q in qs:
    #     for modes_model in modes_models:
    #         one_mode_bayes_histogram(modes_data, modes_model, detector, num, q, cores)