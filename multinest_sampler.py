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

        file_path = f'data/multinest/chains/{self.q_mass}_{label_data}_{label_model}_mass_{round(mass,1)}_redshift_{self.redshift}/'
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
            file_path = f"{path}/{self.q_mass}_data{label_data_modes}_model{label_model_modes}_par_{parameter}.dat"
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
    dy_sampler = MultiNestSampler(modes_data, modes_model, detector, mass, redshift, q, "FH")
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

def logB_redshift(mass, modes_data, modes_model, detector, q, cores = 16, z_min = 1e-2, z_max = 5e-1):
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

    redshifts =  np.logspace(np.log10(z_min), np.log10(z_max), N, endpoint=True)
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

def spectrocopy_horizon_sec_error(seed, mass, z0, modes_data, modes_model, detector, q, label_data, label_model):
    file_path_mass = f"data/horizon/masses/horizon_{q}_{label_data}_{label_model}_{round(mass,1)}_{seed}.dat"

    with open(file_path_mass, "w") as myfile:
        myfile.write(f"#(0)seed(1)redshift(2)logB(3)logBerr - mass = {mass}\n")

    correct = 8

    rescale = 0.5
    # z_min, z_max = 0,0
    B_fac, B_fac_err = find_logB(z0, modes_data, modes_model, detector, mass, q, seed)
    with open(file_path_mass, "a") as myfile:
        myfile.write(f"{seed}\t{z0}\t{B_fac}\t{B_fac_err}\n")
    n = 1
    if  B_fac - B_fac_err > correct:
        z_min = z0
        f_min = B_fac - 8
        while B_fac - B_fac_err  > correct:
            z_min = z0
            z0 *= 1 + rescale
            B_fac, B_fac_err  = find_logB(z0, modes_data, modes_model, detector, mass, q, seed)
            n +=1
            with open(file_path_mass, "a") as myfile:
                myfile.write(f"{seed}\t{z0}\t{B_fac}\t{B_fac_err}\n")
        z_max = z0
        f_max = B_fac - B_fac_err - 8

    elif B_fac + B_fac_err < correct:
        z_max = z0
        f_max = B_fac - 8
        while B_fac + B_fac_err < correct:
            z_max = z0
            if z_max < 1e-2:
                z0 = 1e-3
                break
            z0 *= 1 - rescale
            B_fac, B_fac_err  = find_logB(z0, modes_data, modes_model, detector, mass, q, seed)
            n +=1
            with open(file_path_mass, "a") as myfile:
                myfile.write(f"{seed}\t{z0}\t{B_fac}\t{B_fac_err}\n")
        z_min = z0
        f_min = B_fac + B_fac_err - 8


    while not B_fac - B_fac_err <= correct <= B_fac + B_fac_err:
        np.random.seed(None)
        # if n > 100:
        #     break
        z0 = z_min - f_min*(z_max - z_min)/(f_max - f_min)

        if z_max < 1e-2:
            z0 = 1e-3
            break
        B_fac, B_fac_err = find_logB(z0, modes_data, modes_model, detector, mass, q, seed)
        n +=1
        with open(file_path_mass, "a") as myfile:
            myfile.write(f"{seed}\t{z0}\t{B_fac}\t{B_fac_err}\n")

        f_0 = B_fac - 8
        
        if f_0 > 0:
            z_max = z0
            f_max = f_0
        elif f_0 < 0:
            z_min = z0
            f_min = f_0

    with open(f"data/horizon/horizon_{q}_{label_data}_{label_model}_new.dat", "a") as myfile:
        myfile.write(f"{seed}\t{mass}\t{z0}\t{B_fac}\t{B_fac_err}\n")

    return (mass, z0, B_fac)


def spectrocopy_horizon(seed, mass, z0, modes_data, modes_model, detector, q, label_data, label_model):
    file_path_mass = f"data/horizon/masses/horizon_{q}_{label_data}_{label_model}_{round(mass,1)}_{seed}.dat"

    with open(file_path_mass, "w") as myfile:
        myfile.write(f"#(0)seed(1)redshift(2)logB(3)logBerr - mass = {mass}\n")

    correct = 8

    rescale = 0.2
    # z_min, z_max = 0,0
    B_fac, B_fac_err = find_logB(z0, modes_data, modes_model, detector, mass, q, seed)
    with open(file_path_mass, "a") as myfile:
        myfile.write(f"{seed}\t{z0}\t{B_fac}\t{B_fac_err}\n")
    
    if  B_fac - B_fac_err > correct:
        z_min = z0
        while B_fac - B_fac_err  > correct:
            z_min = z0
            z0 *= 1 + rescale
            B_fac, B_fac_err  = find_logB(z0, modes_data, modes_model, detector, mass, q, seed)
            with open(file_path_mass, "a") as myfile:
                myfile.write(f"{seed}\t{z0}\t{B_fac}\t{B_fac_err}\n")
        z_max = z0

    elif B_fac + B_fac_err < correct:
        z_max = z0
        while B_fac + B_fac_err < correct:
            z_max = z0
            if z_max < 1e-2:
                z0 = 1e-3
                break
            z0 *= 1 - rescale
            B_fac, B_fac_err  = find_logB(z0, modes_data, modes_model, detector, mass, q, seed)
            with open(file_path_mass, "a") as myfile:
                myfile.write(f"{seed}\t{z0}\t{B_fac}\t{B_fac_err}\n")
        z_min = z0


    while not B_fac - B_fac_err <= correct <= B_fac + B_fac_err:
        np.random.seed(None)

        if z_max < 1e-2:
            z0 = 1e-3
            break

        if round(float(z_min), 6) == round(float(z_max), 6):
            if B_fac - B_fac_err > correct:
                z_max *= 1 + rescale
            elif B_fac + B_fac_err < correct:
                z_min *= 1 - rescale
        z0 = np.random.uniform(z_min,z_max)
        B_fac, B_fac_err = find_logB(z0, modes_data, modes_model, detector, mass, q, seed)

        with open(file_path_mass, "a") as myfile:
            myfile.write(f"{seed}\t{z0}\t{B_fac}\t{B_fac_err}\n")
        if B_fac - B_fac_err > correct:
            z_max = z0
        elif B_fac + B_fac_err < correct:
            z_min = z0
    with open(f"data/horizon/horizon_{q}_{label_data}_{label_model}_new.dat", "a") as myfile:
        myfile.write(f"{seed}\t{mass}\t{z0}\t{B_fac}\t{B_fac_err}\n")

    return (mass, z0, B_fac)


from scipy import interpolate
def compute_horizon_masses(modes_data, modes_model, detector, q, num_procs = 7):
    pathlib.Path('data/horizon').mkdir(parents=True, exist_ok=True) 
    pathlib.Path('data/horizon/masses').mkdir(parents=True, exist_ok=True) 

    label_data = 'data'
    for mode in modes_data:
        label_data += '_'+mode[1]+mode[3]+mode[5]
    label_model = 'model'
    for mode in modes_model:
        label_model += '_'+mode[1]+mode[3]+mode[5]

    if not os.path.exists(f"data/horizon/horizon_{q}_{label_data}_{label_model}.dat"):
        with open(f"data/horizon/horizon_{q}_{label_data}_{label_model}.dat", "w") as myfile:
            myfile.write(f"#(0)seed(1)mass(2)redshift(3)logB(4)logBerr\n")

    mode1 = modes_data[0]
    mode2 = modes_data[1][:7]
    masses_rayleigh, redshifts_rayleigh = np.genfromtxt(f'data/horizon/rayleigh/rayleigh_horizon_{q}_{detector}_{mode1}+{mode2}_detector.dat').T
    itp_rayleigh = interpolate.interp1d(
        masses_rayleigh,
        redshifts_rayleigh,
        kind='cubic',
        )
    N = num_procs
    masses = np.logspace(np.log10(30), 3.5, N, endpoint=True)
    redshifts = []
    for mass in masses:
        try: 
            redshifts.append(itp_rayleigh(mass))
        except:
            redshifts.append(1e-2)
    seeds = np.random.randint(1e3,9e3, N)

    # values = tuple((seeds[i], masses[i], redshifts[i], modes_data, modes_model, detector, q, label_data, label_model) for i in range(len(masses)))
    values = [(seeds[i], masses[i], redshifts[i], modes_data, modes_model, detector, q, label_data, label_model) for i in range(len(masses))]
    # np.random.shuffle(values)

    i = 1#2#3#4#5#6#7#8#9#10#11#12#13#14#15#16#17#18#19#20#21#22#23#24#25#26#27#28#29#30#31#32#33#34#35#36#37#38#39#40#41#42#43#44#45#46#47#48
    i -= 1
    spectrocopy_horizon(seeds[i], masses[i], redshifts[i], modes_data, modes_model, detector, q, label_data, label_model)

    # processes = []
    # j = 0
    # while j+(num_procs-1) < len(masses):
    #     for i in range(j,j+num_procs):
    #         p = multiprocessing.Process(target=spectrocopy_horizon, args=(seeds[i], masses[i], redshifts[i], modes_data, modes_model, detector, q, label_data, label_model))
    #         p.start()
    #         processes.append(p)
            
    #     for process in processes:
    #         process.join()
    #     j += num_procs
    # if j < len(masses):
    #     for i in range(j,len(masses)):
    #         p = multiprocessing.Process(target=spectrocopy_horizon, args=(seeds[i], masses[i], redshifts[i], modes_data, modes_model, detector, q, label_data, label_model))
    #         p.start()
    #         processes.append(p)
            
    #     for process in processes:
    #         process.join()



    # values_multi = values

    # if len(values) > num_procs:
    #     values_multi = values[:num_procs]

    # with Pool() as pool:
    #     res = pool.starmap(spectrocopy_horizon, values_multi)

    # if len(values) - len(values_multi) > num_procs:
    #     values_multi = values[num_procs:2*num_procs]
    # else: values_multi = values[num_procs:]

    # with Pool() as pool:
    #     res = pool.starmap(spectrocopy_horizon, values_multi)

if __name__ == '__main__':
    """GW190521
    final mass = 150.3
    redshift = 0.72
    spectrocopy horizon = 0.148689
    """
    m_f = 150
    z = 0.01
    q = 1.5
    # np.random.seed(1234)
    # detector = "LIGO"
    # modes = ["(2,2,0)", "(2,2,1) I"]
    # modes = ["(2,2,0)"]
    # modes_model = ["(2,2,0)", "(2,2,1) I"]
    # modes_model = ["(2,2,0)"]
    # seed = 12345
    # teste = MultiNestSampler(modes, modes_model, True, detector, m_f, z, q, "EF", seed)
    # model = "freq_tau"
    # label = 'simulation'
    # label = 'qnm'

    # print(teste.compute_bayes_factor('freq_tau'))
    # teste.run_sampler(model, label)

    modes_model = ["(2,2,0)"]
    detector = "LIGO"
    # modes_data = ["(2,2,0)", "(2,2,1) II"]
    modes_data = ["(2,2,0)", "(3,3,0)"]
    # modes_data = ["(2,2,0)", "(4,4,0)"]
    # modes_data = ["(2,2,0)", "(2,1,0)"]
    q = 10
    cores = 16
    z_max = 1e-1
    z_min = 8e-3
    masses = np.logspace(np.log10(80), 3.5, 40, endpoint=True)
    for mass in masses:
        logB_redshift(mass, modes_data, modes_model, detector, q, cores, z_min, z_max)

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

    # num_procs = 48

    # np.random.seed(330)
    # modes_data = ["(2,2,0)", "(3,3,0)"]
    # modes_model = ["(2,2,0)"]
    # detector = "LIGO"
    # q = 1.5
    # compute_horizon_masses(modes_data, modes_model, detector, q, num_procs)

    # np.random.seed(210)
    # modes_data = ["(2,2,0)", "(2,1,0)"]
    # modes_model = ["(2,2,0)"]
    # detector = "LIGO"
    # q = 1.5
    # compute_horizon_masses(modes_data, modes_model, detector, q, num_procs)
    
    # np.random.seed(221)
    # modes_data = ["(2,2,0)", "(2,2,1) II"]
    # modes_model = ["(2,2,0)"]
    # detector = "LIGO"
    # q = 1.5
    # compute_horizon_masses(modes_data, modes_model, detector, q, num_procs)

    # np.random.seed(440)
    # modes_data = ["(2,2,0)", "(4,4,0)"]
    # modes_model = ["(2,2,0)"]
    # detector = "LIGO"
    # q = 1.5
    # compute_horizon_masses(modes_data, modes_model, detector, q, num_procs)
