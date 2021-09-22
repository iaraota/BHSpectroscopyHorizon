import os
import json
from datetime import datetime
import pathlib
import time

# from mpi4py import MPI

import scipy.stats as stats
from scipy import integrate

from multiprocessing import Pool, cpu_count

import multiprocessing
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas as pd

from pymultinest.solve import solve
import pymultinest
pathlib.Path(
    'data/multinest/chains').mkdir(parents=True, exist_ok=True)

from SourceData import SourceData
from modules import MCMCFunctions, GWFunctions
from Models import Models, TrueParameters, Priors


class MultiNestSampler(SourceData):
    """Compute posterior probability densities for
    quasinormal modes in the frequency domain."""

    def __init__(self, modes_data: list, modes_model: list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modes_data = modes_data
        self.modes_model = modes_model
        self.args = args
        self.kwargs = kwargs

        # construct self.data
        self.inject_data(self.modes_data)

        # construct model = data
        self.models = Models(self.modes_model, *args, **kwargs)
        self.true_pars = TrueParameters(self.modes_model, *args, **kwargs)
        self.priors = Priors(self.modes_model, *args, **kwargs)

        # construct different model
        self.models_data = Models(self.modes_data, *args, **kwargs)
        self.true_pars_data = TrueParameters(self.modes_data, *args, **kwargs)
        self.priors_data = Priors(self.modes_data, *args, **kwargs)

    def compute_bayes_factor(self, model: str, seed=np.random.get_state()[1][0],):
        # this method compute the log of the Bayes factor of a model compatible with
        # the data over a given model. Ideally the Bayes factor should be greater than
        # 1 as the data model is correct.
        mass = self.final_mass

        label_data = 'data'
        for mode in self.modes_data:
            label_data += '_' + mode[1] + mode[3] + mode[5]
        label_model = 'model'
        for mode in self.modes_model:
            label_model += '_' + mode[1] + mode[3] + mode[5]

        detector_label = self.detector['label'].split(' ')[0]

        # create multinest folder to save multinest files
        file_path = f'data/multinest/chains/{detector_label}_{self.q_mass}_{label_data}_{label_model}_mass_{round(mass,1)}_redshift_{self.redshift}_seed_{seed}/'
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

        # compute evidence for the model
        self.true_pars.choose_theta_true(model)
        self.priors.cube_uniform_prior(model)
        self.models.choose_model(model)
        ndim = len(self.true_pars.theta_true)

        result_model = solve(
            LogLikelihood=lambda theta: self.loglikelihood(
                self.models.model, theta),
            Prior=self.priors.prior_function,
            n_dims=ndim,
            n_live_points=max(500, ndim * 50),
            outputfiles_basename=file_path + 'model_',
            verbose=False,
            use_MPI=False,
        )
        logZ_model = result_model['logZ']
        logZerr_model = result_model['logZerr']

        # compute evidence for the data model
        self.true_pars_data.choose_theta_true(model)
        self.priors_data.cube_uniform_prior(model)
        self.models_data.choose_model(model)

        ndim_data = len(self.true_pars_data.theta_true)

        result_data = solve(
            LogLikelihood=lambda theta: self.loglikelihood(
                self.models_data.model, theta),
            Prior=self.priors_data.prior_function,
            n_dims=ndim_data,
            n_live_points=max(500, 50 * ndim),
            outputfiles_basename=file_path + 'data_',
            verbose=False,
            use_MPI=False,
        )
        logZ_data = result_data['logZ']
        logZerr_data = result_data['logZerr']

        logB = logZ_data - logZ_model
        logBerr = np.sqrt(logZerr_data**2 + logZerr_model**2)

        return logZ_model, logZ_data, logB, logBerr

    def compute_bayes_factor_multi_modes(
        self,
        N_modes: int=3,
    ):
        # This method compute the Bayes factor of models with multiple modes.
        # The data should contain a sum of the mos relevant modes
        # The large the number of modes in the model modes
        # the longer it will take to compute the evidences.

        # cannot change the model, beucause of the prior
        model = 'freq_tau_multi'
        combinations = [[self.modes_model[0]]]

        N_modes = min(len(self.modes_model), N_modes)

        for i in range(2, N_modes + 1):
            combinations.append(self.modes_model[:i])

        mass = self.final_mass

        label_data = 'data'
        for mode in self.modes_data:
            label_data += '_' + mode[1] + mode[3] + mode[5]

        detector_label = self.detector['label'].split(' ')[0]

        # create dictionaries to save the vidence and its errors
        logZ = {}
        logZerr = {}
        # loop through models with N = 1 to N = number of modes
        for modes_model in combinations:
            # create models
            models = Models(modes_model, *self.args, **self.kwargs)
            true_pars = TrueParameters(modes_model, *self.args, **self.kwargs)
            priors = Priors(modes_model, *self.args, **self.kwargs)

            true_pars.choose_theta_true(model)
            priors.cube_uniform_prior(model)
            models.choose_model(model)
            ndim = len(true_pars.theta_true)

            n = len(modes_model)
            label_model = f'model_{n}modes'

            # this seed is only to save a different folder, the noise is already created
            # and the data should be the same for all the models
            seed = np.random.get_state()[1][0]

            # create folder for multinest files
            file_path = f'data/multinest/chains/{detector_label}_{self.q_mass}_{label_data}_{label_model}_mass_{round(mass,1)}_redshift_{self.redshift}_seed_{seed}/'
            pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

            result = solve(
                LogLikelihood=lambda theta: self.loglikelihood(
                    models.model, theta),
                Prior=priors.prior_function,
                n_dims=ndim,
                n_live_points=max(500, 50 * ndim),
                outputfiles_basename=file_path + 'model_',
                verbose=False,
                use_MPI=False,
            )

            # save evidence in the dictionary
            logZ[n] = result['logZ']
            logZerr[n] = result['logZerr']

        # compute logB_{n-1}^n
        logB = {}
        logBerr = {}
        for i in range(N_modes, 1, -1):
            logB[i] = logZ[i] - logZ[i - 1]
            logBerr[i] = np.sqrt(logZerr[i]**2 + logZerr[i - 1]**2)

        return logZ, logZerr, logB, logBerr

    def compute_parameters_multi_modes(
        self,
        label='multi',
    ):
        # This will estimate the parameters (and compute the evidence) of
        # a chosen model with N modes. This is used to estimate the parameters
        # at a specific distance, usually the sprectroscopy horizon for this model.

        # cannot be another model
        model = 'freq_tau_multi'

        mass = self.final_mass
        modes_model = self.modes_model

        label_data = 'data'
        for mode in self.modes_data:
            label_data += '_' + mode[1] + mode[3] + mode[5]
        detector_label = self.detector['label'].split(' ')[0]

        models = Models(modes_model, *self.args, **self.kwargs)
        true_pars = TrueParameters(modes_model, *self.args, **self.kwargs)
        priors = Priors(modes_model, *self.args, **self.kwargs)

        true_pars.choose_theta_true(model)
        priors.cube_uniform_prior(model)
        models.choose_model(model)
        ndim = len(true_pars.theta_true)

        n = len(modes_model)
        label_model = f'model_{n}'

        detector_label = self.detector['label'].split(' ')[0]

        seed = np.random.get_state()[1][0]
        file_path = f'data/multinest/chains/pars_{detector_label}_{self.q_mass}_{label_data}_{label_model}_mass_{round(mass,1)}_redshift_{self.redshift}_seed_{seed}/'
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        result = solve(
            LogLikelihood=lambda theta: self.loglikelihood(
                models.model, theta),
            Prior=priors.prior_function,
            n_dims=ndim,
            n_live_points=max(500, 50 * ndim),
            outputfiles_basename=file_path + 'multi-',
            verbose=False,
            use_MPI=False,
        )

        end_time = time.time()
        # Save parameters
        self.multi_save_injected_deviation(
            result['samples'],
            true_pars.theta_labels_plain,
            true_pars.theta_true,
            self.modes_model,
            self.modes_data,
            label,
        )

        path_data = 'data/'
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        file_path_time = f"{path_data}elapsed_time.dat"
        if not pathlib.Path(file_path_time).is_file():
            with open(file_path_time, "w") as myfile:
                myfile.write(f"#(0)mass(1)redshift(2)elapsed_time\n")

        with open(file_path_time, "a") as myfile:
            myfile.write(f"{self.final_mass}\t")
            myfile.write(f"{self.redshift}\t")
            myfile.write(f"{end_time - start_time}\n")

        return result

    def run_sampler(
            self,
            model: str,
            label: str,):
        # test running multinest for 1 case, this computes the evidence but do not compute the Bayes factor
        from time import time               # use for timing functions
        self.true_pars.choose_theta_true(model)
        self.priors.cube_uniform_prior(model)
        self.models.choose_model(model)
        print(self.true_pars.theta_true)
        ndim = len(self.true_pars.theta_true)
        t0 = time()
        seed = np.random.get_state()[1][0]
        result = solve(
            LogLikelihood=lambda theta: self.loglikelihood(
                self.models.model, theta),
            Prior=self.priors.prior_function,
            n_dims=ndim,
            n_live_points=500,
            outputfiles_basename=f'data/multinest/chains/{label}-{seed}-',
            verbose=True,
            use_MPI=False,
        )
        t1 = time()
        print()
        print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
        print()
        print('parameter values:')
        parameters = self.true_pars.theta_labels_plain
        for name, col in zip(parameters, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
            print(
                f'percentil: {np.percentile(col,50)}+{np.percentile(col,90) - np.percentile(col,50)}-{np.percentile(col,10) - np.percentile(col,50)}')
        print(f'\ntotal time:{t1-t0}')
        print('trues:', self.true_pars.theta_true)
        # make marginal plots by running:
        # $ python multinest_marginals.py chains/3-
        # For that, we need to store the parameter names:
        corner.corner(result['samples'], quantiles=[
                      0.05, 0.5, 0.95], truths=self.true_pars.theta_true, show_titles=True)
        # plt.show()

    def loglikelihood(self, model, theta: list):
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

    def multi_save_injected_deviation(
            self,
            samples,
            label_pars,
            true_pars,
            modes_model,
            modes_data,
            label='norm_psd'):

        df_samples = pd.DataFrame(samples, columns=label_pars)

        trues = {}
        for i in range(len(label_pars)):
            trues[label_pars[i]] = true_pars[i]
        label_data_modes = ''
        for mode in modes_data:
            label_data_modes += '_' + mode[1] + mode[3] + mode[5]

        label_model_modes = ''
        for mode in modes_model:
            label_model_modes += '_' + mode[1] + mode[3] + mode[5]

        path = 'data/samples_pars/norm_psd'
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        for par in label_pars:
            file_path = f"{path}/{label}_{self.q_mass}_data{label_data_modes}_model{label_model_modes}_par_{par}.dat"
            pos = sorted(df_samples[par].values)
            kde_pos = stats.gaussian_kde(pos)

            errors = {}
            if par in ['A_220', 'phi_220', 'freq_220', 'tau_220']:
                inj_x = trues[par]
                int_inj = integrate.quad(kde_pos, -np.inf, inj_x)[0]
                errors['220'] = stats.norm.ppf(int_inj)

                if not pathlib.Path(file_path).is_file():
                    with open(file_path, "w") as myfile:
                        myfile.write('#(0)mass(1)redshift(2)error-220\n')

                with open(file_path, "a") as myfile:
                    myfile.write(f"{self.final_mass}\t")
                    myfile.write(f"{self.redshift}\t")
                    myfile.write(f"{errors['220']}\n")

            else:
                for mode in ['(2,2,1) II', '(3,3,0)', '(4,4,0)', '(2,1,0)', '(2,2,0)']:
                    inj_x = trues[par][mode]
                    int_inj = integrate.quad(
                        kde_pos, -np.inf, inj_x)[0]
                    errors[mode] = stats.norm.ppf(int_inj)

                if not pathlib.Path(file_path).is_file():
                    with open(file_path, "w") as myfile:
                        myfile.write(
                            '#(0)mass(1)redshift(2)error-221(3)error-330(4)error-440(5)error-210(6)error-220\n')

                with open(file_path, 'a') as file:
                    file.write(f'{self.final_mass}\t')
                    file.write(f'{self.redshift}\t')
                    file.write(f'{errors["(2,2,1) II"]}\t')
                    file.write(f'{errors["(3,3,0)"]}\t')
                    file.write(f'{errors["(4,4,0)"]}\t')
                    file.write(f'{errors["(2,1,0)"]}\t')
                    file.write(f'{errors["(2,2,0)"]}\n')


def compute_log_B(modes_data, modes_model, detector, mass, redshift, q, seed, label='multi'):
    noise_seed = np.random.seed(seed)

    save_seed = np.random.get_state()[1][0]
    multi_sampler = MultiNestSampler(
        modes_data, modes_model, detector, mass, redshift, q, "FH", noise_seed)
    result = multi_sampler.compute_parameters_multi_modes(label)
    return result


def one_mode_bayes_histogram(modes_data, modes_model, detector, num, q, cores=4):
    label_data = 'logZ: ' + modes_data[0]
    label_model = 'logZ:'
    for mode in modes_model:
        label_model += ' ' + mode

    masses = np.random.choice(np.power(10, np.linspace(
        1, np.log10(5 * 10**3), num * 10)), num, replace=False)
    redshifts = np.random.choice(
        np.power(10, np.linspace(-2, 0, num * 10)), num, replace=False)
    seeds = np.random.randint(1, 1e4, num)

    pathlib.Path('data/histogram').mkdir(parents=True, exist_ok=True)
    with open(f"data/histogram/freq_tau_histogram_{q}_{modes_data}_{modes_model}_{num}.dat", "w") as myfile:
        myfile.write(
            f"#(0)seed(1)mass(2)redshift(3){label_data}(4){label_model}(5)logB\n")

    values = [(modes_data, modes_model, detector, masses[i],
               redshifts[i], q, num, seeds[i]) for i in range(len(redshifts))]
    with Pool(processes=cores) as pool:
        res = pool.starmap(compute_log_B, values)


def find_logB(redshift, modes_data, modes_model, detector, mass, q, noise_seed):
    dy_sampler = MultiNestSampler(
        modes_data, modes_model, detector, mass, redshift, q, "FH", noise_seed)
    logZ_model, logZ_data, logB, logBerr = dy_sampler.compute_bayes_factor(
        'freq_tau')
    return logB, logBerr

def single_logB_redshift(redshift, modes_data, modes_model, detector, mass, q, noise_seed):

    label_data = 'data'
    for mode in modes_data:
        label_data += '_' + mode[1] + mode[3] + mode[5]
    label_model = 'model'
    for mode in modes_model:
        label_model += '_' + mode[1] + mode[3] + mode[5]

    file_path = f'data/horizon/logB_redshift/{detector}_{q}_{label_data}_{label_model}/{mass}.dat'

    B_fac, B_fac_err = find_logB(
        redshift, modes_data, modes_model, detector, mass, q, noise_seed)

    with open(file_path, "a") as myfile:
        myfile.write(
            f"{noise_seed}\t{mass}\t{redshift}\t{B_fac}\t{B_fac_err}\n")

    return redshift, B_fac, B_fac_err


def compute_logB_2modes(q, sub_mode, detector, masses, redshifts, cores=4):
    # Compute the Bayes factor for model containing 220+lmn

    modes_model = ["(2,2,0)"]
    modes_data = ["(2,2,0)", sub_mode]

    pathlib.Path('data/horizon').mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        'data/horizon/logB_redshift').mkdir(parents=True, exist_ok=True)

    label_data = 'data'
    for mode_data in modes_data:
        label_data += '_' + mode_data[1] + mode_data[3] + mode_data[5]
    label_model = 'model'
    for mode_model in modes_model:
        label_model += '_' + \
            mode_model[1] + mode_model[3] + mode_model[5]

    mode_folder = f'data/horizon/logB_redshift/{detector}_{q}_{label_data}_{label_model}'
    pathlib.Path(mode_folder).mkdir(parents=True, exist_ok=True)

    seeds = np.random.randint(1e3, 9e3, 10000)
    values = [(
        redshifts[mass] * x,
        modes_data,
        modes_model,
        detector,
        mass,
        q,
        np.random.choice(seeds)
    )
        for mass in masses for x in np.linspace(0.5, 1.5, 10)
    ]
    for mass in masses:
        if not os.path.exists(mode_folder + f"/{mass}.dat"):
            with open(mode_folder + f"/{mass}.dat", "w") as myfile:
                myfile.write(
                    f"#(0)seed(1)mass(2)redshift(3)logB(4)logBerr\n")

    with Pool(processes=cores) as pool:
        res = pool.starmap(single_logB_redshift, values)


if __name__ == '__main__':
    detector = "CE"
    mass_ratio = 10
    cores = 48

    N_masses = 10
    N_redshifts = 10

    horizons_coeffs = {
        1.5: {
            '(2,2,1) II': [-0.37701513, 1.63425174, -0.26873549, -2.9867969],
            '(3,3,0)': [-0.3247828, 1.19301331, 1.10440705, -4.70486748],
            '(4,4,0)': [-0.56241913, 3.19875007, -3.76132209, -1.75035348],
            '(2,1,0)': [-0.54952892, 2.92598436, -3.4023145, -1.27849752],
        },
        10: {
            '(2,2,1) II': [-0.37701513, 1.63425174, -0.26873549, -2.9867969],
            '(3,3,0)': [-0.3247828, 1.19301331, 1.10440705, -4.70486748],
            '(4,4,0)': [-0.56241913, 3.19875007, -3.76132209, -1.75035348],
            '(2,1,0)': [-0.54952892, 2.92598436, -3.4023145, -1.27849752],
        }
    }
    masses = np.logspace(np.log10(10), np.log10(1e4), N_masses, endpoint=True)
    redshifts = np.logspace(np.log10(1e-2), np.log10(1e1),
                            N_redshifts, endpoint=True)

    # modes = ["(2,2,1) II", "(3,3,0)", "(4,4,0)", "(2,1,0)"]
    modes = ["(2,2,1) II"]
    # modes = ["(3,3,0)"]
    # modes = ["(4,4,0)"]
    # modes = ["(2,1,0)"]

    for mode in modes:

        horizon = np.poly1d(horizons_coeffs[mass_ratio][mode])
        redshifts_horizon = {
            mass: 10**horizon(np.log10(mass)) for mass in masses}
        # mass = masses[0]
        # mass = masses[1]
        # mass = masses[2]
        # mass = masses[3]
        # mass = masses[4]
        # mass = masses[5]
        # mass = masses[6]
        # mass = masses[7]
        # mass = masses[8]
        # mass = masses[9]
        # mass = masses[9]
        # mass = masses[10]
        # mass = masses[11]
        # mass = masses[12]
        # mass = masses[13]
        # mass = masses[14]
        # mass = masses[15]
        # mass = masses[16]
        # mass = masses[17]
        # mass = masses[18]
        # mass = masses[19]
        # mass = masses[20]
        # mass = masses[21]
        # mass = masses[22]
        # mass = masses[23]
        # mass = masses[24]
        # mass = masses[25]
        # mass = masses[26]
        # mass = masses[27]
        # mass = masses[28]
        # mass = masses[29]
        # mass = masses[30]
        # mass = masses[31]
        # mass = masses[32]
        # mass = masses[33]
        # mass = masses[34]
        # mass = masses[35]
        # mass = masses[36]
        # mass = masses[37]
        # mass = masses[38]
        # mass = masses[39]

        compute_logB_2modes(mass_ratio, mode, detector,
                            masses, redshifts_horizon, cores)
