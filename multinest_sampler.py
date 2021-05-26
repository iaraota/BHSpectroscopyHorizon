import os
import json
from datetime import datetime
import pathlib

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

    def __init__(self, modes_data: list, modes_model: list, *args, **kwargs):
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
        model: str,
        seed=np.random.get_state()[1][0],
    ):
        self.true_pars.choose_theta_true(model)
        self.priors.cube_uniform_prior(model)
        self.models.choose_model(model)
        mass = self.final_mass
        ndim = len(self.true_pars.theta_true)

        label_data = 'data'
        for mode in self.modes_data:
            label_data += '_' + mode[1] + mode[3] + mode[5]
        label_model = 'model'
        for mode in self.modes_model:
            label_model += '_' + mode[1] + mode[3] + mode[5]

        file_path = f'data/multinest/chains/{self.q_mass}_{label_data}_{label_model}_mass_{round(mass,1)}_redshift_{self.redshift}_seed_{seed}/'
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

        result_model = solve(
            LogLikelihood=lambda theta: self.loglikelihood(
                self.models.model, theta),
            Prior=self.priors.prior_function,
            n_dims=ndim,
            n_live_points=500,
            outputfiles_basename=file_path + 'model-',
            verbose=False,
        )
        logZ_model = result_model['logZ']
        logZerr_model = result_model['logZerr']

        self.true_pars_data.choose_theta_true(model)
        self.priors_data.cube_uniform_prior(model)
        self.models_data.choose_model(model)

        ndim_data = len(self.true_pars_data.theta_true)

        result_data = solve(
            LogLikelihood=lambda theta: self.loglikelihood(
                self.models_data.model, theta),
            Prior=self.priors_data.prior_function,
            n_dims=ndim_data,
            n_live_points=500,
            outputfiles_basename=file_path + 'data-',
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
        N_modes: int=3,
    ):
        model = 'freq_tau_multi'
        combinations = [[self.modes_model[0]]]

        N_modes = min(len(self.modes_model), N_modes)

        for i in range(2, N_modes + 1):
            combinations.append(self.modes_model[:i])

        mass = self.final_mass

        label_data = 'data'
        for mode in self.modes_data:
            label_data += '_' + mode[1] + mode[3] + mode[5]

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
            label_model = f'model_{n}modes'

            seed = np.random.get_state()[1][0]
            file_path = f'data/multinest/chains/{self.q_mass}_{label_data}_{label_model}_mass_{round(mass,1)}_redshift_{self.redshift}_seed_{seed}/'
            pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

            result = solve(
                LogLikelihood=lambda theta: self.loglikelihood(
                    models.model, theta),
                Prior=priors.prior_function,
                n_dims=ndim,
                n_live_points=max(500, 50 * ndim),
                outputfiles_basename=file_path + 'model-',
                verbose=False,
            )
            logZ[n] = result['logZ']
            logZerr[n] = result['logZerr']

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
        model = 'freq_tau_multi'

        mass = self.final_mass
        modes_model = self.modes_model

        label_data = 'data'
        for mode in self.modes_data:
            label_data += '_' + mode[1] + mode[3] + mode[5]

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
        file_path = f'data/multinest/chains/pars_{self.q_mass}_{label_data}_{label_model}_mass_{round(mass,1)}_redshift_{self.redshift}_seed_{seed}/'
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

        result = solve(
            LogLikelihood=lambda theta: self.loglikelihood(
                models.model, theta),
            Prior=priors.prior_function,
            n_dims=ndim,
            n_live_points=max(500, 50 * ndim),
            outputfiles_basename=file_path + 'multi-',
            verbose=False,
        )

        # Save parameters
        self.multi_save_injected_deviation(
            result['samples'],
            true_pars.theta_labels_plain,
            true_pars.theta_true,
            self.modes_model,
            self.modes_data,
            label,
        )
        return result

    def run_sampler(
            self,
            model: str,
            label: str,):
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

    def save_estimated_values_and_errors(
            self,
            samples,
            label_pars,
            true_pars,
            modes_model,
            modes_data,
            logB,
            logBerr,
            label='pars'):

        df_samples = pd.DataFrame(samples, columns=label_pars)

        trues = {}
        for i in range(len(label_pars)):
            trues[label_pars[i]] = true_pars[i]

        path = 'data/samples_pars'
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        label_data_modes = ''
        for mode in modes_data:
            label_data_modes += '_' + mode[1] + mode[3] + mode[5]

        label_model_modes = ''
        for mode in modes_model:
            label_model_modes += '_' + mode[1] + mode[3] + mode[5]

        for parameter in label_pars:
            file_path = f"{path}/{label}_{self.q_mass}_data{label_data_modes}_model{label_model_modes}_par_{parameter}.dat"
            if pathlib.Path(file_path).is_file():
                with open(file_path, "a") as myfile:
                    myfile.write(f"{self.final_mass}\t")
                    myfile.write(f"{self.redshift}\t")
                    myfile.write(f"{logB}\t")
                    myfile.write(f"{logBerr}\t")
                    myfile.write(f"{trues[parameter]}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.5)}\t")
                    myfile.write(
                        f"{df_samples[parameter].quantile(.84135)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(
                        f"{df_samples[parameter].quantile(.15865)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(
                        f"{df_samples[parameter].quantile(.97725)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(
                        f"{df_samples[parameter].quantile(.2275)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(
                        f"{df_samples[parameter].quantile(.99865)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(
                        f"{df_samples[parameter].quantile(.00135)-df_samples[parameter].quantile(.5)}\n")
            else:
                with open(file_path, "w") as myfile:
                    myfile.write(
                        f"#(0)mass(1)redshift(2)logB(3)logBerr(4)true(5)estimated(50%)")
                    myfile.write(
                        f'(6)upper-1sigma(7)lower-1sigma(8)upper-2sigma(9)lower-2sigma(10)upper-3sigma(11)lower-3sigma\n')
                    myfile.write(f"{self.final_mass}\t")
                    myfile.write(f"{self.redshift}\t")
                    myfile.write(f"{logB}\t")
                    myfile.write(f"{logBerr}\t")
                    myfile.write(f"{trues[parameter]}\t")
                    myfile.write(f"{df_samples[parameter].quantile(.5)}\t")
                    myfile.write(
                        f"{df_samples[parameter].quantile(.84135)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(
                        f"{df_samples[parameter].quantile(.15865)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(
                        f"{df_samples[parameter].quantile(.97725)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(
                        f"{df_samples[parameter].quantile(.2275)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(
                        f"{df_samples[parameter].quantile(.99865)-df_samples[parameter].quantile(.5)}\t")
                    myfile.write(
                        f"{df_samples[parameter].quantile(.00135)-df_samples[parameter].quantile(.5)}\n")

    def multi_save_estimated_values_and_errors(
            self,
            samples,
            label_pars,
            true_pars,
            modes_model,
            modes_data,
            label='pars'):

        df_samples = pd.DataFrame(samples, columns=label_pars)

        trues = {}
        for i in range(len(label_pars)):
            trues[label_pars[i]] = true_pars[i]
        print(trues)

        trues_label = {}
        trues_keys = {}
        for key, value in trues.items():
            if isinstance(value, dict):
                trues_keys[key] = list(value.keys())
                trues_label[key] = f'true-{trues_keys[key]}'
            else:
                trues_label[key] = 'true'

        for key in trues_keys.keys():
            aux = trues[key].values()
            trues[key] = ''
            for k in aux:
                trues[key] += f'{k}\t'
            trues[key] = trues[key][:-1]

        path = 'data/samples_pars'
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        label_data_modes = ''
        for mode in modes_data:
            label_data_modes += '_' + mode[1] + mode[3] + mode[5]

        label_model_modes = ''
        for mode in modes_model:
            label_model_modes += '_' + mode[1] + mode[3] + mode[5]

        for parameter in label_pars:
            file_path = f"{path}/{label}_{self.q_mass}_data{label_data_modes}_model{label_model_modes}_par_{parameter}.dat"
            if not pathlib.Path(file_path).is_file():
                with open(file_path, "w") as myfile:
                    myfile.write(
                        f"#(0)mass(1)redshift(2){trues_label[parameter]}(3)estimated(50%)")
                    myfile.write(
                        f'(4)upper-1sigma(5)lower-1sigma(6)upper-2sigma(7)lower-2sigma(8)upper-3sigma(9)lower-3sigma\n')

            with open(file_path, "a") as myfile:
                myfile.write(f"{self.final_mass}\t")
                myfile.write(f"{self.redshift}\t")
                myfile.write(f"{trues[parameter]}\t")
                myfile.write(f"{df_samples[parameter].quantile(.5)}\t")
                myfile.write(
                    f"{df_samples[parameter].quantile(.84135)-df_samples[parameter].quantile(.5)}\t")
                myfile.write(
                    f"{df_samples[parameter].quantile(.15865)-df_samples[parameter].quantile(.5)}\t")
                myfile.write(
                    f"{df_samples[parameter].quantile(.97725)-df_samples[parameter].quantile(.5)}\t")
                myfile.write(
                    f"{df_samples[parameter].quantile(.2275)-df_samples[parameter].quantile(.5)}\t")
                myfile.write(
                    f"{df_samples[parameter].quantile(.99865)-df_samples[parameter].quantile(.5)}\t")
                myfile.write(
                    f"{df_samples[parameter].quantile(.00135)-df_samples[parameter].quantile(.5)}\n")

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


def multimodes_logB_redshift(modes_data, modes_model, detector, q, N_modes=2, cores=16):
    horizons_coeffs = {
        1.5: {
            2: [-0.46587733, 2.70800683, - 3.90075654, - 0.7769546],
            3: [-0.50953798, 3.14680919, - 5.29445604, 0.20208096],
        },
        10: {
            2: [-0.37815505, 1.89320109, - 1.767064, - 2.69569894],
            3: [-0.72004788, 4.79212836, - 9.4927626, 3.30517209],
        }
    }
    masses_range = {
        1.5: {
            2: [3e1, 5e3, 50],
            3: [3e1, 5e3, 50],
        },
        10: {
            2: [5e1, 4e3, 45],
            3: [1e2, 4e3, 20],
        }

    }

    # N_modes = len(modes_model)
    horizon = np.poly1d(horizons_coeffs[q][N_modes])
    masses = np.logspace(np.log10(masses_range[q][N_modes][0]), np.log10(
        masses_range[q][N_modes][1]), masses_range[q][N_modes][2], endpoint=True)
    seeds = np.random.randint(1e3, 9e3, 10000)
    values = [(
        10**horizon(np.log10(mass)) * x,
        modes_data,
        modes_model,
        detector,
        mass,
        q,
        np.random.choice(seeds),
        N_modes,
    )
        for mass in masses for x in np.linspace(0.5, 1.5, 10)
    ]
    plt.loglog(masses, 10**horizon(np.log10(masses)))
    plt.show()

    pathlib.Path('data/horizon').mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        'data/horizon/logB_redshift').mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        'data/horizon/logB_redshift/multimode').mkdir(parents=True, exist_ok=True)

    label_data = 'data'
    for mode in modes_data:
        label_data += '_' + mode[1] + mode[3] + mode[5]

    num_modes = 0
    for mode in modes_model:
        num_modes += 1
    label_model = f'model_{num_modes}_modes'

    mode_folder = f'data/horizon/logB_redshift/multimode-2/{q}_{label_data}_{label_model}'
    pathlib.Path(mode_folder).mkdir(parents=True, exist_ok=True)
    for mass in masses:
        if not os.path.exists(mode_folder + f"/{mass}.dat"):
            with open(mode_folder + f"/{mass}.dat", "w") as myfile:
                myfile.write(f"#(0)seed(1)mass(2)redshift")
                for i in range(2, N_modes + 1):
                    myfile.write(
                        f"({1+i})logB-{i}modes({2+i})logBerr-{i}modes")
                myfile.write("\n")

        if not os.path.exists(mode_folder + f"/logZ-{mass}.dat"):
            with open(mode_folder + f"/logZ-{mass}.dat", "w") as myfile:
                myfile.write(f"#(0)seed(1)mass(2)redshift")
                for i in range(1, N_modes + 1):
                    myfile.write(
                        f"({2+i})logZ-{i}modes({3+i})logBerr-{i}modes")
                myfile.write("\n")

    with Pool(processes=cores) as pool:
        res = pool.starmap(multi_logB_redshift, values)


def multi_logB_redshift(redshift, modes_data, modes_model, detector, mass, q, noise_seed, N_modes):
    label_data = 'data'
    for mode in modes_data:
        label_data += '_' + mode[1] + mode[3] + mode[5]

    num_modes = 0
    for mode in modes_model:
        num_modes += 1
    label_model = f'model_{num_modes}_modes'

    file_path = f'data/horizon/logB_redshift/multimode-2/{q}_{label_data}_{label_model}/{mass}.dat'
    file_path_logZ = f'data/horizon/logB_redshift/multimode-2/{q}_{label_data}_{label_model}/logZ-{mass}.dat'

    sampler = MultiNestSampler(
        modes_data, modes_model, detector, mass, redshift, q, "FH", noise_seed)
    logZ, logZerr, logB, logBerr = sampler.compute_bayes_factor_multi_modes(
        N_modes)
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


def compute_log_B(modes_data, modes_model, detector, mass, redshift, q, seed, label='multi'):
    noise_seed = np.random.seed(seed)

    save_seed = np.random.get_state()[1][0]
    multi_sampler = MultiNestSampler(
        modes_data, modes_model, detector, mass, redshift, q, "FH", noise_seed)
    result = multi_sampler.compute_parameters_multi_modes(label)
    return result
    # with open(f"data/histogram/freq_tau_histogram_{q}_{modes_data}_{modes_model}_{num}.dat", "a") as myfile:
    #     myfile.write(
    #         f"{seed}\t{mass}\t{redshift}\t{logZ_data}\t{logZ_model}\t{logB}\n")


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

    # for i in range(len(masses)):
    #     compute_log_B(modes_data, modes_model, detector, masses[i], redshifts[i], q, num, seeds[i])

    values = [(modes_data, modes_model, detector, masses[i],
               redshifts[i], q, num, seeds[i]) for i in range(len(redshifts))]
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

    file_path = f'data/horizon/logB_redshift/new_{q}_{label_data}_{label_model}/{mass}.dat'

    B_fac, B_fac_err = find_logB(
        redshift, modes_data, modes_model, detector, mass, q, noise_seed)

    with open(file_path, "a") as myfile:
        myfile.write(
            f"{noise_seed}\t{mass}\t{redshift}\t{B_fac}\t{B_fac_err}\n")

    return redshift, B_fac, B_fac_err


def logB_redshift(mass, modes_data, modes_model, detector, q, cores=4, z_min=1e-2, z_max=5e-1):
    N = cores
    pathlib.Path('data/horizon').mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        'data/horizon/logB_redshift').mkdir(parents=True, exist_ok=True)

    label_data = 'data'
    for mode in modes_data:
        label_data += '_' + mode[1] + mode[3] + mode[5]
    label_model = 'model'
    for mode in modes_model:
        label_model += '_' + mode[1] + mode[3] + mode[5]

    mode_folder = f'data/horizon/logB_redshift/{q}_{label_data}_{label_model}'
    pathlib.Path(mode_folder).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(mode_folder + f"/{mass}.dat"):
        with open(mode_folder + f"/{mass}.dat", "w") as myfile:
            myfile.write(f"#(0)seed(1)mass(2)redshift(3)logB(4)logBerr\n")
    num = 25
    # redshifts =  np.logspace(np.log10(z_min), np.log10(z_max), N, endpoint=True)
    redshifts = np.random.choice(
        np.power(10, np.linspace(-2, 0, num * 10)), num, replace=False)
    seeds = np.random.randint(1e3, 9e3, N)
    values = [(redshifts[i], modes_data, modes_model, detector,
               mass, q, seeds[0]) for i in range(len(redshifts))]
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


def sample_parameters(cores=4):
    # fitted coefficients 2 modes bayes horizon
    horizons_coeffs = {
        1.5: {
            '(2,2,1) II': [-0.59790336,  3.51790636, - 5.48829357,  0.19168019],
            '(3,3,0)': [-0.53512716,  3.10947882, - 4.57448291, - 0.74713459],
            '(4,4,0)': [-0.50520507,  3.16990084, - 5.20025844, - 0.50965154],
            '(2,1,0)': [-0.07565346, - 0.48994185,  4.24017978, - 8.06686243],
        },
        10: {
            '(2,2,1) II': [-0.92650429,   6.49717926, - 14.184255,     7.6065047],
            '(3,3,0)': [-0.79401161,  4.97300019, - 9.17977374,  2.98709633],
            '(4,4,0)': [-0.86390609,   5.71336636, - 11.27759125,   4.55742276],
            '(2,1,0)': [4.27004559, - 35.63003256,  98.30067291, - 91.76135004],
        }
    }
    masses_range = {
        1.5: {
            '(2,2,1) II': [4e1, 4e3, 40],
            '(3,3,0)': [6e1, 4e3, 36],
            '(4,4,0)': [1e2, 5e3, 30],
            '(2,1,0)': [1e2, 2e3, 22],
        },
        10: {
            '(2,2,1) II': [1e2, 2e3, 22],
            '(3,3,0)': [7e1, 3e3, 32],
            '(4,4,0)': [1e2, 3e3, 24],
            '(2,1,0)': [2.5e2, 4.5e2, 10],
        }

    }

    modes_model = ["(2,2,0)"]
    detector = "LIGO"
    modes = ['(3,3,0)', '(4,4,0)', '(2,1,0)', '(2,2,1) II', ]
    qs = [1.5, 10]
    for q in qs:
        for mode in modes:
            modes_data = ["(2,2,0)", mode]

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

            mode_folder = f'data/horizon/logB_redshift/new_{q}_{label_data}_{label_model}'
            pathlib.Path(mode_folder).mkdir(parents=True, exist_ok=True)

            horizon = np.poly1d(horizons_coeffs[q][mode])
            masses = np.logspace(np.log10(masses_range[q][mode][0]), np.log10(
                masses_range[q][mode][1]), masses_range[q][mode][2], endpoint=True)
            seeds = np.random.randint(1e3, 9e3, 10000)
            values = [(
                10**horizon(np.log10(mass)) * x,
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
    # sample_parameters()
    """GW190521
    final mass = 150.3
    redshift = 0.72
    spectrocopy horizon = 0.148689
    """
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
    # teste = MultiNestSampler(modes_data, modes_model,
    #                          detector, m_f, z, q, "FH", seed)

    # print(teste.compute_bayes_factor_multi_modes('freq_tau'))
    # label = 'teste'
    # teste.run_sampler(model, label)

    # modes_model = ["(2,2,0)"]
    # detector = "LIGO"
    # modes_data = ["(2,2,0)", "(2,2,1) II"]
    # # modes_data = ["(2,2,0)", "(3,3,0)"]
    # # modes_data = ["(2,2,0)", "(4,4,0)"]
    # # modes_data = ["(2,2,0)", "(2,1,0)"]
    # q = 1.5
    # cores = 4
    # z_max = 1e-1
    # z_min = 8e-3
    # masses = np.logspace(np.log10(80), 3.5, 40, endpoint=True)
    # num = 25
    # for mass in masses:
    #     logB_redshift(mass, modes_data, modes_model,
    #                   detector, q, cores, z_min, z_max)

    # # multimode horizon
    # modes_data = ["(2,2,0)", "(2,2,1) II", "(3,3,0)", "(4,4,0)", "(2,1,0)"]
    # detector = "LIGO"
    # modes_models = [["(2,2,0)", "(2,2,1) II", '(3,3,0)']]
    # qs = [1.5, 10]
    # cores = 48
    # N_modes = 2
    # for q in qs:
    #     for model in modes_models:
    #         multimodes_logB_redshift(
    #             modes_data, model, detector, q, N_modes, cores)

    # #histogram

    # #parameters multimode horizon
    modes_data = ["(2,2,0)", "(2,2,1) II", "(3,3,0)", "(4,4,0)", "(2,1,0)"]
    # ,["(2,2,0)", "(3,3,0)"], ["(2,2,0)", "(4,4,0)"], ["(2,2,0)", "(2,1,0)"]]
    modes_models = [["(2,2,0)", "(2,2,1) II"], [
        "(2,2,0)", "(2,2,1) II", '(3,3,0)']]
    detector = "LIGO"
    qs = [1.5, 10]
    cores = 1
    horizons_coeffs = {
        1.5: {
            2: [-0.42087011, 2.27776219, - 2.6453478, - 1.91044062],
            3: [-0.29792073, 1.37664448, - 0.54110655, - 3.89201678],
        },
        # TODO: NEED TO UPDATE q = 10 values!
        10: {
            2: [-0.16837121, 0.2268148, 2.54345812, -6.3327004],
            3: [-0.75199848, 4.87464029, -9.19930238, 2.57682028],
        }
    }

    masses_range = {
        1.5: {
            2: [3e1, 5e3, 50],
            3: [8e1, 4e3, 45],
        },
        10: {
            2: [6e1, 3.5e3, 45],
            3: [2e2, 3e3, 30],

        }
    }
    N_permass = 100

    for q in qs:
        for modes_model in modes_models:
            N_modes = len(modes_model)
            horizon = np.poly1d(horizons_coeffs[q][N_modes])

            label_data = 'logZ: ' + modes_data[0]
            label_model = 'logZ:'

            for mode in modes_model:
                label_model += ' ' + mode

            masses = np.logspace(np.log10(masses_range[q][N_modes][0]), np.log10(
                masses_range[q][N_modes][1]), masses_range[q][N_modes][2], endpoint=True)
            values = [(
                modes_data,
                modes_model,
                detector,
                mass,
                10**horizon(np.log10(mass)),
                q,
                np.random.randint(x, 1e4),
                f'{N_modes}_modes',
            )
                for mass in masses for x in [1] * N_permass
            ]
            # values = [(modes_data, modes_model, detector, masses[i],
            #            redshifts[i], q, num, seeds[i], f'{N_modes}_modes') for i in range(len(redshifts))]

            with Pool(processes=cores) as pool:
                res = pool.starmap(compute_log_B, values)
