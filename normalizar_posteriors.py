import glob
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
from scipy import integrate


def compute_normalized_posterior():
    columns_inj = ('mass',
                   'redshift',
                   '210',
                   '220',
                   '(2,2,1) I',
                   '221',
                   '330',
                   '440',
                   'estimated',
                   'upper-1sigma',
                   'lower-1sigma',
                   'upper-2sigma',
                   'lower-2sigma',
                   'upper-3sigma',
                   'lower-3sigma',
                   )
    columns_inj_220 = ('mass',
                       'redshift',
                       '220',
                       'estimated',
                       'upper-1sigma',
                       'lower-1sigma',
                       'upper-2sigma',
                       'lower-2sigma',
                       'upper-3sigma',
                       'lower-3sigma',
                       )
    qs = [1.5]
    N_modes = [3]
    pars = {
        2: [
            'A_220',
            'phi_220',
            'freq_220',
            'tau_220',
            'R_2modes',
            'phi_2modes',
            'freq_2modes',
            'tau_2modes', ],
    }
    pars[3] = pars[2] + [
        'R_3modes',
        'phi_3modes',
        'freq_3modes',
        'tau_3modes', ]

    # Golm path
    save_path = '/home/iaraota/Julia/new_q_nospin/mcmc_qnm/data/samples_pars/norm_psd/'

    for N_mode in N_modes:
        for q in qs:
            file_path = f'pars_{q}_data_220_221_330_440_210_model_{N_mode}_mass_'
            folders = glob.glob(file_path + '*')
            masses_redshifts = set()
            for folder in folders:
                mass = re.search('mass_(.*)_redshift', folder)
                mass = float(mass.group(1))
                redshift = re.search('redshift_(.*)_seed', folder)
                redshift = float(redshift.group(1))
                masses_redshifts.add((mass, redshift))
            save_file = {}
            df_inj = {}
            for par in pars[N_mode]:
                file_inj = f'/home/iaraota/Julia/new_q_nospin/mcmc_qnm/data/samples_pars/injected/{N_mode}_modes_{q}_data_220_221_330_440_210_model_220_221_par_{par}.dat'
                if par in ['A_220', 'phi_220', 'freq_220', 'tau_220']:
                    df_inj[par] = pd.DataFrame(np.genfromtxt(
                        file_inj), columns=columns_inj_220)
                else:
                    df_inj[par] = pd.DataFrame(
                        np.genfromtxt(file_inj), columns=columns_inj)

                save_file[par] = f'{q}_data_220_221_330_440_210_model_{N_mode}_par_{par}.dat'
                with open(save_path + save_file[par], 'w') as file:
                    file.write(
                        '#(0)mass(1)redshift(2)error-221(3)error-330(4)error-440(5)error-210(6)error-220\n')

            for mass, redshift in tqdm(masses_redshifts):
                sampler_path = file_path + f'{mass}_redshift_{redshift}_seed_'
                sampler_folders = glob.glob(sampler_path + '*')
                for sampler_folder in sampler_folders:
                    posteriors_file = sampler_folder + '/multi-post_equal_weights.dat'
                    posteriors = np.genfromtxt(posteriors_file)
                    try:
                        df_samples = pd.DataFrame(
                            posteriors, columns=pars[N_mode] + ['logZ'])
                        for par in pars[N_mode]:
                            df_inj_new = df_inj[par][round(
                                df_inj[par]['mass'], 1) == mass]
                            pos = sorted(df_samples[par].values)
                            kde_pos = stats.gaussian_kde(pos)

                            xs = np.linspace(min(pos), max(pos))
                            errors = {}
                            if par in ['A_220', 'phi_220', 'freq_220', 'tau_220']:
                                inj_x = df_inj_new['220'].values[0]
                                int_inj = integrate.quad(
                                    kde_pos, -np.inf, inj_x)[0]
                                errors['220'] = stats.norm.ppf(int_inj)

                                with open(save_path + save_file[par], 'a') as file:
                                    file.write(f'{mass}\t')
                                    file.write(f'{redshift}\t')
                                    file.write(f'{errors["220"]}\n')

                            else:
                                for mode in ['221', '330', '440', '210', '220']:
                                    inj_x = df_inj_new[mode].values[0]
                                    int_inj = integrate.quad(
                                        kde_pos, -np.inf, inj_x)[0]
                                    errors[mode] = stats.norm.ppf(int_inj)

                                with open(save_path + save_file[par], 'a') as file:
                                    file.write(f'{mass}\t')
                                    file.write(f'{redshift}\t')
                                    file.write(f'{errors["221"]}\t')
                                    file.write(f'{errors["330"]}\t')
                                    file.write(f'{errors["440"]}\t')
                                    file.write(f'{errors["210"]}\t')
                                    file.write(f'{errors["220"]}\n')

                    except:
                        pass


compute_normalized_posterior()
