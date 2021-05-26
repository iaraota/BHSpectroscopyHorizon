import glob
import numpy as np
import re
import pandas as pd


def hpd(trace, mass_frac=0.68):
    """
    Returns highest probability density region given by
    a set of samples.

    Parameters
    ----------
    trace : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For example, `massfrac` = 0.95 gives a
        95% HPD.

    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD
    """
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)

    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)

    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n - n_samples]

    # Pick out minimal interval
    min_int = np.argmin(int_width)

    # Return interval
    return np.array([d[min_int], d[min_int + n_samples]])


def compute_hpd_masses():
    qs = [1.5]
    N_modes = [2]
    pars = {
        2: ['A_220',
            'phi_220',
            'freq_220',
            'tau_220',
            'R_2',
            'phi_2',
            'freq_2',
            'tau_2', ],
    }
    pars[3] = pars[2] + [
        'R_3',
        'phi_3',
        'freq_3',
        'tau_3', ]

    # Golm path
    save_path = '/home/iaraota/Julia/new_q_nospin/mcmc_qnm/data/samples_pars/hdp/'

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
            for par in pars[N_mode]:
                save_file[par] = f'{q}_data_220_221_330_440_210_model_{N_mode}_par_{par}modes.dat'
                with open(save_path + save_file[par], 'w') as file:
                    file.write(
                        '#(0)mass(1)redshift(2)lower-1sigma(3)upper-1sigma(4)lower-2sigma(5)upper-2sigma(6)lower-3sigma(7)upper-3sigma\n')

            for mass, redshift in masses_redshifts:
                sampler_path = file_path + f'{mass}_redshift_{redshift}_seed_'
                sampler_folders = glob.glob(sampler_path + '*')
                for sampler_folder in sampler_folders:
                    posteriors_file = sampler_folder + '/multi-post_equal_weights.dat'
                    posteriors = np.genfromtxt(posteriors_file)
                    df_samples = pd.DataFrame(
                        posteriors, columns=pars[N_mode] + ['logZ'])

                    for par in pars[N_mode]:
                        with open(save_path + save_file[par], 'a') as file:
                            file.write(f'{mass}\t')
                            file.write(f'{redshift}\t')
                            file.write(f'{hpd(df_samples[par], 0.6827)[0]}\t')
                            file.write(f'{hpd(df_samples[par], 0.6827)[1]}\t')
                            file.write(f'{hpd(df_samples[par], 0.9545)[0]}\t')
                            file.write(f'{hpd(df_samples[par], 0.9545)[1]}\t')
                            file.write(f'{hpd(df_samples[par], 0.9973)[0]}\t')
                            file.write(f'{hpd(df_samples[par], 0.9973)[1]}\n')


compute_hpd_masses()
