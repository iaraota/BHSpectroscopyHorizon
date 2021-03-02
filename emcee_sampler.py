import os
from datetime import datetime

# pair_plot method libs:
from scipy.stats import gaussian_kde
from matplotlib import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import emcee

import Plots
from SourceData import SourceData
from Models import Models, TrueParameters, Priors
from modules import MCMCFunctions, GWFunctions

import corner

class EmceeSampler(SourceData):
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


    def joint_plot_mass_spin(self,
        model,
        ratio,
        ):
        self.run_sampler(model, ratio)

        df = pd.DataFrame(self.flat_samples, columns=self.true_pars.theta_labels)
        m0, m1 = self.modes_model
        l0 = '('+m0[1]+','+m0[3]+','+m0[5]+')'
        l1 = '('+m1[1]+','+m1[3]+','+m1[5]+')'
        df_colors = pd.DataFrame({
            l0: ['tab:blue'],
            l1: ['tab:orange'],
            },
            index=['color']
            )
        
        df_masses = pd.DataFrame({
            l0: [df[r"$M_{{{0}}}$".format(m0)].values],
            l1: [df[r"$M_{{{0}}}$".format(m1)].values],
            'true': [self.final_mass],
            'label': ['final mass'],
            },
            index=['y']
            )
        df_spins = pd.DataFrame({
            l0: [df[r"$a_{{{0}}}$".format(m0)].values],
            l1: [df[r"$a_{{{0}}}$".format(m1)].values],
            'teste': [df[r"$a_{{{0}}}$".format(m1)].values],
            'true': [self.final_spin],
            'label': ['final spin'],
            },
            index=['x']
            )
        
        data = df_colors.append([df_masses, df_spins])

        plots = Plots.Plots()
        plots.pair_plot(data,
            )

        plt.savefig("figs/mass_spin"+str(len(self.modes_data))+"data"+str(len(self.modes_model))+"model"
                    + str(self.final_mass) + "_" + str(self.redshift) + "_"
                    + self.detector["label"] + ".pdf", dpi = 360)
        plt.show()

    def run_sampler(self,
        model:str,
        ratio:bool,
        ):
        """Run MCMC sampler.

        Parameters
        ----------
        model : str
            QNM model. Can be set to {"kerr", "mass_spin",
            "freq_tau", "omegas"}

        ratio : bool
            Choose true if model has amplitude ratios
            and False if model fits all amplitudes
        """
        self._compute_logpdf_function(model, ratio)

        self.true_pars.choose_theta_true(model, ratio)
        print(self.true_pars.theta_true)
        ndim = len(self.true_pars.theta_true)
        self.nwalkers = 50
        self.nsteps = 1500
        self.thin = 30
        # self.nwalkers = 20
        # self.nsteps = 500
        # self.thin = 5
        # self.nwalkers = 20
        # self.nsteps = 100
        # self.thin = 15


        # pos = self.prior_transform(np.random.rand(self.nwalkers, ndim))
        pos = (self.true_pars.theta_true + 1e-4 * np.random.randn(self.nwalkers, ndim))

        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.log_pdf)
        sampler.run_mcmc(pos, self.nsteps, progress=True)
        self.samples = sampler.get_chain()
        self.flat_samples = sampler.get_chain(discard=int(self.nsteps/2), thin=self.thin, flat=True)

        corner.corner(self.flat_samples, truths=self.true_pars.theta_true, labels = self.true_pars.theta_labels)

        plt.savefig("figs/corner/"+str(model)+str(len(self.modes_data))+"data"+str(len(self.modes_model))+"model"
                    + str(self.final_mass) + "_" + str(self.redshift) + "_"
                    + self.detector["label"] + ".pdf", dpi = 360)
        plt.show()

    def _compute_logpdf_function(
        self,
        model:str,
        ratio:bool,
        ):
        """Generate log of probability density function.

        Parameters
        ----------
        model : str
            QNM model. Can be set to {"kerr", "mass_spin",
            "freq_tau", "omegas"}

        ratio : bool
            Choose true if model has amplitude ratios
            and False if model fits all amplitudes
        """
        #TODO: choose prior other than uniform
        self.models.choose_model(model, ratio)
        self.priors.uniform_prior(model, ratio)

        self.log_pdf = lambda theta: MCMCFunctions.log_probability_qnm(
            theta,
            self.priors.prior_function,
            self.models.model,
            self.data,
            self.detector["freq"],
            self.detector["psd"]
            )



if __name__ == '__main__':
    np.random.seed(123)
    """GW190521
    final mass = 150.3
    redshift = 0.72
    spectrocopy horizon = 0.148689
    """
    m_f = 500
    z = 0.1
    q = 1.5

    m_f = 150.3
    # z = 0.5
    z = 0.72
    z = 0.15
    z = 0.05
    z = 0.1
    m_f = 1e4
    detector = "LIGO"
    detector = "CE"
    modes = ["(2,2,0)"]
    modes = ["(2,2,0)", "(2,2,1) II"]
    # modes = ["(2,2,0)", "(2,2,1) I", "(3,3,0)", "(4,4,0)", "(2,1,0)"]
    modes = ["(2,2,0)", "(4,4,0)"]
    # modes = ["(2,2,0)", "(3,3,0)"]
    modes_model = ["(2,2,0)"]
    modes_model = ["(2,2,0)", "(2,2,1) II"]
    # modes_model = ["(2,2,0)", "(2,2,1) I", "(3,3,0)", "(4,4,0)", "(2,1,0)"]
    modes_model = ["(2,2,0)", "(4,4,0)"]
    # modes_model = ["(2,2,0)", "(3,3,0)"]
    teste = EmceeSampler(modes, modes_model, detector, m_f, z, q, "FH")
    teste.run_sampler('kerr', False)
    df = pd.DataFrame(teste.flat_samples, columns=teste.true_pars.theta_labels)




def plot_allmasses_spin():

    modeskerr = EmceeSampler(modes, modes_model, detector, m_f, z, q, "FH")
    modeskerr.run_sampler('kerr', False)
    modes2 = EmceeSampler(modes, modes_model, detector, m_f, z, q, "FH")
    modes2.run_sampler('mass_spin', False)
    modes1 = EmceeSampler(modes, [modes_model[0]], detector, m_f, z, q, "FH")
    modes1.run_sampler('mass_spin', False)


    dfkerr = pd.DataFrame(modeskerr.flat_samples, columns=modeskerr.true_pars.theta_labels)
    df2 = pd.DataFrame(modes2.flat_samples, columns=modes2.true_pars.theta_labels)
    df1 = pd.DataFrame(modes1.flat_samples, columns=modes1.true_pars.theta_labels)
    m0, m1 = modes_model
    l0 = '2 modes, fundamental'
    l1 = '2 modes, overtone'
    l2 = '1 mode'
    l3 = '2 modes, kerr'
    df_colors = pd.DataFrame({
        l0: ['tab:blue'],
        l1: ['tab:red'],
        l2: ['tab:green'],
        l3: ['tab:purple']
        },
        index=['color']
        )
    
    df_masses = pd.DataFrame({
        l0: [df2[r"$M_{{{0}}}$".format(m0)].values],
        l1: [df2[r"$M_{{{0}}}$".format(m1)].values],
        l2: [df1[r"$M_{{{0}}}$".format(m0)].values],
        l3: [dfkerr[r"$M_f$"].values],
        'true': [modes2.final_mass],
        'label': [r'final mass [$M_\odot$]'],
        },
        index=['y']
        )
    df_spins = pd.DataFrame({
        l0: [df2[r"$a_{{{0}}}$".format(m0)].values],
        l1: [df2[r"$a_{{{0}}}$".format(m1)].values],
        l2: [df1[r"$a_{{{0}}}$".format(m0)].values],
        l3: [dfkerr[r"$a_f$"].values],
        'true': [modes2.final_spin],
        'label': [r'final spin [$M_f$]'],
        },
        index=['x']
        )
    
    data = df_colors.append([df_masses, df_spins])

    plots = Plots.Plots()
    plots.pair_plot(data,
        )

    plt.savefig("figs/kerr_mass_spin"+str(len(modes2.modes_data))+"data"+str(len(modes2.modes_model))+"model"
                + str(modes2.final_mass) + "_" + str(modes2.redshift) + "_"
                + modes2.detector["label"] + ".pdf", dpi = 360)
    plt.show()