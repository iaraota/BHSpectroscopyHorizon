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


class EmceeSamplerMassSpin(SourceData):
    """Compute posterior probability densities for
    quasinormal modes in the frequency domain."""

    def __init__(self, modes_data:list, modes_model:list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modes_data = modes_data
        self.modes_model = modes_model
        self.inject_data(self.modes_data) # construct self.data

        self.fit_coeff = {}
        for mode in self.modes_model:
            self.fit_coeff[mode] = self.transf_fit_coeff(mode)

        self.df_a_omegas = {}
        for mode in self.modes_model:
            self.df_a_omegas[mode] = self.create_a_over_M_omegas_dataframe(mode)

    def corner_omegas(self):
        self.run_sampler()
        
        df = pd.DataFrame(self.flat_samples, columns=self.theta_labels)
        
        x_true, y_true = self.final_spin, self.final_mass

        x, y = [], []
        labels = []
        for mode in reversed(self.modes_model):
            # import data
            x.append(df[r"$M_{{{0}}}$".format(mode)].values)
            y.append(df[r"$a_{{{0}}}$".format(mode)].values)
            labels.append('('+mode[1]+','+mode[3]+','+mode[5]+')')
        print(type(x[0]))
        omega_r0, omega_i0, omega_r1, omega_i1 = np.zeros(len(x[0])), np.zeros(len(x[0])), np.zeros(len(x[0])), np.zeros(len(x[0])), 
        for i in range(len(x[0])):
            x[0][i] /=self.mass_initial
            x[1][i] /=self.mass_initial
            omega_r0[i], omega_i0[i] = self.transform_mass_spin_to_omegas(
                x[0][i],
                y[0][i],
                self.df_a_omegas[self.modes_model[0]],
            )
            omega_r1[i], omega_i1[i] = self.transform_mass_spin_to_omegas(
                x[0][i],
                y[1][i],
                self.df_a_omegas[self.modes_model[1]],
            )
        plt.scatter(self.qnm_modes[self.modes_model[0]].omega_r, self.qnm_modes[self.modes_model[0]].omega_i, marker = "+")
        plt.scatter(omega_r0, omega_i0, alpha = .5)
        # corner.corner(np.vstack([omega_r0, omega_i0, omega_r1, omega_i1]))
        plt.show()

    def joint_plot_mass_spin(self):
        self.run_sampler()
        self._plot_parameters()

        df = pd.DataFrame(self.flat_samples, columns=self.theta_labels)
        colors = ['tab:blue', 'tab:orange']

        x_true, y_true = self.final_spin, self.final_mass

        x, y = [], []
        labels = []
        for mode in reversed(self.modes_model):
            # import data
            x.append(df[r"$a_{{{0}}}$".format(mode)].values)
            y.append(df[r"$M_{{{0}}}$".format(mode)].values)
            labels.append('('+mode[1]+','+mode[3]+','+mode[5]+')')
        self.pair_plot(x[0], y[0], x[1], y[1],
            x_true, y_true,
            xlabel = 'final spin',
            ylabel = 'final mass',
            plot_color= colors,
            label=labels,
            )

        plt.savefig("figs/mass_spin"+str(len(self.modes_data))+"data"+str(len(self.modes_model))+"model"
                    + str(self.final_mass) + "_" + str(self.redshift) + "_"
                    + self.detector["label"] + ".pdf", dpi = 360)
        plt.show()

    def pair_plot(self,
        x1:list,
        y1:list,
        x2:list,
        y2:list,
        x_true:float,
        y_true:float,
        plot_color:str,
        xlabel:str,
        ylabel:str,
        label:dict,
        levels = [90],
        style="both",
        clabel=False,
        ):
        """Create a posterior distribuition plot for two parameters.

        Parameters
        ----------
        x : list
            first parameter sample.
        y : list
            secondi parameter sample.
        x_true : float
            first parameter true value.
        y_true : float
            second parameter true value.
        plot_color : str
            plot color.
        xlabel : str
            x axis label.
        ylabel : str
            y axis label.
        labels : list
            values labels.
        levels : list, optional
            Confidential intervals, by default [90]
        style : str, optional
            Choose to plot true values style {'both', 'point', 'lines'}, by default "both"
        clabel : bool, optional
            Show interval labels, by default False
        """
    
        # start plot figure
        fig, axScatter = plt.subplots(figsize=(10, 10))
        
        # create new axes on the right and on the top of the current axes
        # The first argument of the new_vertical(new_horizontal) method is
        # the height (width) of the axes to be created in inches.
        divider = make_axes_locatable(axScatter)
        axHistx = divider.append_axes("top", 1.5, pad=0., sharex=axScatter)
        axHisty = divider.append_axes("right", 1.5, pad=0., sharey=axScatter)

        # make some labels invisible
        # axHistx.axis('off')
        axHistx.xaxis.set_tick_params(labelbottom=False)
        axHistx.set_yticks([])

        axHisty.set_xticks([])
        axHisty.yaxis.set_tick_params(labelleft=False)
        
        # plot true values
        if style=='both':
            axScatter.scatter(x_true, y_true, marker = '+', color = 'k')
            axScatter.axvline(x = x_true, lw=2, ls=':', color = 'k', alpha=.5)
            axScatter.axhline(y = y_true, lw=2, ls=':', color = 'k', alpha=.5)
        elif style=='point':
            axScatter.scatter(teste.final_spin, teste.final_mass, marker = '+', color = 'k')
        elif style=='lines':
            axScatter.axvline(x = x_true, lw=2, ls=':', color = 'k', alpha=.5)
            axScatter.axhline(y = y_true, lw=2, ls=':', color = 'k', alpha=.5)
        i = 0
        for (x,y) in [(x1,y1), (x2,y2)]:

            # compute density
            k = gaussian_kde(np.vstack([x, y]))
            
            # create grid
            xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            
            #set zi to 0-1 scale
            zi = (zi-zi.min())/(zi.max() - zi.min())
            zi =zi.reshape(xi.shape)
            

            # the scatter plot:
            # axScatter.scatter(x, y, alpha = 0.05, color = plot_color[i], marker='p')
                
            #set up plot
            origin = 'lower'
            lvls = []
            for level in levels:
                lvls.append(1 - level/100)

            CS = axScatter.contour(xi, yi, zi,levels = lvls,
                    colors=(plot_color[i],),
                    linewidths=(2,),
                    origin=origin)

            # fill inside
            lvls.append(1)

            axScatter.contourf(xi, yi, zi, levels = lvls,
                colors=(plot_color[i],),
                alpha = 0.3,
                origin=origin,
                )
    
            if clabel == True:
                axScatter.clabel(CS, fmt=levels, colors = plot_color[i])
                axScatter.ticklabel_format(axis='y', style='sci', scilimits=(0,0))



            # axHistx.hist(x, bins = 100, density = True, alpha = 0.3)
            xx = np.linspace(min(x), max(x),100)
            x_max = max(np.concatenate((gaussian_kde(x1)(xx),gaussian_kde(x2)(xx))))

            axHistx.fill_between(xx,0,gaussian_kde(x)(xx)/x_max, color = plot_color[i], alpha = 0.3)
            axHistx.plot(xx,gaussian_kde(x)(xx)/x_max, color = plot_color[i], linewidth = 2)
            axHistx.axvline(x = np.percentile(x, 5), lw=2, ls='--', color = plot_color[i])
            axHistx.axvline(x = np.percentile(x, 95), lw=2, ls='--', color = plot_color[i])


            # first of all, the base transformation of the data points is needed
            base = plt.gca().transData
            rot = transforms.Affine2D().rotate_deg(-90)

            # axHisty.hist(y, bins=100, orientation='horizontal', alpha = 0.2)
            yy = np.linspace(min(y), max(y),100)
            y_max = max(np.concatenate((gaussian_kde(y1)(yy),gaussian_kde(y2)(yy))))
            axHisty.fill_between(-yy,gaussian_kde(y)(yy)/y_max, 0, color = plot_color[i], alpha = 0.3, transform= rot + base)
            axHisty.plot(-yy, gaussian_kde(y)(yy)/y_max, color = plot_color[i], linewidth = 2, transform= rot + base, label=label[i])
            axHisty.axhline(y = np.percentile(y, 5), lw=2, ls='--', color = plot_color[i])
            axHisty.axhline(y = np.percentile(y, 95), lw=2, ls='--', color = plot_color[i])

            i+=1
            
        # add this after calling the pair_plot function to remove gap between plots
        axHistx.set_ylim(0, 1.05)    
        axHisty.set_xlim(0, 1.05)
        # axScatter.set_xlim(min(np.concatenate((x1,x2))), max(np.concatenate((x1,x2))))
        # axScatter.set_ylim(min(np.concatenate((y1,y2))), max(np.concatenate((y1,y2))))
        
        axScatter.set_xlim(0.3,0.8)
        axScatter.set_ylim(120, 170)
        
        axScatter.set_xlabel(xlabel)
        axScatter.set_ylabel(ylabel)
        plt.legend(bbox_to_anchor=(0,1))

        fig.tight_layout()

    def run_sampler(self):
        """Compute posterior density function of the parameters.
        """
        self._prior_and_logpdf()
        self._theta_true()
        ndim = len(self.theta_true)
        self.nwalkers = 100
        self.nsteps = 1000
        self.thin = 30
        # self.nwalkers = 30
        # self.nsteps = 100
        # self.thin = 15
        self.nwalkers = 50
        self.nsteps = 800
        self.thin = 15


        # pos = self.prior_transform(np.random.rand(self.nwalkers, ndim))
        pos = (self.theta_true + 1e-4 * np.random.randn(self.nwalkers, ndim))

        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.log_pdf)
        sampler.run_mcmc(pos, self.nsteps, progress=True)
        self.samples = sampler.get_chain()
        self.flat_samples = sampler.get_chain(discard=int(self.nsteps/2), thin=self.thin, flat=True)

        # corner.corner(self.flat_samples, truths=self.theta_true, labels = self.theta_labels)
        # plt.show()

    def prior_transform(self, hypercube):
        """Transforms the uniform random variable 'hypercube ~ Unif[0., 1.)'
        to the parameter of interest 'theta ~ Unif[true/100,true*100]'."""
        transform = lambda a, b, x: a + (b - a) * x
        cube = np.array(hypercube)
        for i in range(len(self.modes_model)):
            cube[0+4*i] = transform(0.0, 10,cube[0 + 4*i])
            cube[1+4*i] = transform(0.0, 2*np.pi,cube[1 + 4*i])
            cube[2+4*i] = transform(0.9, 1,cube[2 + 4*i])
            cube[3+4*i] = transform(0.0, 0.9999,cube[3 + 4*i])
        return cube

    def _prior_and_logpdf(self):
        """Define priors for the model
        """
        
        limit_min = []
        limit_max = []
        for i in range(len(self.modes_model)):
            limit_min.extend([0.,
                0.,
                # 0.1,
                self.final_mass/10,
                0,
                ])
            limit_max.extend([
                10.,
                2*np.pi,
                # 10,
                self.final_mass*10,
                0.9999,
                ])

        self.prior_function = lambda theta: MCMCFunctions.noninfor_log_prior(theta, limit_min, limit_max)
        self.log_pdf = lambda theta: MCMCFunctions.log_probability_qnm(
            theta,
            self.prior_function,
            self.model_function,
            self.data,
            self.detector["freq"],
            self.detector["psd"]
            )

    def _theta_true(self):
        """Generate a list of the true injected parameters.
        """
        self.theta_true = []
        self.theta_labels = []

        for mode in self.modes_model:
            self.theta_true.extend(
                [self.qnm_modes[mode].amplitude,
                float(self.qnm_modes[mode].phase),
                # self.mass_final,
                self.final_mass,
                self.final_spin])
            self.theta_labels.extend([
                r"$A_{{{0}}}$".format(mode),
                r"$\phi_{{{0}}}$".format(mode),
                r"$M_{{{0}}}$".format(mode),
                r"$a_{{{0}}}$".format(mode),
            ])
        self.theta_true = tuple(self.theta_true)

    def model_function(self, theta:list):
        """Generate waveform model function of QNMs.

        Parameters
        ----------
        theta : array_like
            Model parameters.

        Returns
        -------
        function
            Waveform model as a function of parameters theta.
        """
        h_model = 0
        for i in range(len(self.modes_model)):
            A, phi, M, a = theta[0 + 4*i: 4 + 4*i]
            M = M/self.mass_initial
            omega_r, omega_i = self.transform_mass_spin_to_omegas(
                M,
                a,
                self.df_a_omegas[self.modes_model[i]],
                # self.modes_model[i],
                # self.fit_coeff[self.modes_model[i]]
            )
            h_model += self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
                self.detector["freq"]*self.time_convert, A, phi, omega_r, omega_i, 
                part = "real", convention = self.ft_convention)
        return h_model

    def _plot_parameters(self):
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.family"] = "STIXGeneral"
        # plt.rcParams["figure.figsize"] = [20, 8]  # plot image size

        SMALL_SIZE = 20
        MEDIUM_SIZE = 25
        BIGGER_SIZE = 35

        plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
        plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)

class EmceeSamplerOmegas(SourceData):
    """Compute posterior probability densities for
    quasinormal modes in the frequency domain."""

    def __init__(self, modes_data:list, modes_model:list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modes_data = modes_data
        self.modes_model = modes_model
        self.inject_data(self.modes_data) # construct self.data

        self.fit_coeff = {}
        for mode in self.modes_model:
            self.fit_coeff[mode] = self.transf_fit_coeff(mode)

        self.df_a_omegas = {}
        for mode in self.modes_model:
            self.df_a_omegas[mode] = self.create_a_over_M_omegas_dataframe(mode)

    def corner_omegas(self):
        self.run_sampler()
        
        df = pd.DataFrame(self.flat_samples, columns=self.theta_labels)
        
        x_true, y_true = self.final_spin, self.final_mass

        x, y = [], []
        labels = []
        for mode in reversed(self.modes_model):
            # import data
            x.append(df[r"$M_{{{0}}}$".format(mode)].values)
            y.append(df[r"$a_{{{0}}}$".format(mode)].values)
            labels.append('('+mode[1]+','+mode[3]+','+mode[5]+')')
        print(type(x[0]))
        omega_r0, omega_i0, omega_r1, omega_i1 = np.zeros(len(x[0])), np.zeros(len(x[0])), np.zeros(len(x[0])), np.zeros(len(x[0])), 
        for i in range(len(x[0])):
            x[0][i] /=self.mass_initial
            x[1][i] /=self.mass_initial
            omega_r0[i], omega_i0[i] = self.transform_mass_spin_to_omegas(
                x[0][i],
                y[0][i],
                self.df_a_omegas[self.modes_model[0]],
            )
            omega_r1[i], omega_i1[i] = self.transform_mass_spin_to_omegas(
                x[0][i],
                y[1][i],
                self.df_a_omegas[self.modes_model[1]],
            )
        plt.scatter(self.qnm_modes[self.modes_model[0]].omega_r, self.qnm_modes[self.modes_model[0]].omega_i, marker = "+")
        plt.scatter(omega_r0, omega_i0, alpha = .5)
        # corner.corner(np.vstack([omega_r0, omega_i0, omega_r1, omega_i1]))
        plt.show()

    def joint_plot_mass_spin(self):
        self.run_sampler()
        self._plot_parameters()

        df = pd.DataFrame(self.flat_samples, columns=self.theta_labels)
        colors = ['tab:orange', 'tab:blue']

        wr, wi = [], []
        labels = []
        for mode in self.modes_model:
            # import data
            wr.append(df[r"$\omega^r_{{{0}}}$".format(mode)].values)
            wi.append(df[r"$\omega^i_{{{0}}}$".format(mode)].values)
            labels.append('('+mode[1]+','+mode[3]+','+mode[5]+')')
        M, a = [np.zeros(len(wr[0])), np.zeros(len(wr[0]))], [np.zeros(len(wr[0])), np.zeros(len(wr[0]))], 

        for i in range(2):
            for j in range(len(wr[0])):
                M[i][j], a[i][j] = self.transform_omegas_to_mass_spin(
                    wr[i][j],
                    wi[i][j],
                    self.df_a_omegas[self.modes_model[i]],
                    self.fit_coeff[self.modes_model[i]],
                    )
            M[i] *= self.initial_mass
        
        x_true, y_true = self.final_spin, self.final_mass
    
        self.pair_plot(a[0], M[0], a[1], M[1],
            x_true, y_true,
            xlabel = 'final spin',
            ylabel = 'final mass',
            plot_color= colors,
            label=labels,
            )

        plt.savefig("figs/mass_spin_omegasfit_"+str(len(self.modes_data))+"data"+str(len(self.modes_model))+"model"
                    + str(self.final_mass) + "_" + str(self.redshift) + "_"
                    + self.detector["label"] + ".pdf", dpi = 360)
        plt.show()

    def pair_plot(self,
        x1:list,
        y1:list,
        x2:list,
        y2:list,
        x_true:float,
        y_true:float,
        plot_color:str,
        xlabel:str,
        ylabel:str,
        label:dict,
        levels = [90],
        style="both",
        clabel=False,
        ):
        """Create a posterior distribuition plot for two parameters.

        Parameters
        ----------
        x : list
            first parameter sample.
        y : list
            secondi parameter sample.
        x_true : float
            first parameter true value.
        y_true : float
            second parameter true value.
        plot_color : str
            plot color.
        xlabel : str
            x axis label.
        ylabel : str
            y axis label.
        labels : list
            values labels.
        levels : list, optional
            Confidential intervals, by default [90]
        style : str, optional
            Choose to plot true values style {'both', 'point', 'lines'}, by default "both"
        clabel : bool, optional
            Show interval labels, by default False
        """
    
        # start plot figure
        fig, axScatter = plt.subplots(figsize=(10, 10))
        
        # create new axes on the right and on the top of the current axes
        # The first argument of the new_vertical(new_horizontal) method is
        # the height (width) of the axes to be created in inches.
        divider = make_axes_locatable(axScatter)
        axHistx = divider.append_axes("top", 1.5, pad=0., sharex=axScatter)
        axHisty = divider.append_axes("right", 1.5, pad=0., sharey=axScatter)

        # make some labels invisible
        # axHistx.axis('off')
        axHistx.xaxis.set_tick_params(labelbottom=False)
        axHistx.set_yticks([])

        axHisty.set_xticks([])
        axHisty.yaxis.set_tick_params(labelleft=False)
        
        # plot true values
        if style=='both':
            axScatter.scatter(x_true, y_true, marker = '+', color = 'k')
            axScatter.axvline(x = x_true, lw=2, ls=':', color = 'k', alpha=.5)
            axScatter.axhline(y = y_true, lw=2, ls=':', color = 'k', alpha=.5)
        elif style=='point':
            axScatter.scatter(teste.final_spin, teste.final_mass, marker = '+', color = 'k')
        elif style=='lines':
            axScatter.axvline(x = x_true, lw=2, ls=':', color = 'k', alpha=.5)
            axScatter.axhline(y = y_true, lw=2, ls=':', color = 'k', alpha=.5)
        i = 0
        for (x,y) in [(x1,y1), (x2,y2)]:

            # compute density
            k = gaussian_kde(np.vstack([x, y]))
            
            # create grid
            xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            
            #set zi to 0-1 scale
            zi = (zi-zi.min())/(zi.max() - zi.min())
            zi =zi.reshape(xi.shape)
            

            # the scatter plot:
            # axScatter.scatter(x, y, alpha = 0.05, color = plot_color[i], marker='p')
                
            #set up plot
            origin = 'lower'
            lvls = []
            for level in levels:
                lvls.append(1 - level/100)

            CS = axScatter.contour(xi, yi, zi,levels = lvls,
                    colors=(plot_color[i],),
                    linewidths=(2,),
                    origin=origin)

            # fill inside
            lvls.append(1)

            axScatter.contourf(xi, yi, zi, levels = lvls,
                colors=(plot_color[i],),
                alpha = 0.3,
                origin=origin,
                )
    
            if clabel == True:
                axScatter.clabel(CS, fmt=levels, colors = plot_color[i])
                axScatter.ticklabel_format(axis='y', style='sci', scilimits=(0,0))



            # axHistx.hist(x, bins = 100, density = True, alpha = 0.3)
            xx = np.linspace(min(x), max(x),100)
            x_max = max(np.concatenate((gaussian_kde(x1)(xx),gaussian_kde(x2)(xx))))

            axHistx.fill_between(xx,0,gaussian_kde(x)(xx)/x_max, color = plot_color[i], alpha = 0.3)
            axHistx.plot(xx,gaussian_kde(x)(xx)/x_max, color = plot_color[i], linewidth = 2)
            axHistx.axvline(x = np.percentile(x, 5), lw=2, ls='--', color = plot_color[i])
            axHistx.axvline(x = np.percentile(x, 95), lw=2, ls='--', color = plot_color[i])


            # first of all, the base transformation of the data points is needed
            base = plt.gca().transData
            rot = transforms.Affine2D().rotate_deg(-90)

            # axHisty.hist(y, bins=100, orientation='horizontal', alpha = 0.2)
            yy = np.linspace(min(y), max(y),100)
            y_max = max(np.concatenate((gaussian_kde(y1)(yy),gaussian_kde(y2)(yy))))
            axHisty.fill_between(-yy,gaussian_kde(y)(yy)/y_max, 0, color = plot_color[i], alpha = 0.3, transform= rot + base)
            axHisty.plot(-yy, gaussian_kde(y)(yy)/y_max, color = plot_color[i], linewidth = 2, transform= rot + base, label=label[i])
            axHisty.axhline(y = np.percentile(y, 5), lw=2, ls='--', color = plot_color[i])
            axHisty.axhline(y = np.percentile(y, 95), lw=2, ls='--', color = plot_color[i])

            i+=1
            
        # add this after calling the pair_plot function to remove gap between plots
        axHistx.set_ylim(0, 1.05)    
        axHisty.set_xlim(0, 1.05)
        # axScatter.set_xlim(min(np.concatenate((x1,x2))), max(np.concatenate((x1,x2))))
        # axScatter.set_ylim(min(np.concatenate((y1,y2))), max(np.concatenate((y1,y2))))
        
        axScatter.set_xlim(0.3,0.8)
        axScatter.set_ylim(120, 170)
        
        axScatter.set_xlabel(xlabel)
        axScatter.set_ylabel(ylabel)
        plt.legend(bbox_to_anchor=(0,1))

        fig.tight_layout()

    def run_sampler(self):
        """Compute posterior density function of the parameters.
        """
        self._theta_true()
        self._prior_and_logpdf()
        ndim = len(self.theta_true)
        self.nwalkers = 100
        self.nsteps = 1000
        self.thin = 25
        self.nwalkers = 50
        self.nsteps = 800
        self.thin = 15

        # pos = self.prior_transform(np.random.rand(self.nwalkers, ndim))
        pos = (self.theta_true + 1e-4 * np.random.randn(self.nwalkers, ndim))

        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.log_pdf)
        sampler.run_mcmc(pos, self.nsteps, progress=True)
        self.samples = sampler.get_chain()
        self.flat_samples = sampler.get_chain(discard=int(self.nsteps/2), thin=self.thin, flat=True)

        # corner.corner(self.flat_samples, truths=self.theta_true, labels = self.theta_labels)
        # plt.show()

    def _prior_and_logpdf(self):
        """Define priors for the model
        """
        
        limit_min = []
        limit_max = []
        for i in range(len(self.modes_model)):
            limit_min.extend([0., 0., self.theta_true[2 + 4*i]/10, self.theta_true[3 + 4*i]/10])
            limit_max.extend([100., 2*np.pi, self.theta_true[2 + 4*i]*10, self.theta_true[3 + 4*i]*10])

        self.prior_function = lambda theta: MCMCFunctions.noninfor_log_prior(theta, limit_min, limit_max)
        self.log_pdf = lambda theta: MCMCFunctions.log_probability_qnm(
            theta, self.prior_function, self.model_function, self.data,
            self.detector["freq"], self.detector["psd"])

    def _theta_true(self):
        """Generate a list of the true injected parameters.
        """
        self.theta_true = []
        self.theta_labels = []

        for mode in self.modes_model:
            self.theta_true.extend(
                [
                self.qnm_modes[mode].amplitude,
                float(self.qnm_modes[mode].phase),
                self.qnm_modes[mode].omega_r,
                self.qnm_modes[mode].omega_i
                # self.qnm_modes[mode].frequency,
                # self.qnm_modes[mode].decay_time*1e3
                ]
                )
            self.theta_labels.extend([
                r"$A_{{{0}}}$".format(mode),
                r"$\phi_{{{0}}}$".format(mode),
                r"$\omega^r_{{{0}}}$".format(mode),
                r"$\omega^i_{{{0}}}$".format(mode),
            ])
        self.theta_true = tuple(self.theta_true)

    def model_function(self, theta:list):
        """Generate waveform model function of QNMs.

        Parameters
        ----------
        theta : array_like
            Model parameters.

        Returns
        -------
        function
            Waveform model as a function of parameters theta.
        """
        h_model = 0
        for i in range(len(self.modes_model)):
            A, phi, omega_r, omega_i = theta[0 + 4*i: 4 + 4*i]
            # A, phi, freq, tau = theta[0 + 4*i: 4 + 4*i]
            # tau *= 1e-3
            # omega_r = freq*2*np.pi*self.time_convert
            # omega_i = self.time_convert/tau
            h_model += self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
                self.detector["freq"]*self.time_convert, A, phi, omega_r, omega_i, 
                part = "real", convention = self.ft_convention)
        return h_model

    def _plot_parameters(self):
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.family"] = "STIXGeneral"
        # plt.rcParams["figure.figsize"] = [20, 8]  # plot image size

        SMALL_SIZE = 20
        MEDIUM_SIZE = 25
        BIGGER_SIZE = 35

        plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
        plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)



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
    # z = 0.72
    z = 0.15
    z = 0.05
    z = 0.01
    detector = "LIGO"
    modes = ["(2,2,0)"]
    # modes = ["(2,2,0)", "(2,2,1) I"]
    # modes = ["(2,2,0)", "(4,4,0)"]
    # modes = ["(2,2,0)", "(3,3,0)"]
    modes_model = ["(2,2,0)"]
    modes_model = ["(2,2,0)", "(2,2,1) I"]
    # modes_model = ["(2,2,0)", "(4,4,0)"]
    # modes_model = ["(2,2,0)", "(3,3,0)"]
    teste = EmceeSampler(modes, modes_model, detector, m_f, z, q, "FH")
    teste.run_sampler('df_dtau_sub', False)
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