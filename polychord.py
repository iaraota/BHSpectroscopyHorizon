import os

import numpy as np
import matplotlib.pyplot as plt

import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

from SourceData import SourceData
from modules import MCMCFunctions, GWFunctions
import getdist.plots

class Polychord(SourceData):
    """Compute posterior probability densities for
    quasinormal modes in the frequency domain."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modes_data = self.modes
        self.modes_model = self.modes
        self._inject_data(self.modes_data)
        self._theta_true()

    def run_all(self):
        # self.modes_data = self.modes
        # self._inject_data(self.modes_data)

        # self.modes_model = self.modes
        # self._theta_true()
        self.plot_posterior()
        
        # self.modes_model = [self.modes[0]]
        # self._theta_true()
        # self.plot_posterior()


        # self.modes_data = [self.modes[0]]
        # self._inject_data(self.modes_data)

        # self.modes_model = self.modes
        # self._theta_true()
        # self.plot_posterior()

        # self.modes_model = [self.modes[0]]
        # self._theta_true()
        # self.plot_posterior()

    def run_sampler(self):
        nDims = len(self.theta_true)
        nDerived = 1
        
        path = "z-"+str(self.redshift)+"_M-"+str(self.final_mass)
        for mode in self.modes:
            path += '-'
            mode = mode[1]+mode[3]+mode[5]
            path += mode

        if not os.path.exists("chains/"+path):
            os.makedirs("chains/"+path)
        if not os.path.exists("chains/clusters/"+path):
            os.makedirs("chains/clusters/"+path)

        filename = path + '/qnm_data'
        for mode in self.modes_data:
            filename += '-'
            mode = mode[1]+mode[3]+mode[5]
            filename += mode
        filename += '_model'
        for mode in self.modes_model:
            filename += '-'
            mode = mode[1]+mode[3]+mode[5]
            filename += mode
        
        settings = PolyChordSettings(nDims, nDerived)
        settings.file_root = filename
        settings.nlive = 100
        settings.do_clustering = True
        settings.read_resume = False


        output = pypolychord.run_polychord(
            self.loglikelihood, 
            nDims,
            nDerived,
            settings,
            self.prior,
            self.dumper)
        self._parameters_labels()
        output.make_paramnames_files(self.labels)
        return output, filename

    def plot_posterior(self):
        output, filename = self.run_sampler()

        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.family"] = "STIXGeneral"
        posterior = output.posterior
        g = getdist.plots.getSubplotPlotter()
        pars = []
        markers = {}
        for i in range(len(self.theta_true)):
            pars.append('p'+str(i))
            markers['p'+str(i)] = self.theta_true[i]
        
        g.triangle_plot(posterior, pars, filled=True, markers = markers)
        g.export('chains/'+filename+'.pdf')


    def loglikelihood(self, theta):
        return MCMCFunctions.log_likelihood_qnm(theta,
            self.model_function, self.data, self.detector["freq"], self.detector["psd"]
            ), sum(theta**2)

    def dumper(self, live, dead, logweights, logZ, logZerr):
        print("Last dead point:", dead[-1])


    def prior(self, hypercube):
        """ Uniform prior from [true/100,true*100]. """
        cube = np.array(hypercube)
        for i in range(len(self.modes_model)):
            cube[0+4*i] = UniformPrior(0, 10)(cube[0 + 4*i])
            cube[1+4*i] = UniformPrior(0, 2*np.pi)(cube[1 + 4*i])
            # cube[2+4*i] = UniformPrior(self.theta_true[2 + 4*i]/100, self.theta_true[2 + 4*i]*100)(cube[2 + 4*i])
            cube[2+4*i] = UniformPrior(0, 5000)(cube[2 + 4*i])
            cube[3+4*i] = UniformPrior(self.theta_true[3 + 4*i]/100, self.theta_true[3 + 4*i]*100)(cube[3 + 4*i])
        return cube


    def _theta_true(self):
        """Injected parameters.
        """
        self.theta_true = []
        for mode in self.modes_model:
            self.theta_true.extend([self.qnm_modes[mode].amplitude,
                float(self.qnm_modes[mode].phase),
                self.qnm_modes[mode].frequency*1e-2,
                self.qnm_modes[mode].decay_time*1e3])
        self.theta_true = tuple(self.theta_true)

    def _parameters_labels(self):
        self.labels = []
        i = 0
        for mode in self.modes_model:
            mode = mode[1]+mode[3]+mode[5]
            # if mode == "(2,2,1) I" or mode == "(2,2,1) II":
            #     mode = "(2,2,1)"
            self.labels.extend([
                ("p{}".format(0+4*i), r"A_{{{0}}}".format(mode)),
                ("p{}".format(1+4*i), r"\phi_{{{0}}}".format(mode)),
                ("p{}".format(2+4*i), r"f_{{{0}}} [Hz]".format(mode)),
                ("p{}".format(3+4*i), r"\tau_{{{0}}} [ms]".format(mode))
                ])
            i+=1
        self.labels += [('r*', 'r')]

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
            A, phi, freq, tau = theta[0 + 4*i: 4 + 4*i]
            freq *= 1e2
            tau *= 1e-3
            omega_r = freq*2*np.pi*self.time_convert
            omega_i = self.time_convert/tau
            h_model += self.time_convert*self.amplitude_scale*GWFunctions.compute_qnm_fourier(
                self.detector["freq"]*self.time_convert, A, phi, omega_r, omega_i, 
                part = "real", convention = self.ft_convention)
        return h_model


if __name__ == '__main__':
    from datetime import datetime
    np.random.seed(1234)
    start=datetime.now()
    m_f = 500
    z = 0.1
    q = 1.5
    detector = "LIGO"
    modes = ("(2,2,0)", "(2,2,1) I")
    # modes = ("(2,2,0)", "(4,4,0)")
    # modes = ["(2,2,0)"]
    teste = Polychord(detector, modes, m_f, z, q, "FH")
    teste.run_all()
    print(datetime.now()-start)
