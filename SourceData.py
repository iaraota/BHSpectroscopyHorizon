import json
import numpy as np
from modules import GWFunctions, MCMCFunctions, ImportData, PlotFunctions
import os
import pandas as pd

class SourceData:
    """Generate waveforms, noise and injected data
    from choosen QNMs and detector in frequency domain.

    Parameters
    ----------
    detector : str
        Detector used to compute noise, can be set to
        "LIGO", "LISA", "ET", "CE = CE2silicon" and "CE2silica".
    modes : list
        Quasinormal modes considered.
    final_mass : float
        Final mass in the source frame.
    redshift : float
        SOurce redshift.
    q_mass : float
        Mass ratio of the binary.
    convention : str, optional
        Convention used to compute de Fourier transform of the QNMs,
        can be set do "FH" or "EF", by default "FH"
    """

    def __init__(self,
        detector:str,
        final_mass:float,
        redshift:float,
        q_mass:float,
        convention:str="FH"):

        self.final_mass = final_mass
        self.redshift = redshift
        self.q_mass = q_mass
        self.ft_convention = convention

        
        # get QNM parameters from simulation
        self.qnm_pars, self.mass_f, self.final_spin = self.import_simulation_qnm_parameters(self.q_mass)
        
        # import detector strain
        #self.detector = ImportData.import_detector(detector, False)
        self._import_detector_psd(detector)

        # Compute inifical mass
        self.initial_mass = self.final_mass/self.mass_f

        # get convertion factor for time and amplitude
        self.time_convert, self.amplitude_scale = GWFunctions.convert_units(
            self.final_mass, self.redshift, self.mass_f)

        # compute QNMs waveforms
        self.dist_Gpc = GWFunctions.luminosity_distance(self.redshift)*1e-3
        self._compute_qnm_modes()

        # compute final spin in final mass units
        self.mass_initial = self.final_mass/self.mass_f
        # compute noise
        self._random_noise()

        # compute tables

        self.fit_coeff = {}
        for mode in self.qnm_modes.keys():
            self.fit_coeff[mode] = self.transf_fit_coeff(mode)

        self.df_a_omegas = {}
        for mode in self.qnm_modes.keys():
            self.df_a_omegas[mode] = self.create_a_over_M_omegas_dataframe(mode)

    def _compute_qnm_modes(self):
        """Comptute QNM waveforms in frequency domain.
        """
        qnm_modes = dict()
        for (k,v) in self.qnm_pars.items():
            qnm_modes[k] = GWFunctions.QuasinormalMode(v["amplitudes"], v["phases"], v['omegas']['omega_r'], 
                            v['omegas']['omega_i'], self.final_mass, self.redshift, self.mass_f)
            qnm_modes[k].qnm_f = {
                "real": qnm_modes[k].qnm_fourier(self.detector["freq"],
                        "real", self.ft_convention, "SI"),
                "imag": qnm_modes[k].qnm_fourier(self.detector["freq"],
                        "imag", self.ft_convention, "SI")
                }
            # qnm_modes[k].qnm_t = strain_unit*qnm_modes[k].qnm_time(times/time_unit, part, "NR")
        self.qnm_modes = qnm_modes
    def _import_detector_psd(
        self,
        detector:str,
        ):
        """Import interpolated detector and create array with lower df.

        Parameters
        ----------
        detector : str
            Detector's name
        """
        #TODO: freqs are not compatible with LISA
        # import detector data and interpolation function
        detector_data, itp_detector = ImportData.import_detector(detector, True)

        # get minimun and maximum frequencies and smallest df
        f_min, f_max = min(detector_data["freq"]), max(detector_data["freq"])
        df = detector_data["freq"][1] - detector_data["freq"][0]
        df = 0.1
        
        # taus = []
        # for (k,v) in self.qnm_pars.items():
        #     qnm = GWFunctions.QuasinormalMode(v["amplitudes"], v["phases"], v['omegas']['omega_r'], 
        #                     v['omegas']['omega_i'], self.final_mass, self.redshift, self.mass_f)
        #     taus.append(qnm.decay_time)
        # tau_max = max(taus)
        # print(tau_max)
        # T = 1000*tau_max
        # df = 1/T

        # create frequency array
        # freqs_1 = np.arange(f_min, 100, df)
        # freqs_2 = np.arange(100+1*df, 1000, 10*df)
        # freqs_3 = np.arange(1000+10*df, f_max, 100*df)
        # freqs = np.concatenate((freqs_1, freqs_2, freqs_3))
        freqs = np.arange(f_min, f_max, df)
        
        detector_psd = itp_detector(freqs)
        self.detector = {
            "freq": freqs,
            "psd": detector_psd,
            "label": detector_data["label"],
        }

    def _random_noise(self):
        """Generate noise in frequency domain.
        """
        N_data = len(self.detector["psd"]) 
        df = self.detector["freq"][1] - self.detector["freq"][0]
        # dfs =  np.diff(self.detector["freq"])
        # dfs = np.append(dfs, dfs[-1])
        
        sigma = 0.5 * (self.detector["psd"]**2 /df) ** (0.5)

        not_zero = (sigma != 0)

        sigma_red = sigma[not_zero]
        noise_re = np.random.normal(0, sigma_red)
        noise_co = np.random.normal(0, sigma_red)
        noise_red = noise_re + 1j * noise_co

        noise = np.zeros(len(sigma), dtype='complex128')
        noise[not_zero] = noise_red        

        self.noise = noise

        # generate random noise from PSD
        # self.noise = self.detector["psd"]*np.exp(1j*np.random.uniform(0,np.pi*2,N_data))*(1+np.random.uniform(np.nextafter(-1,0),1,N_data))
        # make noise an immutable array
        self.noise.flags.writeable = False

    def inject_data(self, modes_data):
        """Generate data from noise and QNM waveform.
        """
        # d = n
        self.data = np.copy(self.noise)
        # d = n + modes
        angular_mean = np.sqrt(1/5/4/np.pi)
        # angular_mean = 1

        for mode in modes_data:
            self.data += angular_mean*(
                self.qnm_modes[mode].qnm_f["real"]
                + self.qnm_modes[mode].qnm_f["imag"]
            )

    def transf_fit_coeff(
        self,
        mode:str,
        ):
        """Fits coefficients to Kerr QNM frequencies 
        to transform mass M and spin a to frequency and time.
        https://pages.jh.edu/eberti2/ringdown/

        files columns: l, m, n, f1, f2, f3, q1, q2, q3
        l, m, n are the quasinormal modes indices

        fit formulas:
        M*omega_r = f1 + f2*(1 - a/M)^f3
        Q = q1 + q2*(1 - a/M)^q3
        Q = omega_r/(2*omega_i)

        Parameters
        ----------
        mode : str
            quasinormal mode to get the coefficients

        Returns
        -------
        tuple
            Fit coefficients f1, f2, f3, q1, q2, q3 for the chosen mode.
        """
        file_path = os.path.join(os.getcwd(), "..", "fitcoeffs.dat")
        l = float(mode[1])
        m = float(mode[3])
        n = float(mode[5])
        file = np.genfromtxt(file_path)
        
        df = pd.DataFrame(file, 
            columns = ['l', 'm', 'n', 'f1', 'f2', 'f3', 'q1', 'q2', 'q3'])

        f1,f2,f3,q1,q2,q3 = df[
            (df['l'] == l) & (df['m'] == m) & (df['n'] == n)][
                ['f1', 'f2', 'f3', 'q1', 'q2', 'q3']
            ].values[0]
        return (f1,f2,f3,q1,q2,q3)

    def create_a_over_M_omegas_dataframe(
        self,
        mode:str,
        ):
        """Creates a pandas DataFrame with a/M as index
        and M*omega_r and M*omega_i as columnns. Data 
        file available at 
        https://pages.jh.edu/eberti2/ringdown/

        Parameters
        ----------
        mode : str
            '(l,m,n)' mode to get the DataFrame.

        Returns
        -------
        pandas DataFrame
            DataFrame with a/M as index and M*omega_r 
            and M*omega_i as columnns.
        """

        files = np.genfromtxt(f'../frequencies_l{mode[1]}/n{str(int(mode[5])+1)}l{mode[1]}m{mode[3]}.dat', usecols=range(3))

        df = pd.DataFrame({"omega_r": files[:,1], "omega_i": -files[:,2]}, index = files[:,0])

        return df

    def transform_mass_spin_to_omegas(
        self,
        M:float,
        a_over_M:float,
        df:'DataFrame',
        ):
        """Transform mass and spin do quasinormal mode omegas (frequencies)

        Parameters
        ----------
        M : float
            Black hole final mass in units of initial mass.
            (M_final/M_initial)
        a_over_M : float
            Black hole spin in units of initial final mass.
        df : pandas DataFrame
            DataFrame containing a_over_M as index, omega_r
            and omega_i columns (in units of final mass). 
            Dataframe values computed with method 
            'create_a_over_M_omegas_dataframe'.

        Returns
        -------
        float, float
            Quasinormal mode frequencies in NR units.
        """
        omega_r = df.loc[round(a_over_M,4)].omega_r/M
        omega_i = df.loc[round(a_over_M,4)].omega_i/M
        return omega_r, omega_i

    def transform_omegas_to_mass_spin(
        self,
        omega_r:float,
        omega_i:float,
        df,
        fit_coeff:list,
        ):
        """Transform mass and spin do quasinormal mode omegas (frequencies)

        Parameters
        ----------
        omega_r : float
            qnm real frequency in code units (initial mass).
        omega_i : float
            qnm imaginary frequency in code units (initial mass).
        fit_coeff : array_like
            Fits coefficient to Kerr QNM frequencies. 
            See transf_fit_coeff method or  
            https://pages.jh.edu/eberti2/ringdown/

        Returns
        -------
        float, float
            Black hole mass and spin both in units of initial mass.
        """

        f1,f2,f3,q1,q2,q3 = fit_coeff

        factor = ((omega_r/(2*omega_i) - q1)/q2)**(1/q3)
        M = (f1 + f2*factor**f3)/omega_r
        a_over_M = (1 - factor)
        
        
        wr_aux = df.loc[round(a_over_M,4)].omega_r
        wi_aux = df.loc[round(a_over_M,4)].omega_i
        M = wr_aux/omega_r

        return M, a_over_M

    def import_simulation_qnm_parameters(
        self,
        q_mass,
        ):
        folders_path = os.path.join(os.getcwd(), "../simulations")
        for folders in os.listdir(folders_path):
            if folders.find(str(q_mass)) != -1:     
                simu_folder = folders_path+'/'+folders+'/data/qnm_pars/'

                parameters = {}
                for par in ('ratios', 'amplitudes', 'phases', 'omegas', 'bh_pars'):
                    with open(f'{simu_folder}{par}.json', 'r') as file:
                        parameters[par] = json.load(file)
                        
        parameters['omegas']['(2,2,1) I'] = parameters['omegas']['(2,2,1)']
        parameters['omegas']['(2,2,1) II'] = parameters['omegas']['(2,2,1)']

        mass_f = parameters['bh_pars']['remnant_mass']
        final_spin = parameters['bh_pars']['remnant_spin']
        del parameters['bh_pars']

        modes = {}
        for mode in parameters['ratios']:
            modes[mode] = {}
            for (par, value) in parameters.items():
                modes[mode][par] = value[mode]

        return modes, mass_f, final_spin


    def __str__(self):
        return ('Create QNMs for a binary with:\n\t' 
            +f'mass ratio {self.q_mass},\n\t'
            +f'final mass {self.final_mass},\n\t'
            +f'redshift {self.redshift}\n\t'
            +f'and {self.detector["label"]} detector.\n\n'
            +'The method inject_data(modes_data) injects the selected QNMs in the detector noise.'
            )

    def __repr__(self):
        return (f'Create QNMs for a binary with:\n\t' 
            + 'mass ratio {self.q_mass},\n\t'
            +'final mass {self.final_mass},\n\t'
            +'redshift {self.redshift}\n\t'
            +'and {self.detector["label"]} detector.\n\n'
            +'The method inject_data(modes_data) injects the selected QNMs in the detector noise.'
            )



if __name__ == '__main__':

    m_f = 500
    z = 0.1
    q = 1.5
    detector = "LIGO"
    teste = SourceData(detector, m_f, z, q, "FH")
    fits = teste.transf_fit_coeff("(2,2,0)")
    df = teste.create_a_over_M_omegas_dataframe("(2,2,0)")
    M, a = teste.transform_omegas_to_mass_spin(teste.qnm_modes["(2,2,0)"].omega_r, teste.qnm_modes["(2,2,0)"].omega_i,df, fits)
    omega_r, omega_i = teste.transform_mass_spin_to_omegas(teste.final_mass/teste.initial_mass, teste.final_spin, df)
    # M, a = teste.transform_omegas_to_mass_spin(omega_r, omega_i, fits)
    print(teste.mass_f)
    print(teste.final_spin)
    print(M, a)
    print(teste.qnm_modes["(2,2,0)"].omega_r, teste.qnm_modes["(2,2,0)"].omega_i)
    print(omega_r, omega_i)
