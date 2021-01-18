import os
import numpy as np
from scipy import interpolate
import h5py
import re


def import_detector(detector, interpolation = False):
    """Import detector noise power specrtal density

    Parameters
    ----------
        detector: string 
            Detector name 'LIGO', 'LISA', 'CE' = 'CE2silicon', 'CE2silica' or 'ET'
        interpolation: bool
            Interpolate noise PSD

    Returns
    -------
        Dictionary: Returns detector frequency array relative to detector psd and detector label. If interpolation = true, also returns interpolated function.
    """
    # choose noise
    noise = {}
    i_freq = 0
    i_psd = 1
    if detector == "LIGO":
        file_name = "aLIGODesign.txt"
        noise["label"] = "LIGO - Design sensitivity"
    elif detector == "LISA":
        file_name = "LISA_Strain_Sensitivity_range.txt"
        noise["label"] = "LISA sensitivity"
    elif detector == "ET":
        i_psd = 3
        file_name = "ET/ETDSensitivityCurve.txt"
        noise["label"] = "ET_D sum sensitivity"
    elif detector == "CE" or detector == "CE2silicon":
        file_name = "CE/CE2silicon.txt"
        noise["label"] = "CE silicon sensitivity"
    elif detector == "CE2silica":
        file_name = "CE/CE2silica.txt"
        noise["label"] = "CE silica sensitivity"
    else:
        raise ValueError("Wrong detector option! Choose \"LIGO\", \"LISA\", \"CE\" = \"CE2silicon\", \"CE2silica\" or \"ET\"")
    
    file_path = os.path.join(os.getcwd(), "../detectors", file_name)
    noise_file = np.genfromtxt(file_path)
    noise["freq"], noise["psd"] = noise_file[:,i_freq], noise_file[:,i_psd]
    # make noise arrays immutable arrays
    noise["freq"].flags.writeable = False
    noise["psd"].flags.writeable = False
    
    if interpolation == False:
        return noise
    else:
        itp = interpolate.interp1d(noise["freq"], noise["psd"], "cubic")
        return noise, itp

def import_simulation_qnm_parameters(q_mass):
    folders_path = os.path.join(os.getcwd(), "../q_change")
    for folders in os.listdir(folders_path):
        if folders.find(str(q_mass)) != -1:     
            simu_folder = folders_path+"/"+folders

            with open(simu_folder+"/import_data/metadata.txt", 'rt') as metadata:
                for line in metadata:
                    if 'remnant-mass' in line:
                        mass_f = line   
                        mass_f = [float(s) for s in re.findall(r"\d+\.\d+", mass_f)][0]

            ratios, amplitudes, phases, omega = [{} for i in range(4)]
            parameters = [[ratios, "ratios"], [amplitudes, "amplitudes"], 
            [phases, "phases"], [omega, "omega"]]
            for par in parameters:
                with h5py.File(simu_folder+"/arrays/fits/"+par[1]+".h5", "r") as file:
                    for keys in file.keys():
                        par[0][keys] = np.asarray(file[keys])
    ratios["(2,2,0)"] = 1.

    for k in ratios.keys():
        amplitudes[k] = np.abs(amplitudes[k])
        ratios[k] = np.abs(ratios[k])
        while phases[k] > 2*np.pi:
            phases[k] -= 2*np.pi
    
    modes = dict()
    for k in amplitudes.keys():
        modes[k] = dict()
        for pars in parameters:
            modes[k][pars[1]] = pars[0][k]

    return modes, mass_f
