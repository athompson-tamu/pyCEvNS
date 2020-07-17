# Primakoff explanation of excess
import numpy as np
from numpy import sqrt, pi

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.integrate import quad

import sys
sys.path.append("../")
from pyCEvNS.axion import primakoff_scattering_xs_complete,primakoff_scattering_xs_CAF
from pyCEvNS.constants import *

import pymultinest





# Define constants
Z_Xe = 54
r0Xe = 2.45e-10 / meter_by_mev
det_mass = 1000 # kg
exposure_time = 365 * 3600 * 24 # s
n_atoms_Xe = det_mass * mev_per_kg / 122.3e3
prefactor = exposure_time * n_atoms_Xe * meter_by_mev ** 2  # s m^2 MeV^2
m2_to_cm2 = 10000


# Read in data
FILE_XENON1T = np.genfromtxt("data/XENON1T.txt", delimiter=",")
energy_edges = np.arange(1, 31, 1)
energy_bins = (energy_edges[1:] + energy_edges[:-1])/2
observations = FILE_XENON1T[:,1]
errors = np.sqrt(observations)

# Read in the ABC flux
FILE_ABC = np.genfromtxt("data/gaeflux.txt")

# Read in the Background model
B0_XENON1T = np.genfromtxt("data/XENON1T_B0.txt", delimiter=",")

# Read in the tritium model
H3_XENON1T = np.genfromtxt("data/XENON1T_3H.txt", delimiter=",")

# Read in the XENON1T ABC and Primakoff response data.
PrimakoffFlux_XENON1T = np.genfromtxt("data/primakoff_flux_XENON1T_compton_response.txt", delimiter=",")
ABCFlux_XENON1T = np.genfromtxt("data/abc_flux_XENON1T_compton_response.txt", delimiter=",")



# Experimental Efficiency
def eff(er):
    a = 0.87310709
    k = 3.27543615
    x0 = 1.50913422
    return a / (1 + np.exp(-k * (er - x0)))

# Solar axion flux
def FluxPrimakoff(ea, g): # g: GeV^-1; ea: keV
     # in keV^-1 cm^-2 s^-1
    return 6e30 * g**2 * np.power(ea, 2.481) * np.exp(-ea / 1.205)

# ABC Flux model
def FluxABC(keV, gae): # in 1/(keV cm day)
    coupling = 1e19 * (gae/(0.511*1e-10))**2 / 24 / 3600
    smeared_events = coupling * gaussian_filter(FILE_ABC[:,1], sigma=10, mode="nearest")
    return np.interp(keV, FILE_ABC[:,0], smeared_events)

def PrimakoffRate(keV, g, ma):
    # g in GeV^-1 (it gets converted to MeV^-1)
    # er in keV
    # primakoff_scattering_xs: MeV^-2
    # prefactor: s m^2 MeV^-2
    #return prefactor * primakoff_scattering_xs_complete(keV/1000, 1e-3 * g, ma, Z_Xe, r0Xe)
    #return prefactor * primakoff_scattering_xs(keV/1000,Z_Xe,ma,1e-3 * g)
    return prefactor * primakoff_scattering_xs_CAF(keV/1000, 1e-3 * g, ma, Z_Xe, r0Xe)

# ABC flux, compton response
def FluxABCComptonResponse(keV, ge):
    return (ge / 5e-12)**4 * np.interp(keV, ABCFlux_XENON1T[:,0], ABCFlux_XENON1T[:,1])

# Primakoff flux, compton response
def FluxPrimakoffComptonResponse(keV, gg, ge):
    return (ge / 5e-12)**2 * (gg / 2e-10)**2 * np.interp(keV, PrimakoffFlux_XENON1T[:,0], PrimakoffFlux_XENON1T[:,1])


# Background model
def B0(keV):
    return np.interp(keV, B0_XENON1T[:,0], B0_XENON1T[:,1])

# Tritium model
def H3(keV):
    return np.interp(keV, H3_XENON1T[:,0], H3_XENON1T[:,1])

background = B0(energy_bins)
h3 = H3(energy_bins)

# UNITS ###
# g in GeV^-1
# er in keV
# FluxPrimakoff: s^-1 cm^-2 keV^-1
# PriamkoffRate: s m^2

# Events generators
def EventsGenerator(cube):
    g = np.power(10.0, cube[0])  # in GeV^-1
    ma = np.power(10.0,cube[1]-3) # convert keV to MeV with -3
    events = m2_to_cm2*np.array([FluxPrimakoff(ea, g)*PrimakoffRate(ea,g,ma)*eff(ea) for ea in energy_bins])
    return background + events

def EventsGeneratorH3(cube):
    return background + cube[0]*h3

def EventsGeneratorPrimakoffH3(cube):
    g = np.power(10.0, cube[0])  # in GeV^-1
    ma = 1e-9
    events = m2_to_cm2*np.array([FluxPrimakoff(ea, g)*PrimakoffRate(ea,g,ma)*eff(ea) for ea in energy_bins])
    return background + cube[1]*h3 + events

def EventsGeneratorABC(cube):
    gg = np.power(10.0, cube[0])  # in GeV^-1
    ge = np.power(10.0, cube[1])
    ma = 1e-6
    events_1 = m2_to_cm2*np.array([FluxPrimakoff(ea, gg)*PrimakoffRate(ea,gg,ma)*eff(ea) for ea in energy_bins])
    events_2 = m2_to_cm2*np.array([FluxABC(ea, ge)*PrimakoffRate(ea,gg,ma)*eff(ea) for ea in energy_bins])
    events_3 = FluxPrimakoffComptonResponse(energy_bins, gg, ge)
    events_4 = FluxABCComptonResponse(energy_bins, ge)
    background = B0(energy_bins)
    return events_3 + events_4 + events_1 + events_2

def EventsGeneratorABCH3(cube):
    gg = np.power(10.0, cube[0])  # in GeV^-1
    ge = np.power(10.0, cube[1])
    ma = 1e-6
    events_1 = m2_to_cm2*np.array([FluxPrimakoff(ea, gg)*PrimakoffRate(ea,gg,ma)*eff(ea) for ea in energy_bins])
    events_2 = m2_to_cm2*np.array([FluxABC(ea, ge)*PrimakoffRate(ea,gg,ma)*eff(ea) for ea in energy_bins])
    events_3 = FluxPrimakoffComptonResponse(energy_bins, gg, ge)
    events_4 = FluxABCComptonResponse(energy_bins, ge)
    background = B0(energy_bins)
    return background + events_3 + events_4 + events_1 + events_2 + cube[2]*h3






# Set up likelihoods and priors
def Likelihood(cube, D, N):
    n_signal = EventsGeneratorABCH3(cube)
    #n_signal = EventsGeneratorPrimakoffH3(cube)
    #n_signal = EventsGenerator(cube)
    #n_signal = EventsGeneratorABC(cube)
    #n_signal = cube[0]*background #EventsGeneratorH3(cube)
    likelihood = -0.5 * np.log(2 * pi * errors ** 2) - 0.5 * ((n_signal + h3 - observations) / errors) ** 2
    return np.sum(likelihood)


nton_background = 1000*B0(energy_bins)
nton_errors = np.sqrt(nton_background)
def LikelihoodExclusions(cube, D, N):
    n_signal = 1000*EventsGeneratorABC(cube)
    likelihood = -0.5 * np.log(2 * pi * errors ** 2) - 0.5 * ((n_signal) / errors) ** 2
    return np.sum(likelihood)

def PriorH3(cube, N, D):
    cube[0] = 2*cube[0]

def PriorPrimakoffH3(cube, N, D):
    cube[0] = cube[0]*(-6)-6 # coupling in GeV^-1
    cube[1] = (2*cube[1] - 1)

def PriorPrimakoff(cube, N, D):
    cube[0] = cube[0]*(-6)-6 # coupling in GeV^-1
    cube[1] = -7*cube[1] + 1# mass in eV

def PriorABC(cube, N, D):
    cube[0] = cube[0]*(-5)-7 # a-photon coupling in GeV^-1
    cube[1] = cube[1]*(-4)-11 # aee coupling
    #cube[1] = 1e-11*cube[1]

def PriorABCH3(cube, N, D):
    cube[0] = cube[0]*(-5)-7 # a-photon coupling in GeV^-1
    cube[1] = cube[1]*(-4)-11 # aee coupling
    cube[2] = (2*cube[2] - 1)





def main():
    if False:
        fine_energy_edges = np.linspace(0, 30, 500)
        fine_energies = (fine_energy_edges[1:]+fine_energy_edges[:-1])/2

        # Benchmark mass models
        signal_1keV = EventsGeneratorABC([np.log10(1e-10), np.log10(3e-12), -6], fine_energies)
        plt.errorbar(energy_bins, observations, yerr=errors, color='k', marker='o', capsize=2, ls='none',
                alpha=0.8, label="XENON1T SR1 Data")
        plt.plot(fine_energies, B0(fine_energies), color='r', label="XENON1T B0 model")
        plt.plot(fine_energies, signal_1keV, color="b", ls="dotted", label=r"$g_{a\gamma\gamma} = 1\cdot 10^{-12}, g_{aee} = 3\cdot 10^{-12}$")
        plt.xlabel(r"$E$ [keV]", fontsize=15)
        plt.ylabel(r"Events / (t$\cdot$y$\cdot$keV)", fontsize=15)
        plt.legend(framealpha=1.0, loc='upper right', fontsize=11)
        plt.xlim((0, 30))
        plt.ylim((0,150))
        plt.show()
        plt.close()
    
    pymultinest.run(LikelihoodExclusions, PriorABC, 2,
                    outputfiles_basename="multinest/abc_exclusions/abc_exclusions",
                    resume=False, verbose=True, n_live_points=5000, evidence_tolerance=0.1,
                    sampling_efficiency=0.8)


if __name__ == "__main__":
    main()