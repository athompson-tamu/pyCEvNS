# Primakoff explanation of excess
import numpy as np
from numpy import sqrt, pi
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.pylab import rc
import matplotlib.ticker as tickr
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

import sys
sys.path.append("../")
from pyCEvNS.axion import primakoff_scattering_xs_CAF
from pyCEvNS.constants import *


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

# Read in the Background model
B0_XENON1T = np.genfromtxt("data/XENON1T_B0.txt", delimiter=",")

# Read in the tritium model
H3_XENON1T = np.genfromtxt("data/XENON1T_3H.txt", delimiter=",")

# Read in the ABC flux
FILE_ABC = np.genfromtxt("data/gaeflux.txt")

# Read in the XENON1T ABC and Primakoff response data.
PrimakoffFlux_XENON1T = np.genfromtxt("data/primakoff_flux_XENON1T_compton_response.txt", delimiter=",")
ABCFlux_XENON1T = np.genfromtxt("data/abc_flux_XENON1T_compton_response.txt", delimiter=",")

# Plot the data
if False:
    plt.errorbar(energy_bins, observations, yerr=errors, color='k', marker='o', capsize=2, ls='none', alpha=0.8)
    plt.plot(B0_XENON1T[:,0], B0_XENON1T[:,1], color='r')
    plt.show()
    plt.close()

eff_data = np.genfromtxt("reprod/efficiency.txt", delimiter=',')
eff_interp = interp1d(eff_data[:,0],eff_data[:,3],fill_value=0,bounds_error=False)

def eff(er):
    a = 0.87310709
    k = 3.27543615
    x0 = 1.50913422
    return a / (1 + np.exp(-k * (er - x0)))

def eff2(erg):
    return eff_interp(erg)

energies = np.linspace(0,30,1000)
plt.plot(energies, eff(energies))
plt.plot(energies, eff2(energies))
plt.show()
plt.close()

# Solar axion flux
def FluxPrimakoff(ea, g): # g: GeV^-1; ea: keV
    return 6e30 * g**2 * np.power(ea, 2.481) * np.exp(-ea / 1.205) # in keV^-1 cm^-2 s^-1

# Solar ABC flux
def FluxABC(keV, gae): # in 1/(keV cm day)
    coupling = 1e19 * (gae/(0.511*1e-10))**2 / 24 / 3600
    smeared_events = coupling * gaussian_filter(FILE_ABC[:,1], sigma=10, mode="nearest")
    return np.interp(keV, FILE_ABC[:,0], smeared_events)

def FluxFe57(keV, gn):
    return 0.6 * 4.56e23 * gn**2 * np.exp(-(keV - 14.4)**2 / 3) # in cm^-2 s^-1


# ABC flux, compton response
def FluxABCComptonResponse(keV, ge):
    return (ge / 5e-12)**4 * np.interp(keV, ABCFlux_XENON1T[:,0], ABCFlux_XENON1T[:,1])

# Primakoff flux, compton response
def FluxPrimakoffComptonResponse(keV, gg, ge):
    return (ge / 5e-12)**2 * (gg / 2e-10)**2 * np.interp(keV, PrimakoffFlux_XENON1T[:,0], PrimakoffFlux_XENON1T[:,1])

def PrimakoffRate(keV, g, ma): # g in GeV^-1
    # primakoff_scattering_xs: MeV^-2
    # prefactor: s m^2 MeV^-2
    return prefactor * primakoff_scattering_xs_CAF(keV/1000, 1e-3 * g, ma, Z_Xe, r0Xe)

def energy_resolution(erg):
    return erg*(31.71/np.sqrt(erg) + 0.15)/100.0


# Signal model
def PrimakoffEventsGenSmeared(keV, g, ma):
    # FluxPrimakoff: s^-1 cm^-2 keV^-1
    # PriamkoffRate: s m^2
    def wrapper_P_2(x, x0):
        return PrimakoffRate(x,g,ma)*FluxPrimakoff(x, g)*norm.pdf(x, x0, energy_resolution(x0))

    integrals_P_2 = np.array([quad(wrapper_P_2, 0.0, np.inf, args=(x0), epsabs=0, epsrel=1.0e-8)[0] for x0 in keV])
    events = m2_to_cm2*integrals_P_2 * eff(keV)
    #events = m2_to_cm2*np.array([FluxPrimakoff(ea, g)*PrimakoffRate(ea,g,ma)*eff(ea) for ea in keV])
    return events

def EventsGeneratorABCPlot(keV, ge, gg):
    ma = 1e-6
    def wrapper_P_2(x, x0):
        return FluxABC(x, ge)*PrimakoffRate(x,gg,ma)*norm.pdf(x, x0, energy_resolution(x0))

    integrals_P_2 = np.array([quad(wrapper_P_2, 0.0, np.inf, args=(x0), epsabs=0, epsrel=1.0e-8)[0] for x0 in keV])
    events = m2_to_cm2*integrals_P_2 * eff(keV)
    #events = m2_to_cm2*np.array([FluxABC(ea, ge)*PrimakoffRate(ea,gg,ma)*eff(ea) for ea in keV])
    return events + B0(keV)

def EventsGeneratorFe57(keV, gn, gg):
    ma = 1e-6
    def wrapper_P_2(x, x0):
        return FluxFe57(x, gn)*PrimakoffRate(x,gg,ma)*norm.pdf(x, x0, energy_resolution(x0))

    integrals_P_2 = np.array([quad(wrapper_P_2, 0.0, np.inf, args=(x0), epsabs=0, epsrel=1.0e-8)[0] for x0 in keV])
    events = m2_to_cm2*integrals_P_2 * eff(keV)
    #events = m2_to_cm2*np.array([FluxFe57(ea, gn)*PrimakoffRate(ea,gg,ma)*eff(ea) for ea in keV])
    return events + B0(keV)


# Background model
def B0(keV):
    return np.interp(keV, B0_XENON1T[:,0], B0_XENON1T[:,1])

# Tritium model
def H3(keV):
    return np.interp(keV, H3_XENON1T[:,0], H3_XENON1T[:,1])

background = B0(energy_bins)
h3 = H3(energy_bins)

def EventsGeneratorPlot(keV, g, ma):
    return B0(keV) + PrimakoffEventsGenSmeared(keV, g,ma)

def EventsGenerator(cube):
    g = np.power(10.0, cube[0])  # in GeV^-1
    ma = np.power(10.0,cube[1]-3) # convert keV to MeV with -3
    events = m2_to_cm2*np.array([FluxPrimakoff(ea, g)*PrimakoffRate(ea,g,ma)*eff(ea) for ea in energy_bins])
    return background + events

def EventsGeneratorPrimakoffH3(cube):
    g = np.power(10.0, cube[0])  # in GeV^-1
    ma = np.power(10.0,cube[1]-3) # convert keV to MeV with -3
    events = m2_to_cm2*np.array([FluxPrimakoff(ea, g)*PrimakoffRate(ea,g,ma)*eff(ea) for ea in energy_bins])
    return background + cube[2]*h3 + events

def EventsGeneratorH3(cube):
    return background + cube[0]*h3

def EventsGeneratorABC(cube):
    gg = np.power(10.0, cube[0])  # in GeV^-1
    ge = np.power(10.0, cube[1])
    ma = 1e-6
    events_1 = m2_to_cm2*np.array([FluxPrimakoff(ea, gg)*PrimakoffRate(ea,gg,ma)*eff(ea) for ea in energy_bins])
    events_2 = m2_to_cm2*np.array([FluxABC(ea, ge)*PrimakoffRate(ea,gg,ma)*eff(ea) for ea in energy_bins])
    events_3 = FluxPrimakoffComptonResponse(energy_bins, gg, ge)
    events_4 = FluxABCComptonResponse(energy_bins, ge)
    background = B0(energy_bins)
    return background + events_3 + events_4 + events_1 + events_2

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


def TestStatistic(n_signal):
    likelihood_0 = - 0.5 * ((background - observations) / errors) ** 2
    likelihood_1 = - 0.5 * ((n_signal - observations) / errors) ** 2
    return np.sum(likelihood_1) - np.sum(likelihood_0)


# Get significance.
print(TestStatistic(EventsGeneratorH3([1]))) # 3H
print(TestStatistic(EventsGeneratorPrimakoffH3([-10.6, -4.69875, 0.58]))) # P + 3H
print(TestStatistic(EventsGenerator([-9.6232138, -4.6068])))  # P
print(TestStatistic(EventsGeneratorABC([-10.01, -11.557])))  # ABC
print(TestStatistic(EventsGeneratorABCH3([-10, -11.5, 0.01])))  # ABC with 3H


# [-9.564899, -4.69875]

# Make plots
fine_energy_edges = np.linspace(0, 30, 60)
fine_energies = (fine_energy_edges[1:]+fine_energy_edges[:-1])/2

# Benchmark mass models
test_ge = 5e-12
test_gn = 0.5e-6
test_gg = 3.0e-10
signal_primakoff = EventsGeneratorPlot(fine_energies, test_gg, 0.00000001)
signal_abc = EventsGeneratorABCPlot(fine_energies, test_ge, test_gg)
signal_fe57 = EventsGeneratorFe57(fine_energies, test_gn, test_gg)


if False:
    signal_test = np.array([PrimakoffRate(er,5e-10,0.0005) for er in fine_energies])
    plt.hist(fine_energies, weights=FluxPrimakoff(fine_energies, 5e-10),
             bins=fine_energy_edges, density=True, histtype='step', label="Flux")
    plt.hist(fine_energies, weights=signal_test, bins=fine_energy_edges, density=True,
             histtype='step', label="XS")
    plt.hist(fine_energies, weights=PrimakoffEventsGen(fine_energies, 5e-10, 0.0005), bins=fine_energy_edges, density=True,
             histtype='step', label="XS*flux")
    plt.legend()
    plt.show()
    plt.close()

if True:
    plt.errorbar(energy_bins, observations, yerr=errors, color='k', marker='o', capsize=2, ls='none',
                alpha=0.8, label="XENON1T SR1 Data")
    plt.plot(fine_energies, B0(fine_energies), color='r', label=r"XENON1T $B_0$ model")
    #plt.plot(fine_energies, H3(fine_energies), color='purple', label=r"$^3$H")
    plt.plot(fine_energies, signal_abc, color="blue", ls="dotted", label=r"ABC Flux + Inverse Primakoff")
    plt.plot(fine_energies, signal_primakoff, color="red", ls="dotted", label=r"Primakoff Flux + Inverse Primakoff")
    plt.plot(fine_energies, signal_fe57, color="green", ls="dotted", label=r"$^{57}$Fe Flux + Inverse Primakoff")

    plt.xlabel(r"$E$ [keV]", fontsize=15)
    plt.ylabel(r"Events / (t$\cdot$y$\cdot$keV)", fontsize=15)
    plt.title(r"$g_{a\gamma} = 3\cdot10^{-10}$ GeV$^{-1}$, $g_{ae} = 5\cdot10^{-12}$, $g_{an}^{eff} = 5\cdot10^{-7}$", loc="right", fontsize=12)
    plt.legend(framealpha=1.0, loc='upper right', fontsize=12)
    plt.xlim((0, 30))
    plt.ylim((0,150))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()
    plt.close()


