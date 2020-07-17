# Primakoff explanation of excess
import numpy as np
from numpy import sqrt, pi, log, interp, exp, power
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Define constants
hbar = 6.58212e-22  # MeV*s
c_light = 2.998e8  # m/s
meter_by_mev = hbar * c_light  # MeV*m
mev_per_kg = 5.6095887e29  # MeV/kg
kAlpha = 1/137
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
errors = sqrt(observations)

# Read in the Background model
B0_XENON1T = np.genfromtxt("data/XENON1T_B0.txt", delimiter=",")

# Read in the ABC flux
FILE_ABC = np.genfromtxt("data/gaeflux.txt")


# Efficiency function (based on fit to XENON plot, used to speed up calculation)
def eff(er):
    a = 0.87310709
    k = 3.27543615
    x0 = 1.50913422
    return a / (1 + exp(-k * (er - x0)))


# Solar Primakoff flux
def FluxPrimakoff(ea, g): # g: GeV^-1; ea: keV
    return 6e30 * g**2 * power(ea, 2.481) * exp(-ea / 1.205) # in keV^-1 cm^-2 s^-1

# Solar ABC flux (Redondo)
def FluxABC(keV, gae): # in 1/(keV cm^2 day)
    coupling = 1e19 * (gae/(0.511*1e-10))**2 / 24 / 3600
    return coupling * interp(keV, FILE_ABC[:,0], FILE_ABC[:,1])

# Solar 57Fe flux
# Based on fit to energy response
def FluxFe57(keV, gn):
    return 0.6 * 4.56e23 * gn**2 * exp(-(keV - 14.4)**2 / 3) # in cm^-2 s^-1



# Scattering cross-section
def primakoff_scattering_xs_CAF(ea, g, ma, z, r0):
    if ea < ma:
        return 0.0
    prefactor = (g * z)**2 / (2*137)
    eta2 = r0**2 * (ea**2 - ma**2)
    return prefactor * (((2*eta2 + 1)/(4*eta2))*log(1+4*eta2) - 1)

# Cross-section * exposure
def PrimakoffRate(keV, g, ma): # g in GeV^-1
    # primakoff_scattering_xs: MeV^-2
    # prefactor: s m^2 MeV^-2
    return prefactor * primakoff_scattering_xs_CAF(keV/1000, 1e-3 * g, ma, Z_Xe, r0Xe)



def energy_resolution(energy):
    return (energy / 100) * ((32 / sqrt(energy)) + 0.15)

# Background model
def B0(keV):
    return interp(keV, B0_XENON1T[:,0], B0_XENON1T[:,1])



# Signal models
# Use gaussian filter (using sigma=1 to heuristically match up with energy resolution)
def EventsGeneratorPrimakoff(keV, g, ma):
    # FluxPrimakoff: s^-1 cm^-2 keV^-1
    # PriamkoffRate: s m^2
    events = gaussian_filter(m2_to_cm2*np.array([FluxPrimakoff(ea, g)*PrimakoffRate(ea,g,ma)*eff(ea) for ea in keV]), sigma=1, mode="nearest")
    return events + B0(keV)

def EventsGeneratorABC(keV, ge, gg):
    ma = 1e-6  # ma << keV
    events = gaussian_filter(m2_to_cm2*np.array([FluxABC(ea, ge)*PrimakoffRate(ea,gg,ma)*eff(ea) for ea in keV]), sigma=1, mode="nearest")
    return events + B0(keV)

def EventsGeneratorFe57(keV, gn, gg):
    ma = 1e-6  # ma << keV
    events = m2_to_cm2*np.array([FluxFe57(ea, gn)*PrimakoffRate(ea,gg,ma)*eff(ea) for ea in keV])  # energy resolution already included in flux
    return events + B0(keV)



# Make plots
fine_energy_edges = np.linspace(0, 30, 60)
fine_energies = (fine_energy_edges[1:]+fine_energy_edges[:-1])/2

# Benchmark mass models
test_ge = 5e-12
test_gn = 0.5e-6
test_gg = 3.0e-10
ma = 1e-6
signal_primakoff = EventsGeneratorPrimakoff(fine_energies, test_gg, ma)
signal_abc = EventsGeneratorABC(fine_energies, test_ge, test_gg)
signal_fe57 = EventsGeneratorFe57(fine_energies, test_gn, test_gg)


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


