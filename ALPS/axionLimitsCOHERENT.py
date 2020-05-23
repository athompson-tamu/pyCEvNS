import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline

from pyCEvNS.detectors import Detector

from pyCEvNS.axion import Axion


# Read in data.
coherent = np.genfromtxt('data/photon_flux_COHERENT_log_binned.txt')
ccm_flux = np.genfromtxt('data/ccm_800mev_photon_spectra_1e5_POT.txt')
miner = np.genfromtxt('data/reactor_photon.txt')
beam = np.genfromtxt('data/beam.txt')
eeinva = np.genfromtxt('data/eeinva.txt')
lep = np.genfromtxt('data/lep.txt')
lsw = np.genfromtxt('data/lsw.txt')
nomad = np.genfromtxt('data/nomad.txt')
pion = [[129, 5e20/24/3600*18324/500000]]

# Declare constants.
detector = "Ar"

# Set the flux.
pot_per_day = 5e20
pot_sample = 1e5
s_per_day = 24 * 3600
flux = coherent # flux at reactor surface
flux[:, 1] *= pot_per_day / pot_sample / s_per_day # photons / sec

# CsI default
det_dis = 19.3
min_decay_length = 10
det_mass = 14.6
det_am = 123.8e3
det_z = 55
days = 2*365
det_density = 4510  # kg/m^3
upper_limit_dir = "limits/csi_14p6/upper_limit.txt"
removed_limit_dir = "limits/csi_14p6/removed_limit.txt"
scatter_limit_dir = "limits/csi_14p6/scatter_limit.txt"
trimmed_limit_dir = "limits/csi_14p6/scatter_limit_trimmed.txt"
# LAr
if detector == "Ar":
  min_decay_length = 14.7
  det_dis = 28.3
  det_mass = 610
  det_am = 37.211e3
  det_z = 18
  days = 3 * 365
  det_density = 1396 # kg/m^3
  upper_limit_dir = "limits/lar_610/upper_limit.txt"
  removed_limit_dir = "limits/lar_610/removed_limit.txt"
  scatter_limit_dir = "limits/lar_610/scatter_limit.txt"
  trimmed_limit_dir = "limits/lar_610/scatter_limit_trimmed.txt"

if detector == "ccm":
  flux = ccm_flux
  flux[:, 1] *= pot_per_day / pot_sample / s_per_day
  min_decay_length = 1
  det_dis = 20
  det_mass = 7000 #610
  det_am = 37.211e3
  det_z = 18
  days = 3 * 365
  det_density = 1396 # kg/m^3
  upper_limit_dir = "limits/ccm/upper_limit.txt"
  removed_limit_dir = "limits/ccm/removed_limit.txt"
  scatter_limit_dir = "limits/ccm/scatter_limit.txt"
  trimmed_limit_dir = "limits/ccm/scatter_limit_trimmed.txt"

# conversion between units
hbar = 6.58212e-22  # MeV*s
c_light = 2.998e8  # m/s
meter_by_mev = hbar * c_light  # MeV*m
mev_per_kg = 5.6095887e29  # MeV/kg
s_per_day = 24 * 3600

# axion parameters
axion_mass = 1e-6 # MeV
axion_coupling = 1e-6

mercury_mass = 186850
mercury_a = 202
mercury_z = 80
mercury_density = 13534*mev_per_kg*meter_by_mev**3/mercury_mass
mercury_n_gamma = 3e-23
det_r = (det_mass/(det_density)*3/(4*np.pi))**(1/3)






# Loop over mass and coupling arrays performing binary search
#axion_sim = Axion([[129, 5e20/24/3600*18324/500000]], 1, 1e-6, mercury_mass, mercury_z, mercury_n_gamma, 20, 10)
axion_sim = Axion(flux, axion_mass, axion_coupling, mercury_mass,
                  mercury_z, mercury_n_gamma, det_dis, min_decay_length)
nsamples = 1000

# SCATTER LIMIT
print("scatter limit")
scatter_a = -8
scatter_b = 0
mass_array_scatter = np.logspace(scatter_a, scatter_b, 15)  # was -3, 0, 25
coup_array_scatter = np.zeros_like(mass_array_scatter)
# days = 308
for i in range(mass_array_scatter.shape[0]):
    lo = -10
    hi = -3
    axion_sim.axion_mass = mass_array_scatter[i]
    print(i, end='...')
    while hi - lo > 0.01: # stopping criterion
        mid = (hi+lo)/2
        axion_sim.axion_coupling = 10**mid
        axion_sim.simulate(nsamples)
        ev = axion_sim.scatter_events(det_mass*mev_per_kg/det_am, det_z, days*s_per_day, 0)
        if np.abs(ev-10) < 1:
            break
        if ev < 10:
            lo = mid
        else:
            hi = mid
    coup_array_scatter[i] = 10**mid


# Save raw arrays.
scatter_arrays = np.array([mass_array_scatter, coup_array_scatter])
scatter_arrays = scatter_arrays.transpose()
np.savetxt(scatter_limit_dir, scatter_arrays)



# Delete jump discontinuties
idx = []
for i in range(0, coup_array_scatter.shape[0]-1):
    if coup_array_scatter[i] > 1e-6:
        idx.append(i)
coup_array_scatter = np.delete(coup_array_scatter, idx, axis=0)
mass_array_scatter = np.delete(mass_array_scatter, idx, axis=0)

scatter_arrays = np.array([mass_array_scatter, coup_array_scatter])
scatter_arrays = scatter_arrays.transpose()
# Save arrays to draw later.
np.savetxt(trimmed_limit_dir, scatter_arrays)


# HIGH LIMIT
high_a = -1
high_b = np.log10(500)
mass_array_upper = np.logspace(high_a, high_b, 12) # masses 1 MeV to 120 MeV
coup_array_lower = np.zeros_like(mass_array_upper)
coup_array_upper = np.zeros_like(mass_array_upper)
print("hi limit")
for i in range(mass_array_upper.shape[0]):
    print(i)
    tmplist = np.logspace(-10, -4, 120)  # temp coupling array
    evlist = np.zeros_like(tmplist)
    axion_sim.axion_mass = mass_array_upper[i]
    print(i, end='...')
    for j in range(tmplist.shape[0]):
        axion_sim.axion_coupling = tmplist[j]
        axion_sim.simulate(nsamples)
        evlist[j] = axion_sim.photon_events(4*np.pi*det_r**2, days*s_per_day, 0)
    for j in range(1, tmplist.shape[0]):
        if evlist[j-1] < 10 and evlist[j] >= 10:
            coup_array_lower[i] = tmplist[j]
        if evlist[j-1] >= 10 and evlist[j] < 10:
            coup_array_upper[i] = tmplist[j]

upper_arrays = np.array([mass_array_upper, coup_array_upper])
upper_arrays = upper_arrays.transpose()
np.savetxt(upper_limit_dir, upper_arrays)


# Look for jump disc. and delete them in the low limit

idx = []
for i in range(1, coup_array_lower.shape[0]-1):
    if coup_array_lower[i]/10 > coup_array_lower[i-1]:
        idx.append(i)
galist_csi_lo_r = np.delete(coup_array_lower, idx, axis=0)
malist_r = np.delete(mass_array_upper, idx, axis=0)

removed_arrays = np.array([malist_r, galist_csi_lo_r])
removed_arrays = removed_arrays.transpose()
np.savetxt(removed_limit_dir, removed_arrays)
