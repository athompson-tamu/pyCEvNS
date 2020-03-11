from pyCEvNS.events import *
from pyCEvNS.flux import *

import matplotlib.pyplot as plt
from matplotlib.pylab import rc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


m_dp = 75
g = 1e-6
m_chi=1
pim_rate_coherent = 0.0457
pim_rate_jsns = 0.4962
pim_rate_ccm = 0.0259
pim_rate = pim_rate_coherent

# Get the DM fluxes.
photon_flux = np.genfromtxt("data/coherent/brem.txt")
Pi0Info = np.genfromtxt("data/coherent/Pi0_Info.txt")
pion_energy = Pi0Info[:,4] - massofpi0
pion_azimuth = np.arccos(Pi0Info[:,3] / np.sqrt(Pi0Info[:,1]**2 + Pi0Info[:,2]**2 + Pi0Info[:,3]**2))  # arccos of the unit z vector gives the azimuth angle
pion_cos = np.cos(np.pi/180 * Pi0Info[:,0])
pion_flux = np.array([pion_azimuth, pion_cos, pion_energy])
pion_flux = pion_flux.transpose()

light_mass = 75
heavy_mass = 138
nsamples = 5000

print("running short...")
dm_pi0_short = DMFluxFromPi0Decay(pi0_distribution=pion_flux, dark_photon_mass=light_mass, life_time=0.001, coupling_quark=g, dark_matter_mass=m_chi)
dm_brem_short = DMFluxIsoPhoton(photon_flux, dark_photon_mass=light_mass, coupling=g, dark_matter_mass=m_chi, life_time=0.001, sampling_size=nsamples)
dm_pim_short = DMFluxFromPiMinusAbsorption(dark_photon_mass=light_mass, life_time=0.001, coupling_quark=g, dark_matter_mass=m_chi,
                                           pion_rate=pim_rate)

print("running long...")
dm_pi0_long = DMFluxFromPi0Decay(pi0_distribution=pion_flux, dark_photon_mass=light_mass, life_time=1, coupling_quark=g, dark_matter_mass=m_chi)
dm_brem_long = DMFluxIsoPhoton(photon_flux, dark_photon_mass=light_mass, coupling=g, dark_matter_mass=m_chi, life_time=1, sampling_size=nsamples)
dm_pim_long = DMFluxFromPiMinusAbsorption(dark_photon_mass=light_mass, life_time=1, coupling_quark=g, dark_matter_mass=m_chi,
                                          pion_rate=pim_rate)

print("running HEAVY... UwU")
dm_brem_heavy = DMFluxIsoPhoton(photon_flux, dark_photon_mass=heavy_mass, coupling=g, dark_matter_mass=m_chi, life_time=1, sampling_size=nsamples)
dm_pim_heavy = DMFluxFromPiMinusAbsorption(dark_photon_mass=heavy_mass, life_time=1, coupling_quark=g, dark_matter_mass=m_chi,
                                           pion_rate=pim_rate)



# Weights
pim_short_norm = dm_pim_short.norm*np.ones(dm_pim_short.timing.shape[0]) / dm_pim_short.timing.shape[0]
pim_long_norm = dm_pim_long.norm*np.ones(dm_pim_long.timing.shape[0]) / dm_pim_long.timing.shape[0]
pim_heavy_norm = dm_pim_heavy.norm*np.ones(dm_pim_heavy.timing.shape[0]) / dm_pim_heavy.timing.shape[0]

pi0_short_norm = dm_pi0_short.norm*np.ones(dm_pi0_short.timing.shape[0]) / dm_pi0_short.timing.shape[0]
pi0_long_norm = dm_pi0_long.norm*np.ones(dm_pi0_long.timing.shape[0]) / dm_pi0_long.timing.shape[0]

print(np.sum(pim_long_norm),dm_brem_long.norm, np.sum(pi0_long_norm))
print(pim_long_norm.shape,len(dm_brem_long.weight), pi0_long_norm.shape)


# Add in the pi- events.
times_short = np.append(np.append(dm_brem_short.timing, dm_pim_short.timing), dm_pi0_short.timing)
weights_short = np.append(np.append(dm_brem_short.weight, pim_short_norm), pi0_short_norm)

times_long = np.append(np.append(dm_brem_long.timing, dm_pim_long.timing), dm_pi0_long.timing)
weights_long = np.append(np.append(dm_brem_long.weight, pim_long_norm), pi0_long_norm)

times_heavy = np.append(dm_brem_heavy.timing, dm_pim_heavy.timing)
weights_heavy = np.append(dm_brem_heavy.weight, pim_heavy_norm)


# Plot timing spectra.
time_bins = np.linspace(0,3,90)
density = True
#plt.hist(dm_brem_long.timing, weights=dm_brem_long.weight, bins=time_bins, histtype='step', density=density, color='k')
#plt.hist(dm_pim_long.timing, weights=pim_long_norm, bins=time_bins, ls='--', histtype='step', density=density, color='k')
#plt.hist(dm_pi0_long.timing, weights=pi0_long_norm, bins=time_bins, ls='dotted', histtype='step', density=density, color='k')
plt.hist(times_short, weights=weights_short, bins=time_bins,
 histtype='step', density=density, color='red', label=r"$M_{A^\prime} = 75$ MeV, $\tau \leq 0.001$ $\mu$s")
plt.hist(times_long, weights=weights_long, bins=time_bins,
 density=density, histtype='step', color='green', label=r"$M_{A^\prime} = 75$ MeV, $\tau = 1$ $\mu$s")
plt.hist(times_heavy, weights=weights_heavy, bins=time_bins,
 density=density, histtype='step', color='blue', label=r"$M_{A^\prime} = 138$ MeV, $\tau = 1$ $\mu$s")
plt.xlim((0,3))
plt.legend()
plt.xlabel(r"Arrival Time [$\mu$s]")
plt.ylabel("a.u.")


plt.savefig("paper_plots/timing_coherent.png")
plt.show()





