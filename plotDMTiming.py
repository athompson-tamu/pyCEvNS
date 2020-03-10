from pyCEvNS.events import *
from pyCEvNS.flux import *

import matplotlib.pyplot as plt
from matplotlib.pylab import rc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


m_dp = 75
g = 1e-6
m_chi=5

pim_rate_coherent = 0.0457

# Get the DM fluxes.
photon_flux = np.genfromtxt("pyCEvNS/data/photon_flux_COHERENT_log_binned.txt")
dm_flux_short = DMFluxIsoPhoton(photon_flux, dark_photon_mass=75, coupling=g, dark_matter_mass=m_chi, life_time=0.001, sampling_size=5000)
dm_pim_short = DMFluxFromPiMinusAbsorption(dark_photon_mass=75, life_time=0.001, coupling_quark=g, dark_matter_mass=m_chi,
                                           pion_rate=pim_rate_coherent)

dm_flux_long = DMFluxIsoPhoton(photon_flux, dark_photon_mass=75, coupling=g, dark_matter_mass=m_chi, life_time=1, sampling_size=5000)
dm_pim_long = DMFluxFromPiMinusAbsorption(dark_photon_mass=75, life_time=1, coupling_quark=g, dark_matter_mass=m_chi,
                                          pion_rate=pim_rate_coherent)

dm_flux_heavy = DMFluxIsoPhoton(photon_flux, dark_photon_mass=138, coupling=g, dark_matter_mass=m_chi, life_time=1, sampling_size=5000)
dm_pim_heavy = DMFluxFromPiMinusAbsorption(dark_photon_mass=138, life_time=1, coupling_quark=g, dark_matter_mass=m_chi,
                                           pion_rate=pim_rate_coherent)


times_short = dm_flux_short.timing
weights_short = dm_flux_short.weight

times_long = dm_flux_long.timing
weights_long = dm_flux_long.weight

times_heavy = dm_flux_heavy.timing
weights_heavy = dm_flux_heavy.weight

times_pim_short = dm_pim_short.timing
times_pim_long = dm_pim_long.timing
times_pim_heavy = dm_pim_heavy.timing

# Add in the pi- events.
times_short = np.append(times_short, times_pim_short)
weights_short = np.append(weights_short, dm_pim_short.norm*np.ones(times_pim_short.shape[0]))

times_long = np.append(times_long, times_pim_long)
weights_long = np.append(weights_long, dm_pim_long.norm*np.ones(times_pim_long.shape[0]))

times_heavy = np.append(times_heavy, times_pim_heavy)
weights_heavy = np.append(weights_heavy, dm_pim_heavy.norm*np.ones(times_pim_heavy.shape[0]))


# Plot timing spectra.
time_bins = np.linspace(0,3,90)
density = True
plt.hist(times_pim_short, weights=dm_pim_short.norm*np.ones(times_pim_short.shape[0]),
 bins=time_bins, color='red', ls='--', histtype='step', density=density, label=r"$\tau = 1$ $\mu$s, pi- short")
plt.hist(times_pim_long, weights=dm_pim_long.norm*np.ones(times_pim_long.shape[0]),
 bins=time_bins, color='green', ls='--',histtype='step', density=density, label=r"$\tau = 1$ $\mu$s, pi- long")
plt.hist(times_pim_heavy, weights=dm_pim_heavy.norm*np.ones(times_pim_heavy.shape[0]),
 bins=time_bins, color='blue', ls='--',histtype='step', density=density, label=r"$\tau = 1$ $\mu$s, pi- heavy")
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


plt.savefig("paper_plots/timing.png")
plt.show()





