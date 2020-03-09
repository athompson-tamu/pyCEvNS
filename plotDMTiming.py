from pyCEvNS.events import *
from pyCEvNS.flux import *

import matplotlib.pyplot as plt


m_dp = 75
g = 1e-6
m_chi=5


photon_flux = np.genfromtxt("pyCEvNS/data/photon_flux_COHERENT_log_binned.txt")
dm_pim = DMFluxFromPiMinusAbsorption(dark_photon_mass=138, life_time=1, coupling_quark=1, dark_matter_mass=5)
dm_flux_long_heavy = DMFluxIsoPhoton(photon_flux, dark_photon_mass=138, coupling=g, dark_matter_mass=m_chi, life_time=5, sampling_size=5000)


#dm_flux_short = DMFluxIsoPhoton(photon_flux, dark_photon_mass=m_dp, coupling=g, dark_matter_mass=m_chi, life_time=1e-3, sampling_size=1000)
#dm_flux_medium = DMFluxIsoPhoton(photon_flux, dark_photon_mass=m_dp, coupling=g, dark_matter_mass=m_chi, life_time=0.1, sampling_size=1000)
#dm_flux_long = DMFluxIsoPhoton(photon_flux, dark_photon_mass=m_dp, coupling=g, dark_matter_mass=m_chi, life_time=1, sampling_size=1000)


times_pim = dm_pim.dm_timing

#times_short = dm_flux_short.timing
#weights_short = dm_flux_short.weight

#times_medium = dm_flux_medium.timing
#weights_medium = dm_flux_medium.weight

#times_long = dm_flux_long.timing
#weights_long = dm_flux_long.weight

times_long_heavy = dm_flux_long_heavy.dm_timing
weights_long_heavy = dm_flux_long_heavy.weight

plt.hist(times_pim, bins=100, histtype='step', density=True, label=r"$\tau = 1$ $\mu$s, pi-")
#plt.hist(times_long, weights=weights_long, bins=100, histtype='step', density=True, label=r"$\tau = 0.001$ $\mu$s")
#plt.hist(times_medium, weights=weights_medium, bins=100, density=True, histtype='step',  label=r"$\tau = 0.1$ $\mu$s")
#plt.hist(times_short, weights=weights_short, bins=100, density=True, histtype='step',  label=r"$\tau = 1$ $\mu$s")
plt.hist(times_long_heavy, weights=weights_long_heavy, bins=100, histtype='step', density=True, label=r"$\tau = 1$ $\mu$s, $m_{A^\prime} = 138$ MeV")
plt.legend()
plt.show()

plt.savefig("Timing_spectra.png")




