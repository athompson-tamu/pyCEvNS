from pyCEvNS.events import *
from pyCEvNS.flux import *

import matplotlib.pyplot as plt
from matplotlib.pylab import rc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


m_dp = 75
g = 1e-6
m_chi=1


photon_flux = np.genfromtxt("pyCEvNS/data/photon_flux_COHERENT_log_binned.txt")
#dm_pim = DMFluxFromPiMinusAbsorption(dark_photon_mass=138, life_time=1, coupling_quark=g, dark_matter_mass=m_chi)
dm_flux_short = DMFluxIsoPhoton(photon_flux, dark_photon_mass=75, coupling=g, dark_matter_mass=m_chi, life_time=0.001, sampling_size=50000)
dm_flux_long = DMFluxIsoPhoton(photon_flux, dark_photon_mass=75, coupling=g, dark_matter_mass=m_chi, life_time=2, sampling_size=50000)
dm_flux_heavy = DMFluxIsoPhoton(photon_flux, dark_photon_mass=469, coupling=g, dark_matter_mass=m_chi, life_time=2, sampling_size=50000)




times_short = dm_flux_short.timing
weights_short = dm_flux_short.weight

times_long = dm_flux_long.timing
weights_long = dm_flux_long.weight

times_heavy = dm_flux_heavy.timing
weights_heavy = dm_flux_heavy.weight

#times_pim = dm_pim.timing

time_bins = np.linspace(0,3,90)

#plt.hist(times_pim, bins=100, histtype='step', density=True, label=r"$\tau = 1$ $\mu$s, pi-")
plt.hist(times_short, weights=weights_short, bins=time_bins,
 histtype='step', density=True, color='red', label=r"$M_{A^\prime} = 75$ MeV, $\tau \leq 0.001$ $\mu$s")
plt.hist(times_long, weights=weights_long, bins=time_bins,
 density=True, histtype='step', color='green', label=r"$M_{A^\prime} = 75$ MeV, $\tau = 2$ $\mu$s")
plt.hist(times_heavy, weights=weights_heavy, bins=time_bins,
 density=True, histtype='step', color='blue', label=r"$M_{A^\prime} = 470$ MeV, $\tau = 2$ $\mu$s")
plt.xlim((0,3))
plt.legend()
plt.xlabel(r"Arrival Time [$\mu$s]")
plt.ylabel("a.u.")


plt.savefig("paper_plots/timing.png")
plt.show()





