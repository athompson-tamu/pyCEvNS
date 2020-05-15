import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from matplotlib.pylab import rc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


import sys

from pyCEvNS.axion import IsotropicAxionFromCompton, IsotropicAxionFromPrimakoff

# conversion between units
hbar = 6.58212e-22  # MeV*s
c_light = 2.998e8  # m/s
meter_by_mev = hbar * c_light  # MeV*m
mev_per_kg = 5.6095887e29  # MeV/kg
s_per_day = 3600 * 24
me = 0.511

# axion parameters
axion_mass = 1  # MeV
axion_coupling = 1e-6

detector = "ge"

if detector == "ge":
    # days set to 1, mass set to 1 for DRU units
    det_dis = 2.5
    det_mass = 1
    det_am = 65.13e3
    det_z = 32
    days = 1
    det_area = 0.2 ** 2
    det_thresh = 0
    png_str = "plots/miner_limits/MINER_limits_electron_ge_DRU_bugfix.png"
    pdf_str = "plots/miner_limits/MINER_limits_electron_ge_DRU_bugfix.pdf"
    legend_str_1dru = "MINER Ge (0.1 DRU)"
    legend_str_0p1dru = "MINER Ge (0.01 DRU)"
    data_1dru_str = "limits/electron/miner_ge_0p1dru.txt"
    data_0p1dru_str = "limits/electron/miner_ge_0p01dru.txt"
if detector == "csi":
    det_dis = 4
    det_mass = 200
    det_am = 123.8e3
    det_z = 55
    days = 1
    det_area = 0.2 ** 2
    det_thresh = 2.6
    png_str = "plots/miner_limits/MINER_limits_electron_csi_DRU_bugfix.png"
    pdf_str = "plots/miner_limits/MINER_limits_electron_csi_DRU_bugfix.pdf"
    legend_str_1dru = "MINER CsI (1 DRU)"
    legend_str_0p1dru = "MINER CsI (0.1 DRU)"
    data_1dru_str = "limits/electron/miner_csi_1dru.txt"
    data_0p1dru_str = "limits/electron/miner_csi_0p1dru.txt"


def PlotSpectra(flux, mass, photon_coupling, electron_coupling):
    # Simulate compton like production
    axion_sim_compton = IsotropicAxionFromCompton(flux, mass, electron_coupling,
                                           240e3, 90, 15e-24, det_dis, 0.2)
    axion_sim_compton.simulate()
    electron_scatter_events = axion_sim_compton.scatter_events_binned(det_mass * mev_per_kg / det_am, det_z,
                                                              days*s_per_day, 1e-5)

    # simulate primakoff production
    axion_sim_primakoff = IsotropicAxionFromPrimakoff(flux, mass, photon_coupling,
                                           240e3, 90, 15e-24, det_dis, 0.2)
    axion_sim_primakoff.simulate()

    prim_photon_events = axion_sim_primakoff.photon_events_binned(det_area, days*s_per_day, det_thresh)
    prim_scatter_events = axion_sim_primakoff.scatter_events_binned(det_mass * mev_per_kg / det_am, det_z, days*s_per_day, 1e-5)
    primakoff_photon_energy = axion_sim_primakoff.photon_energy
    
    # CONVERT TO DRU
    prim_photon_events = [a/(1000*b) for a, b in zip(prim_photon_events, primakoff_photon_energy)]
    prim_scatter_events = [a/(1000*b) for a, b in zip(prim_scatter_events, primakoff_photon_energy)]

    # COMPTON Get energies and weights
    axion_comp_e = axion_sim_compton.axion_energy
    axion_comp_w = axion_sim_compton.axion_weight
    
    electron_e = axion_sim_compton.electron_energy
    electron_w = axion_sim_compton.electron_weight

    # PRIMAKOFF Get energies and weights
    axion_prim_e = axion_sim_primakoff.axion_energy
    axion_prim_w = axion_sim_primakoff.axion_weight

    photon_e = axion_sim_primakoff.photon_energy
    photon_w = axion_sim_primakoff.photon_weight

    # Get bins and bin centers
    comp_axion_bins = np.logspace(-5, np.log10(np.max(axion_comp_e)), flux.shape[0])
    comp_axion_centers = (comp_axion_bins[1:] + comp_axion_bins[:-1])/2
    
    prim_axion_bins = np.logspace(-5, np.log10(np.max(axion_prim_e)), flux.shape[0])
    prim_axion_centers = (prim_axion_bins[1:] + prim_axion_bins[:-1])/2
    
    print("N decays = ", axion_sim_primakoff.photon_events(det_area, days, det_thresh) * s_per_day)
    print("N scatters = ", np.sum(prim_scatter_events))

    # Histogram all processes
    comp_axion_bare_y, comp_axion_bare_x = np.histogram(axion_comp_e, weights=electron_scatter_events, bins=comp_axion_bins)
    # CONVERT TO DRU
    comp_axion_bare_y *= 1/(comp_axion_centers*1000)  # divide by energy in keV
    
    prim_axion_bare_y, prim_axion_bare_x = np.histogram(axion_prim_e, weights=axion_prim_w, bins=prim_axion_bins)
    
    plt.plot(flux[:, 0], flux[:, 1]*s_per_day, color='k', label=r"1 MW Reactor $\gamma$ Flux (core)")
    plt.plot(prim_axion_centers, prim_axion_bare_y*s_per_day, color="red", label=r"ALP flux (core)")
    plt.plot(primakoff_photon_energy, prim_photon_events, color="green", label=r"$a \rightarrow \gamma \gamma$ (detector)")
    plt.plot(primakoff_photon_energy, prim_scatter_events, color="royalblue", label=r"$a + A \rightarrow \gamma + A$ (detector)")
    
    plt.yscale('log')
    plt.xlabel(r"Energy [MeV]", fontsize=15)
    plt.ylabel("Counts / day", fontsize=15)
    plt.title(r"$m_a = 1$ MeV, $g_{a\gamma\gamma} = 10^{-5}$ GeV$^{-1}$", loc="right", fontsize=15)
    plt.legend(fontsize=12, loc="upper right")
    plt.xlim((0,8))
    plt.ylim((1e-10, 1e30))
    plt.show()
    plt.clf()


    # Begin paper plot
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    ax.plot(flux[:, 0], flux[:, 1]*s_per_day, color='k', label="1 MW Reactor Flux")
    ax.plot(primakoff_photon_energy, prim_photon_events, color="crimson", label=r"$a \rightarrow \gamma \gamma$")
    ax.plot(primakoff_photon_energy, prim_scatter_events, color="royalblue", label=r"$a + N \rightarrow \gamma + N$")
    ax.plot(comp_axion_centers, comp_axion_bare_y, color="mediumseagreen", label=r"$a + e^- \rightarrow \gamma + e^-$")

    ax2.plot(flux[:, 0], flux[:, 1]*s_per_day, color='k', label="1 MW Reactor Flux")
    ax2.plot(primakoff_photon_energy, prim_photon_events, color="crimson", label=r"$a \rightarrow \gamma \gamma$")
    ax2.plot(primakoff_photon_energy, prim_scatter_events, color="royalblue", label=r"$a + N \rightarrow \gamma + N$")
    ax2.plot(comp_axion_centers, comp_axion_bare_y, color="mediumseagreen", label=r"$a + e^- \rightarrow \gamma + e^-$")

    ax.set_ylim(1e14*s_per_day, 1e20*s_per_day)  # Reactor flux only
    ax2.set_ylim(7e-7, 1e2)  # Rest of the data
    ax.tick_params(labelbottom=False)  # don't put tick labels at the top
    f.subplots_adjust(hspace=0.1)

    ax.set_yscale('log')
    ax2.set_yscale('log')
    ax.set_xlim((0.01, 6))
    ax2.set_xlabel("E [MeV]")
    ax.set_ylabel(r"Reactor $\gamma$ Flux [day$^{-1}$]")
    ax2.set_ylabel("DRU")
    ax.legend(framealpha=1, fontsize=9, loc="upper right")
    ax.set_title(r"$g_{a\gamma\gamma} = 10^{-3}$ GeV$^{-1}$, $g_{aee} = 10^{-4}$, $m_a =$ 10 keV", loc="right")
    plt.savefig("plots/alps_paper/spectra.png")
    plt.savefig("plots/alps_paper/spectra.pdf")
    plt.show()

    plt.clf()


def main(axion_mass, photon_coupling, electron_coupling):
    miner_flux = np.genfromtxt('data/reactor_photon.txt')  # flux at reactor surface
    miner_flux[:, 1] *= 1e8 # get flux at the core
    #miner_flux = np.array([[2.2,1e22]])
    PlotSpectra(miner_flux, axion_mass, photon_coupling, electron_coupling)




if __name__ == "__main__":
  main(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))