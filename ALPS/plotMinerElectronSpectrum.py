import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline

import sys

from pyCEvNS.axion import MinerAxionElectron, MinerAxionPhoton

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
    det_dis = 2.5
    det_mass = 4
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


def PlotSpectra(flux, mass, coupling):
    # Set exposure
    exposure = meter_by_mev ** 2 * days * det_mass * mev_per_kg / det_am * \
              det_z / (4 * np.pi * det_dis ** 2) * s_per_day
    axion_scale = days * s_per_day / (4 * np.pi * det_dis ** 2)

    # Simulate
    axion_sim = MinerAxionElectron(flux, mass, 100*coupling, 240e3, 90, 15e-24, det_dis, 0)
    axion_sim.simulate()
    electron_scatter_events = axion_sim.scatter_events_binned(det_mass * mev_per_kg / det_am, det_z,
                                                              days*s_per_day, 1e-5)

    axion_sim_primakoff = MinerAxionPhoton(flux, mass, coupling, 240e3, 90, 15e-24, det_dis, 0)
    axion_sim_primakoff.simulate()

    ph_events = axion_sim_primakoff.photon_events_binned(det_area, days*s_per_day, det_thresh)

    sc_events = axion_sim_primakoff.scatter_events_binned(det_mass * mev_per_kg / det_am, det_z, days*s_per_day, 1e-5)
    primakoff_e = axion_sim_primakoff.photon_energy

    print("Scatters: ", axion_sim.scatter_events(det_mass * mev_per_kg / det_am, det_z, days*s_per_day, 1e-5))
    print("e+e- events: ", axion_sim.pair_production_events(det_area, days * s_per_day, 0))

    print("Electrons: ",axion_sim.electron_events_binned(40, det_mass * mev_per_kg / det_am,
                                                         det_z, days*s_per_day, 0))

    axion_e = axion_sim.axion_energy
    axion_w = axion_sim.axion_weight

    electron_e = axion_sim.electron_energy
    electron_w = axion_sim.electron_weight

    photon_e = axion_sim.photon_energy
    photon_w = axion_sim.photon_weight

    axion_bins = np.logspace(-5, np.log10(np.max(axion_e)), flux.shape[0])
    axion_centers = (axion_bins[1:] + axion_bins[:-1])/2

    axion_y, axion_x = np.histogram(axion_e, weights=electron_scatter_events, bins=axion_bins)
    photon_y, photon_x = np.histogram(photon_e, weights=photon_w, bins=axion_bins)
    electron_y, electron_x = np.histogram(electron_e, weights=electron_w, bins=axion_bins)

    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    ax.plot(flux[:, 0], flux[:, 1]*s_per_day, color='k', label="1 MW Reactor Flux")
    ax.plot(primakoff_e, ph_events, color="crimson", label=r"$a \rightarrow \gamma \gamma$")
    ax.plot(primakoff_e, sc_events, color="royalblue", label=r"$a + N \rightarrow \gamma + N$")
    ax.plot(axion_centers, axion_y, color="mediumseagreen", label=r"$a + e^- \rightarrow \gamma + e^-$")

    ax2.plot(flux[:, 0], flux[:, 1]*s_per_day, color='k', label="1 MW Reactor Flux")
    ax2.plot(primakoff_e, ph_events, color="crimson", label=r"$a \rightarrow \gamma \gamma$")
    ax2.plot(primakoff_e, sc_events, color="royalblue", label=r"$a + N \rightarrow \gamma + N$")
    ax2.plot(axion_centers, axion_y, color="mediumseagreen", label=r"$a + e^- \rightarrow \gamma + e^-$")

    ax.set_ylim(1e14*s_per_day, 1e20*s_per_day)  # Reactor flux only
    ax2.set_ylim(1e-3, 1e3)  # Rest of the data
    ax.tick_params(labelbottom=False)  # don't put tick labels at the top
    f.subplots_adjust(hspace=0.1)

    """
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    
    

    
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    """
    #plt.plot(axion_centers, axion_y*axion_scale, label="Axion")
    #plt.plot(axion_e, axion_w)
    #plt.plot(axion_centers, electron_y, label="Electrons")
    #plt.plot(axion_centers, photon_y, label="Photons")


    ax.set_yscale('log')
    ax2.set_yscale('log')
    #plt.xscale('log')
    ax.set_xlim((0.01, 8))
    ax2.set_xlabel("E [MeV]")
    ax.set_ylabel(r"Reactor $\gamma$ Flux [day$^{-1}$]")
    ax2.set_ylabel("4 kg Ge Counts [day$^{-1}$]")
    ax.legend(framealpha=1, fontsize=9, loc="upper right")
    ax.set_title(r"$g_{a\gamma\gamma} = 10^{-3}$ GeV$^{-1}$, $g_{aee} = 10^{-4}$, $m_a =$ 10 keV", loc="right")
    plt.savefig("plots/alps_paper/spectra.png")
    plt.savefig("plots/alps_paper/spectra.pdf")

    plt.clf()


def main(axion_mass, axion_coupling):
    miner_flux = np.genfromtxt('data/reactor_photon.txt')  # flux at reactor surface
    miner_flux[:, 1] *= 1e8 # get flux at the core
    PlotSpectra(miner_flux, axion_mass, axion_coupling)




if __name__ == "__main__":
  main(float(sys.argv[1]), float(sys.argv[2]))