import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline

from pyCEvNS.axion import IsotropicAxionFromCompton
from matplotlib.pylab import rc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Set global variables
miner = np.genfromtxt('data/reactor_photon.txt')
coherent = np.genfromtxt('data/photon_flux_COHERENT_log_binned.txt')
beamee = np.genfromtxt('data/beam_ee.txt', delimiter=',')
barbaree = np.genfromtxt('data/babar_ee.txt', delimiter=',')
elderee = np.genfromtxt('data/edelweis.txt', delimiter=',')
edelweiss3 = np.genfromtxt('data/edelweiss3.txt', delimiter=",")
red_giants = np.genfromtxt('data/redgiant.txt', delimiter=",")


detector = "ge"

if detector == "ge":
  det_dis = 2.5
  det_mass = 4
  det_mass_hi = 4
  det_am = 65.13e3
  det_z = 32
  days = 1
  det_area = 0.2 ** 2
  det_thresh = 2.6
  dru_limit_lo = 0.1
  dru_limit_hi = 0.1
  png_str = "plots/miner_limits/MINER_limits_electron_ge_DRU.png"
  pdf_str = "plots/miner_limits/MINER_limits_electron_ge_DRU.pdf"
  legend_str_1dru = "MINER Ge (4kg, 0.1 DRU)"
  legend_str_0p1dru = "MINER Ge (4kg, 0.1 DRU)"
  data_1dru_str = "limits/electron/miner_ge_0p1dru.txt"
  data_0p1dru_str = "limits/electron/miner_ge_0p01dru.txt"
if detector == "csi":
  det_dis = 2.5
  det_mass = 200
  det_mass_hi = 2000
  det_am = 123.8e3
  det_z = 55
  days = 1
  det_area = 0.4 ** 2
  det_thresh = 2.6
  dru_limit_lo = 0.01  # TODO: just get background model from Rupak
  dru_limit_hi = 0.01
  png_str = "plots/miner_limits/MINER_limits_electron_csi_DRU_200kg_2ton.png"
  pdf_str = "plots/miner_limits/MINER_limits_electron_csi_DRU_200kg_2ton.pdf"
  legend_str_1dru = "MINER CsI (200 kg, 0.01 DRU)"
  legend_str_0p1dru = "MINER CsI (2 ton, 0.01 DRU)"
  data_1dru_str = "limits/electron/miner_csi_1dru.txt"
  data_0p1dru_str = "limits/electron/miner_csi_0p1dru.txt"
if detector == "connie":
  det_dis = 30
  det_mass = 0.1
  det_mass_hi = 0.1
  det_am = 65.13e3 / 2
  det_z = 14
  days = 1000
  det_area = 0.4 ** 2
  det_thresh = 2.6
  dru_limit_lo = 0.01  # TODO: just get background model from Rupak
  dru_limit_hi = 0.01
  png_str = "plots/miner_limits/MINER_limits_electron_csi_DRU_200kg_2ton.png"
  pdf_str = "plots/miner_limits/MINER_limits_electron_csi_DRU_200kg_2ton.pdf"
  legend_str_1dru = "MINER CsI (200 kg, 0.01 DRU)"
  legend_str_0p1dru = "MINER CsI (2 ton, 0.01 DRU)"
  data_1dru_str = "limits/electron/miner_csi_1dru.txt"
  data_0p1dru_str = "limits/electron/miner_csi_0p1dru.txt"


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





def BinarySearch(m_a, m_b, nbins, flux, dru_limit=None, ev_limit=None, detector="ge"):
  det_dis = 2.25
  det_mass = 4
  det_am = 65.13e3
  det_z = 32
  days = 1000
  det_area = 0.2 ** 2
  det_thresh = 1e-3
  dru_limit = 0.1
  bkg_dru = 100
  if detector == "csi":
    det_dis = 2.5
    det_mass = 200
    det_am = 123.8e3
    det_z = 55
    days = 1000
    det_area = 0.4 ** 2
    det_thresh = 1e-3
    dru_limit = 0.01
    bkg_dru = 100
  if detector == "csi_2ton":
    det_dis = 2.5
    det_mass = 2000
    det_am = 123.8e3
    det_z = 55
    days = 1000
    det_area = 0.4 ** 2
    det_thresh = 1e-3
    dru_limit = 0.01
    bkg_dru = 100
  if detector == "connie":
    det_dis = 30
    det_mass = 0.1
    det_am = 65.13e3 / 2
    det_z = 14
    days = 1000
    det_area = 0.4 ** 2
    det_thresh = 0.028e-3
    dru_limit = 0.01
    bkg_dru = 700
  if detector == "conus":
    det_dis = 17
    det_mass = 4
    det_am = 65.13e3
    det_z = 32
    days = 1000
    det_area = 0.4 ** 2
    det_thresh = 1e-3
    dru_limit = 0.01
    bkg_dru = 100
  if detector == "nucleus":
    det_dis = 40
    det_mass = 0.01
    det_am = 65.13e3 * 3
    det_z = 51
    days = 1000
    det_area = 0.4 ** 2
    det_thresh = 0.02e-3
    dru_limit = 0.01
    bkg_dru = 100


  event_limit = dru_limit * days * det_mass
  sig_limit = 2
  bkg = bkg_dru * days * det_mass
  print("Event threshold: ", event_limit)
  malist_miner_estim_scatter = np.logspace(m_a, m_b, nbins)
  galist_miner_estim_scatter = np.zeros_like(malist_miner_estim_scatter)
  photon_gen = IsotropicAxionFromCompton(flux, 1, 1e-6, 240e3, 90, 15e-24, det_dis, 0)
  for i in range(malist_miner_estim_scatter.shape[0]):
    print("Simulating mass point ", i)
    lo = -8
    hi = 2
    photon_gen.axion_mass = malist_miner_estim_scatter[i]
    while hi - lo > 0.01:
      mid = (hi + lo) / 2
      photon_gen.axion_coupling = 10 ** mid
      photon_gen.simulate()
      ev = photon_gen.scatter_events(det_mass * mev_per_kg / det_am, det_z, days, det_thresh) * s_per_day
      ev += photon_gen.pair_production_events(det_area, days, 0) * s_per_day
      sig = ev / np.sqrt(ev + bkg)  # signal only model
      if sig < sig_limit:
        lo = mid
      else:
        hi = mid
    galist_miner_estim_scatter[i] = 10 ** mid
  return galist_miner_estim_scatter, malist_miner_estim_scatter





def main():
  miner_flux = miner  # flux at reactor surface
  miner_flux[:, 1] *= 1e8  # get flux at the core

  # Event limits

  m_array = np.logspace(-5, 1, 100)
  
  rerun = False
  if rerun == True:
    g_array_ge, m_array_ge = BinarySearch(-5, 1, 100, miner_flux, dru_limit=dru_limit_hi, detector="ge")
    miner_flux[:, 1] *= 1e3  # GW scale reactors
    g_array_connie, m_array_connie = BinarySearch(-5, 1, 50, miner_flux, dru_limit=dru_limit_lo, detector="connie")
    g_array_conus, m_array_conus = BinarySearch(-5, 1, 50, miner_flux, dru_limit=dru_limit_lo, detector="conus")
    g_array_nucleus, m_array_nucleus = BinarySearch(-5, 1, 50, miner_flux, dru_limit=dru_limit_lo, detector="nucleus")


    np.savetxt("limits/miner_electron/ge.txt", g_array_ge)
    np.savetxt("limits/miner_electron/connie", g_array_connie)
    np.savetxt("limits/miner_electron/conus.txt", g_array_conus)
    np.savetxt("limits/miner_electron/nucleus.txt", g_array_nucleus)

  else:
    g_array_ge = np.genfromtxt("limits/miner_electron/ge.txt")
    g_array_connie = np.genfromtxt("limits/miner_electron/connie")
    g_array_conus = np.genfromtxt("limits/miner_electron/conus.txt")
    g_array_nucleus = np.genfromtxt("limits/miner_electron/nucleus.txt")



  # Draw existing limits
  astro_alpha = 0.1
  #plt.fill(barbaree[:,0]*1e9, barbaree[:,1]/1e6*(me), color="b", label='BaBar', alpha=0.7)
  plt.fill(beamee[:, 0] * 1e9, beamee[:, 1] / 1e6 * (me), color="purple", alpha=0.7)
  upper_edelweiss = np.linspace(1,1,edelweiss3.shape[0])
  upper_red_giant = np.linspace(1, 1, red_giants.shape[0])
  #plt.fill(np.hstack((elderee[:,0], np.min(elderee[:,0])))*1e9,
  #        np.hstack((elderee[:,1], np.max(elderee[:,1])))/1e6*(me),
  #        label='Edelweiss', color="dimgray", hatch="/", alpha=0.2)
  plt.fill_between(x=edelweiss3[:, 0] * 1e3, y1=edelweiss3[:, 1], y2=upper_edelweiss,
                  color="goldenrod", alpha=0.05)
  plt.fill_between(x=red_giants[:, 0] * 1e9, y1=red_giants[:, 1]/1e6*me, y2=upper_red_giant,
                  color="lightcoral", alpha=astro_alpha)
  
  
  # Draw limits.
  plt.plot(m_array * 1e6, g_array_ge, color='crimson', label="MINER Ge (4 kg)")
  plt.plot(m_array * 1e6, g_array_nucleus, color='navy', ls='dashed', label=r'NUCLEUS CaWO$_4$(Al$_2$O$_3$) (0.01 kg)')
  plt.plot(m_array * 1e6, g_array_connie, color='orange', ls='dashdot', label="CONNIE Si Skipper CCD (0.1 kg)")
  plt.plot(m_array * 1e6, g_array_conus, color='teal', ls='dotted', label="CONUS Ge PPC (4 kg)")
  
  # Draw existing limits text
  text_fs = 14
  plt.text(12,5e-7,'Red Giants', rotation=0, fontsize=text_fs, color="k", weight="bold")
  plt.text(2.3e6,5e-7,'Beam\n Dump', rotation=0, fontsize=text_fs, color="white", weight="bold")
  plt.text(9000,5e-7,'Edelweiss III', rotation=0, fontsize=text_fs, color="k", weight="bold")
  plt.legend(loc="upper left", framealpha=1, fontsize=12)
  plt.xscale('log')
  plt.yscale('log')
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.xlabel('$m_a$ [eV]', fontsize=24)
  plt.ylabel(r'$g_{aee}$', fontsize=24)
  plt.xlim(10, 1e7)
  plt.ylim(1e-7,1e-1)
  #fig.savefig("plots/alps_paper/axion_limits_electron_benchmarks_x2bkg.png")
  #fig.savefig("plots/alps_paper/axion_limits_electron_benchmarks_x2bkg.pdf")
  plt.tight_layout()
  plt.show()


  # Save limit arrays
  """
  limit_1dru = np.array([m_array_1*1e6, g_array_ge*1e3])
  limit_1dru = limit_1dru.transpose()
  limit_0p1dru = np.array([m_array_2 * 1e6, g_array_connie * 1e3])
  limit_0p1dru = limit_0p1dru.transpose()
  np.savetxt(data_0p1dru_str, limit_0p1dru)
  np.savetxt(data_1dru_str, limit_1dru)
  """







if __name__ == "__main__":
  main()