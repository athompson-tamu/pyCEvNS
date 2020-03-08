import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline

from pyCEvNS.axion import MinerAxionElectron


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
  photon_gen = MinerAxionElectron(flux, 1, 1e-6, 240e3, 90, 15e-24, det_dis, 0)
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
      #ev += photon_gen.pair_production_events(det_area, days, 0) * s_per_day
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
  y2_array = np.ones(100)

  rerun = False
  if rerun == True:
    g_array_1, m_array_1 = BinarySearch(-5, 1, 100, miner_flux, dru_limit=dru_limit_hi, detector="ge")
    #g_array_2, m_array_2 = BinarySearch(-5, 1, 20, miner_flux, dru_limit=dru_limit_lo, detector="csi")
    g_array_3, m_array_3 = BinarySearch(-5, 1, 100, miner_flux, dru_limit=dru_limit_lo, detector="csi_2ton")
    miner_flux[:, 1] *= 1e3  # GW scale reactors
    g_array_4, m_array_4 = BinarySearch(-5, 1, 100, miner_flux, dru_limit=dru_limit_lo, detector="connie")
    g_array_5, m_array_5 = BinarySearch(-5, 1, 100, miner_flux, dru_limit=dru_limit_lo, detector="conus")
    g_array_6, m_array_6 = BinarySearch(-5, 1, 100, miner_flux, dru_limit=dru_limit_lo, detector="nucleus")


    np.savetxt("limits/miner_electron/ge.txt", g_array_1)
    np.savetxt("limits/miner_electron/csi_2ton.txt", g_array_3)
    np.savetxt("limits/miner_electron/connie", g_array_4)
    np.savetxt("limits/miner_electron/conus.txt", g_array_5)
    np.savetxt("limits/miner_electron/nucleus.txt", g_array_6)

  else:
    g_array_1 = np.genfromtxt("limits/miner_electron/ge.txt")
    g_array_3 = np.genfromtxt("limits/miner_electron/csi_2ton.txt")
    g_array_4 = np.genfromtxt("limits/miner_electron/connie")
    g_array_5 = np.genfromtxt("limits/miner_electron/conus.txt")
    g_array_6 = np.genfromtxt("limits/miner_electron/nucleus.txt")



  # Draw limits.
  fig, ax = plt.subplots()
  ax.plot(m_array * 1e6, g_array_1, color='crimson', label="MINER Ge (4 kg)")
  #ax.plot(m_array_2 * 1e6, g_array_2, color='crimson', ls="dashed", label="MINER CsI (200 kg)")
  #ax.plot(m_array * 1e6, g_array_3, color='crimson', ls="dashed", label="MINER CsI (2 ton)")
  ax.plot(m_array * 1e6, g_array_6, color='navy', ls='dashed', label=r'NUCLEUS CaWO$_4$(Al$_2$O$_3$) (0.01 kg)')
  ax.plot(m_array * 1e6, g_array_4, color='orange', ls='dashdot', label="CONNIE Si Skipper CCD (0.1 kg)")
  ax.plot(m_array * 1e6, g_array_5, color='teal', ls='dotted', label="CONUS Ge PPC (4 kg)")


  #ax.fill(barbaree[:,0]*1e9, barbaree[:,1]/1e6*(me), color="b", label='BaBar', alpha=0.7)
  #ax.fill(beamee[:, 0] * 1e9, beamee[:, 1] / 1e6 * (me), color="orange", label='Beam Dump', alpha=0.7)
  #upper_edelweiss = np.linspace(1,1,edelweiss3.shape[0])
  #upper_red_giant = np.linspace(1, 1, red_giants.shape[0])
  #ax.fill(np.hstack((elderee[:,0], np.min(elderee[:,0])))*1e9,
   #       np.hstack((elderee[:,1], np.max(elderee[:,1])))/1e6*(me),
    #      label='Edelweiss', color="dimgray", hatch="/", alpha=0.2)
  #ax.fill_between(x=edelweiss3[:, 0] * 1e3, y1=edelweiss3[:, 1], y2=upper_edelweiss,
   #               color="silver", hatch="X", label='Edelweiss III', alpha=0.2)
  #ax.fill_between(x=red_giants[:, 0] * 1e9, y1=red_giants[:, 1]/1e6*me, y2=upper_red_giant,
   #               color="red", hatch="X", label='Red Giants', alpha=0.2)
  ax.legend(loc="upper left", framealpha=1)
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel('$m_a$ [eV]', fontsize=13)
  ax.set_ylabel(r'$g_{aee}$', fontsize=13)
  ax.set_xlim(10, 1e6)
  ax.set_ylim(3e-7,1e-2)
  fig.tight_layout()
  fig.savefig("plots/alps_paper/axion_limits_electron_benchmarks.png")
  fig.savefig("plots/alps_paper/axion_limits_electron_benchmarks.pdf")


  # Save limit arrays
  """
  limit_1dru = np.array([m_array_1*1e6, g_array_1*1e3])
  limit_1dru = limit_1dru.transpose()
  limit_0p1dru = np.array([m_array_2 * 1e6, g_array_2 * 1e3])
  limit_0p1dru = limit_0p1dru.transpose()
  np.savetxt(data_0p1dru_str, limit_0p1dru)
  np.savetxt(data_1dru_str, limit_1dru)
  """







if __name__ == "__main__":
  main()