import sys
import json
import numpy as np
from numpy import log, exp, pi, zeros, genfromtxt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pymultinest
import mpi4py

from MNStats import MNStats

from pyCEvNS.events import*
from pyCEvNS.oscillation import*

"""
import pynverse as pynv
import rpy2
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, ListVector
from rpy2.robjects import r
base = importr('base')
utils = importr('utils')
rcopula = importr('copula')
"""


def FrankCopula(u1, v2, theta):
  u2 = - (1/theta) * np.log(1 + (v2 * (1 - np.exp(-theta))) / (v2 * (np.exp(-theta * u1) - 1) -
                                                                        np.exp(-theta * u1)))
  return u2



"""
# Simulate bivariate pairs empirically.
class EmpricalCopula:
  def __init__(self, datastr, i, j):
    # read in table
    robjects.r('data = read.table(file = "{0}", header=F)'.format(datastr))
    robjects.r('z = pobs(as.matrix(cbind(data[,{0}],data[,{1}])))'.format(i, j))
  def simulate(self, u, v):
    def ddv(v2):
      v2 = float(v2)
      robjects.r('u = matrix(c({0}, {1}), 1, 2)'.format(u, v2))
      return np.asarray(robjects.r('dCn(u, U = z, j.ind = 1)'))
    try:
      return float(pynv.inversefunc(ddv, y_values=v, domain=[0, 1], open_domain=[True, True],
                                    image=[0,1], accuracy=1))
    except:
      #print("passing...")
      return v


emp_ee_mm = EmpricalCopula("python/multinest_posteriors/all_nsi_vector_solar_05_equal_weights.txt", 1, 2)
emp_ee_tt = EmpricalCopula("python/multinest_posteriors/all_nsi_vector_solar_05_equal_weights.txt", 1, 3)
emp_ee_em = EmpricalCopula("python/multinest_posteriors/all_nsi_vector_solar_05_equal_weights.txt", 1, 4)
emp_ee_et = EmpricalCopula("python/multinest_posteriors/all_nsi_vector_solar_05_equal_weights.txt", 1, 5)
emp_ee_mt = EmpricalCopula("python/multinest_posteriors/all_nsi_vector_solar_05_equal_weights.txt", 1, 6)


emp_ud_ee = EmpricalCopula("nsi_multinest/all_nsi_xenon_atmos_coh_prior/all_nsi_xenon_atmos_coh_priorpost_equal_weights.dat", 1, 7)
emp_ud_mm = EmpricalCopula("nsi_multinest/all_nsi_xenon_atmos_coh_prior/all_nsi_xenon_atmos_coh_priorpost_equal_weights.dat", 2, 8)
emp_ud_tt = EmpricalCopula("nsi_multinest/all_nsi_xenon_atmos_coh_prior/all_nsi_xenon_atmos_coh_priorpost_equal_weights.dat", 3, 9)
emp_ud_em = EmpricalCopula("nsi_multinest/all_nsi_xenon_atmos_coh_prior/all_nsi_xenon_atmos_coh_priorpost_equal_weights.dat", 4, 10)
emp_ud_et = EmpricalCopula("nsi_multinest/all_nsi_xenon_atmos_coh_prior/all_nsi_xenon_atmos_coh_priorpost_equal_weights.dat", 5, 11)
emp_ud_mt = EmpricalCopula("nsi_multinest/all_nsi_xenon_atmos_coh_prior/all_nsi_xenon_atmos_coh_priorpost_equal_weights.dat", 6, 12)
"""

# Set globals.
n_z = 20
max_z = -0.025

# Read in the posteriors from Xenon experiments.
xen_ud_posterior = np.genfromtxt("nsi_multinest/all_nsi_xenon_atmos_coh_prior/all_nsi_xenon_atmos_coh_prior.txt")
xen_ud_stats = MNStats(xen_ud_posterior)
xen_uee, xen_uee_cdf = xen_ud_stats.GetMarginal(0)[0], np.cumsum(xen_ud_stats.GetMarginal(0)[1])
xen_umm, xen_umm_cdf = xen_ud_stats.GetMarginal(1)[0], np.cumsum(xen_ud_stats.GetMarginal(1)[1])
xen_utt, xen_utt_cdf = xen_ud_stats.GetMarginal(2)[0], np.cumsum(xen_ud_stats.GetMarginal(2)[1])
xen_uem, xen_uem_cdf = xen_ud_stats.GetMarginal(3)[0], np.cumsum(xen_ud_stats.GetMarginal(3)[1])
xen_uet, xen_uet_cdf = xen_ud_stats.GetMarginal(4)[0], np.cumsum(xen_ud_stats.GetMarginal(4)[1])
xen_umt, xen_umt_cdf = xen_ud_stats.GetMarginal(5)[0], np.cumsum(xen_ud_stats.GetMarginal(5)[1])
xen_dee, xen_dee_cdf = xen_ud_stats.GetMarginal(6)[0], np.cumsum(xen_ud_stats.GetMarginal(6)[1])
xen_dmm, xen_dmm_cdf = xen_ud_stats.GetMarginal(7)[0], np.cumsum(xen_ud_stats.GetMarginal(7)[1])
xen_dtt, xen_dtt_cdf = xen_ud_stats.GetMarginal(8)[0], np.cumsum(xen_ud_stats.GetMarginal(8)[1])
xen_dem, xen_dem_cdf = xen_ud_stats.GetMarginal(9)[0], np.cumsum(xen_ud_stats.GetMarginal(9)[1])
xen_det, xen_det_cdf = xen_ud_stats.GetMarginal(10)[0], np.cumsum(xen_ud_stats.GetMarginal(10)[1])
xen_dmt, xen_dmt_cdf = xen_ud_stats.GetMarginal(11)[0], np.cumsum(xen_ud_stats.GetMarginal(11)[1])


# Set up special technique for tautau 2 solution.
summed_ud = np.genfromtxt("nsi_multinest/all_nsi_xenon_atmos_coh_prior/all_nsi_xenon_atmos_coh_prior_sumUD.txt")
xen_udSum_stats = MNStats(summed_ud)
udSum_tt, udSum_tt_p = xen_udSum_stats.GetMarginal(2)
cdf_udSum_tt = np.cumsum(udSum_tt_p)


# Read in the xenon posteriors using borexino priors (independent copula).
borex_post = np.genfromtxt("nsi_multinest/all_nsi_xenon_borexino_prior/vector_nsi_xenon_borexino_prior.txt")
borex_stats = MNStats(borex_post)
borex_ee, borex_ee_cdf = borex_stats.GetMarginal(0)[0], np.cumsum(borex_stats.GetMarginal(0)[1])
borex_mm, borex_mm_cdf = borex_stats.GetMarginal(1)[0], np.cumsum(borex_stats.GetMarginal(1)[1])
borex_tt, borex_tt_cdf = borex_stats.GetMarginal(2)[0], np.cumsum(borex_stats.GetMarginal(2)[1])
borex_em, borex_em_cdf = borex_stats.GetMarginal(3)[0], np.cumsum(borex_stats.GetMarginal(3)[1])
borex_et, borex_et_cdf = borex_stats.GetMarginal(4)[0], np.cumsum(borex_stats.GetMarginal(4)[1])
borex_mt, borex_mt_cdf = borex_stats.GetMarginal(5)[0], np.cumsum(borex_stats.GetMarginal(5)[1])

# Take in an nsi and return # of events integrated over energy and zenith
def EventsGenerator(nsi_array, expo, flux, osc_factory):
  det = Detector("dune")
  zenith_arr = np.round(np.linspace(-0.975,max_z,n_z), decimals=3)
  energy_arr = np.array([106.00, 119.00, 133.00, 150.00, 168.00, 188.00, 211.00, 237.00, 266.00, 299.00,
                         335.00, 376.00, 422.00, 473.00, 531.00, 596.00, 668.00, 750.00, 841.00, 944.00])
  obs = np.zeros((n_z,energy_arr.shape[0]-1))  # 18 energy bins, 20 zenith bins

  nsi = NSIparameters(0)
  nsi.epe = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2],
             'em': nsi_array[3], 'et': nsi_array[4], 'mt': nsi_array[5]}
  nsi.epu = {'ee': nsi_array[6], 'mm': nsi_array[7], 'tt': nsi_array[8],
             'em': nsi_array[9], 'et': nsi_array[10], 'mt': nsi_array[11]}
  nsi.epd = {'ee': nsi_array[12], 'mm': nsi_array[13], 'tt': nsi_array[14],
             'em': nsi_array[15], 'et': nsi_array[16], 'mt': nsi_array[17]}


  # Begin event loop.
  for i in range (0, zenith_arr.shape[0]):
    osc = osc_factory.get(oscillator_name='atmospheric', zenith=zenith_arr[i], nsi_parameter=nsi,
                          oscillation_parameter=OSCparameters())
    transformed_flux = osc.transform(flux[i])
    gen = NeutrinoNucleonCCQE("mu", transformed_flux)

    e_a = energy_arr[0]
    for j in range (1, energy_arr.shape[0]):
      e_b = energy_arr[j]
      obs[i][j-1] = 2*pi*0.05*gen.events(e_a, e_b, det, expo)

      # Iterate left edge
      e_a = e_b

  return obs



def PriorFlow18DEmpirical(cube, D, N):
  # Copula prior on electron ee,mm,tt NSI.
  cube[0] = np.interp(cube[0], borex_ee_cdf, borex_ee)
  cube[1] = np.interp(cube[1], borex_mm_cdf, borex_mm)
  cube[2] = np.interp(cube[2], borex_tt_cdf, borex_tt)
  cube[3] = np.interp(cube[3], borex_em_cdf, borex_em)
  cube[4] = np.interp(cube[4], borex_et_cdf, borex_et)
  cube[5] = np.interp(cube[5], borex_mt_cdf, borex_mt)

  # Copula prior on quark NSI.
  d_ee = FrankCopula(cube[6], cube[12], theta=-200.0) #emp_ud_ee.simulate(cube[6], cube[12])
  d_mm = FrankCopula(cube[7], cube[13], theta=-200.0) #emp_ud_mm.simulate(cube[7], cube[13])
  d_em = FrankCopula(cube[9], cube[15], theta=-200.0) #emp_ud_em.simulate(cube[9], cube[15])
  d_et = FrankCopula(cube[10], cube[16], theta=-200.0) #emp_ud_et.simulate(cube[10], cube[16])
  d_mt = FrankCopula(cube[11], cube[17], theta=-200.0) #emp_ud_mt.simulate(cube[11], cube[17])
  cube[6] = 1.12*np.interp(cube[6], xen_uee_cdf, xen_uee)
  cube[7] = 1.12*np.interp(cube[7], xen_umm_cdf, xen_umm)
  cube[9] = 1.12*np.interp(cube[9], xen_uem_cdf, xen_uem)
  cube[10] = 1.12*np.interp(cube[10], xen_uet_cdf, xen_uet)
  cube[11] = 1.12*np.interp(cube[11], xen_umt_cdf, xen_umt)
  cube[12] = np.interp(d_ee, xen_dee_cdf, xen_dee)
  cube[13] = np.interp(d_mm, xen_dmm_cdf, xen_dmm)
  cube[15] = np.interp(d_em, xen_dem_cdf, xen_dem)
  cube[16] = np.interp(d_et, xen_det_cdf, xen_det)
  cube[17] = np.interp(d_mt, xen_dmt_cdf, xen_dmt)

  # empirical copula for the tautau part
  d_tt = emp_ud_tt.simulate(cube[8], cube[14])
  cube[8] = np.interp(cube[8], xen_utt_cdf, xen_utt)
  cube[14] = np.interp(d_tt, xen_dtt_cdf, xen_dtt)


def PriorFlow18D(cube, D, N):
  # Copula prior on electron ee,mm,tt NSI.
  cube[0] = np.interp(cube[0], borex_ee_cdf, borex_ee)
  cube[1] = np.interp(cube[1], borex_mm_cdf, borex_mm)
  cube[2] = np.interp(cube[2], borex_tt_cdf, borex_tt)
  cube[3] = np.interp(cube[3], borex_em_cdf, borex_em)
  cube[4] = np.interp(cube[4], borex_et_cdf, borex_et)
  cube[5] = np.interp(cube[5], borex_mt_cdf, borex_mt)

  # Copula prior on quark NSI.
  d_ee = FrankCopula(cube[6], cube[12], theta=-200.0) #emp_ud_ee.simulate(cube[6], cube[12])
  d_mm = FrankCopula(cube[7], cube[13], theta=-200.0) #emp_ud_mm.simulate(cube[7], cube[13])
  #d_tt = FrankCopula(cube[8], cube[14], theta=-200.0)# emp_ud_tt.simulate(cube[8], cube[14])
  d_em = FrankCopula(cube[9], cube[15], theta=-200.0) #emp_ud_em.simulate(cube[9], cube[15])
  d_et = FrankCopula(cube[10], cube[16], theta=-200.0) #emp_ud_et.simulate(cube[10], cube[16])
  d_mt = FrankCopula(cube[11], cube[17], theta=-200.0) #emp_ud_mt.simulate(cube[11], cube[17])
  cube[6] = np.interp(cube[6], xen_uee_cdf, xen_uee)
  cube[7] = np.interp(cube[7], xen_umm_cdf, xen_umm)
  #cube[8] = np.interp(cube[8], xen_utt_cdf, xen_utt)
  cube[9] = np.interp(cube[9], xen_uem_cdf, xen_uem)
  cube[10] = np.interp(cube[10], xen_uet_cdf, xen_uet)
  cube[11] = np.interp(cube[11], xen_umt_cdf, xen_umt)
  cube[12] = np.interp(d_ee, xen_dee_cdf, xen_dee)
  cube[13] = np.interp(d_mm, xen_dmm_cdf, xen_dmm)
  #cube[14] = np.interp(d_tt, xen_dtt_cdf, xen_dtt)
  cube[15] = np.interp(d_em, xen_dem_cdf, xen_dem)
  cube[16] = np.interp(d_et, xen_det_cdf, xen_det)
  cube[17] = np.interp(d_mt, xen_dmt_cdf, xen_dmt)

  # Use custom tautau prior (for 2 solution)
  #cube[8] = np.interp(cube[8], xen_utt_cdf, xen_utt)
  eps_quark_tt = np.interp(cube[14], cdf_udSum_tt, udSum_tt)
  a = max(eps_quark_tt - 1, -1)
  b = min(eps_quark_tt + 1, 1)
  cube[8] = a + (b - a) * cube[8]
  cube[14] = (eps_quark_tt - cube[8]) / 1.1

  #d_tt = emp_ud_tt.simulate(cube[8], cube[14])
  #cube[8] = np.interp(cube[8], xen_utt_cdf, xen_utt)
  #cube[14] = np.interp(d_tt, xen_dtt_cdf, xen_dtt)



def PlotCopula():
  data = np.genfromtxt("python/multinest_posteriors/all_nsi_xenon_atmos_coh_priorpost_equal_weights.dat")
  z1 = data[:,1]
  z2 = data[:,7]
  u2 = np.random.random(15000)
  u1 = np.random.random(15000)
  y1 = np.zeros_like(u1)
  y2 = np.zeros_like(u1)
  for i in range(0,u1.shape[0]):
    print(i)
    #d_ee = emp_ud_ee.simulate(u1[i], u2[i])
    d_ee = FrankCopula(u1[i], u2[i], theta=-18.0)
    y1[i] = np.interp(u1[i], xen_umm_cdf, xen_umm)
    y2[i] = np.interp(d_ee, xen_dmm_cdf, xen_dmm)

  plt.plot(y1,y2, ls="None", marker=".")
  plt.savefig("copulatest.png")
  plt.clf()
  plt.plot(z1, z2, ls="None", marker=".")
  plt.plot()
  plt.savefig("copulatest_ref.png")



def main():
  # Set the exposure.
  kTon = 40
  years = 10
  days_per_year = 365
  kg_per_kton = 1000000
  exposure = years * days_per_year * kTon * kg_per_kton

  #PlotCopula()

  # Set up factories.
  osc_factory = OscillatorFactory()
  flux_factory = NeutrinoFluxFactory()

  # Prepare flux.
  z_bins = np.round(np.linspace(-0.975, max_z, n_z), decimals=3)
  flux_list = []
  for z in range(0, z_bins.shape[0]):
    this_flux = flux_factory.get('atmospheric', zenith=z_bins[z])
    flux_list.append(this_flux)

  # Construct test data.
  sm_params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  sm_events = EventsGenerator(sm_params, exposure, flux_list, osc_factory)
  null = sm_events.flatten()
  width = np.sqrt(null) + 1

  def LogLikelihood(cube, D, N):
    signal = (EventsGenerator(cube, exposure, flux_list, osc_factory)).flatten()
    likelihood = np.zeros(signal.shape[0])

    for i in range(signal.shape[0]):
      if width[i] == 0:
        continue
      likelihood[i] = -0.5 * np.log(2 * pi * width[i] ** 2) - 0.5 * ((signal[i] - null[i]) / width[i]) ** 2
    return np.sum(likelihood)


  # Define model parameters
  parameters = ["eps_e_ee", "eps_e_mumu", "eps_e_tautau", "eps_e_emu", "eps_e_etau", "eps_e_mutau",
                "eps_ud_ee", "eps_ud_mumu", "eps_ud_tautau", "eps_ud_emu", "eps_ud_etau", "eps_ud_mutau"]
  parameters_18d = ["eps_e_ee", "eps_e_mumu", "eps_e_tautau", "eps_e_emu", "eps_e_etau", "eps_e_mutau",
                    "eps_u_ee", "eps_u_mumu", "eps_u_tautau", "eps_u_emu", "eps_u_etau", "eps_u_mutau",
                    "eps_d_ee", "eps_d_mumu", "eps_d_tautau", "eps_d_emu", "eps_d_etau", "eps_d_mutau"]
  n_params = len(parameters_18d)

  file_string = "prior_flow_18D_v3"
  text_string = "nsi_multinest/" + file_string + "/" + file_string
  json_string = "nsi_multinest/" + file_string + "/params.json"



  # run Multinest
  pymultinest.run(LogLikelihood, PriorFlow18D, n_params, outputfiles_basename=text_string,resume=False, verbose=True,
                  n_live_points=8000, evidence_tolerance=5, sampling_efficiency=0.8)
  json.dump(parameters_18d, open(json_string, 'w'))  # save parameter names
  print("Saving to: \n" + text_string)



if __name__ == "__main__":
  main()

