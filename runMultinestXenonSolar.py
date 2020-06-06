import sys
import json
import numpy as np
from numpy import log, exp, pi, zeros, genfromtxt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import pymultinest
import mpi4py

from pyCEvNS.events import*
from pyCEvNS.oscillation import*

import pynverse as pynv
from MNStats import MNStats

import rpy2
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, ListVector
from rpy2.robjects import r
base = importr('base')
utils = importr('utils')
rcopula = importr('copula')


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
      return float(pynv.inversefunc(ddv, y_values=v, domain=[0, 1], open_domain=[True, True]))
    except:
      print("passing...")
      return v

emp_lr_ee = EmpricalCopula("nsi_multinest/borexino_12dim_nsi_nuissance/borexino_12dim_nsi_nuissancepost_equal_weights.dat", 1, 2)



# Take in an nsi and return # of events integrated over energy and zenith
def EventsGenerator(nsi_array, expo, flux, osc_factory):
  det = Detector("future-xe")
  nsteps_e = 101
  e_lo = 0.5
  e_hi = 1.0
  energy_arr = np.linspace(e_lo, e_hi, nsteps_e)
  observed_events = np.zeros(nsteps_e - 1)

  # Construct the NSI and flux-oscillation-detection pipeline.
  nsi = NSIparameters(0)
  nsi.epel = {'ee': nsi_array[0], 'mm': nsi_array[2], 'tt': nsi_array[4],
              'em': nsi_array[6], 'et': nsi_array[8], 'mt': nsi_array[10]}
  nsi.eper = {'ee': nsi_array[1], 'mm': nsi_array[3], 'tt': nsi_array[5],
              'em': nsi_array[7], 'et': nsi_array[9], 'mt': nsi_array[11]}
  gen = NeutrinoElectronElasticVector(nsi)
  osc = osc_factory.get(oscillator_name='solar', nsi_parameter=nsi,
                        oscillation_parameter=OSCparameters())
  transformed_flux = osc.transform(flux)

  # Begin event loop.
  this_obs = 0
  flav_arr = np.array(["e", "mu", "tau"])
  e_a = energy_arr[0]
  for j in range (1, nsteps_e):
    e_b = energy_arr[j]

    for f in range(0, flav_arr.shape[0]):  # Integrate over flavors in each energy bin
      observed_events[this_obs] += gen.events(e_a, e_b, str(flav_arr[f]), transformed_flux, det, expo)

    # Iterate left edge
    this_obs += 1
    e_a = e_b

  return gaussian_filter(observed_events, sigma=5, mode='nearest')




# Take the borexino prior.
# Read in the borexino posteriors.
borex_post = np.genfromtxt("nsi_multinest/borexino_12dim_nsi_nuissance/borexino_12dim_nsi_nuissance.txt")
borex_stats = MNStats(borex_post)
borex_lee, borex_lee_cdf = borex_stats.GetMarginal(0)[0], np.cumsum(borex_stats.GetMarginal(0)[1])
borex_ree, borex_ree_cdf = borex_stats.GetMarginal(1)[0], np.cumsum(borex_stats.GetMarginal(1)[1])
borex_lmm, borex_lmm_cdf = borex_stats.GetMarginal(2)[0], np.cumsum(borex_stats.GetMarginal(2)[1])
borex_rmm, borex_rmm_cdf = borex_stats.GetMarginal(3)[0], np.cumsum(borex_stats.GetMarginal(3)[1])
borex_ltt, borex_ltt_cdf = borex_stats.GetMarginal(4)[0], np.cumsum(borex_stats.GetMarginal(4)[1])
borex_rtt, borex_rtt_cdf = borex_stats.GetMarginal(5)[0], np.cumsum(borex_stats.GetMarginal(5)[1])
borex_lem, borex_lem_cdf = borex_stats.GetMarginal(6)[0], np.cumsum(borex_stats.GetMarginal(6)[1])
borex_rem, borex_rem_cdf = borex_stats.GetMarginal(7)[0], np.cumsum(borex_stats.GetMarginal(7)[1])
borex_let, borex_let_cdf = borex_stats.GetMarginal(8)[0], np.cumsum(borex_stats.GetMarginal(8)[1])
borex_ret, borex_ret_cdf = borex_stats.GetMarginal(9)[0], np.cumsum(borex_stats.GetMarginal(9)[1])
borex_lmt, borex_lmt_cdf = borex_stats.GetMarginal(10)[0], np.cumsum(borex_stats.GetMarginal(10)[1])
borex_rmt, borex_rmt_cdf = borex_stats.GetMarginal(11)[0], np.cumsum(borex_stats.GetMarginal(11)[1])

def BorexinoPrior(cube, n, d):
  # The only significant correlation is between eL_ee and eR_ee.
  #eR_ee = emp_lr_ee.simulate(cube[0], cube[1])
  cube[0] = np.interp(cube[0], borex_lee_cdf, borex_lee)
  cube[1] = np.interp(cube[1], borex_ree_cdf, borex_ree)
  cube[2] = np.interp(cube[2], borex_lmm_cdf, borex_lmm)
  cube[3] = np.interp(cube[3], borex_rmm_cdf, borex_rmm)
  cube[4] = np.interp(cube[4], borex_ltt_cdf, borex_ltt)
  cube[5] = np.interp(cube[5], borex_rtt_cdf, borex_rtt)
  cube[6] = np.interp(cube[6], borex_lem_cdf, borex_lem)
  cube[7] = np.interp(cube[7], borex_rem_cdf, borex_rem)
  cube[8] = np.interp(cube[8], borex_let_cdf, borex_let)
  cube[9] = np.interp(cube[9], borex_ret_cdf, borex_ret)
  cube[10] = np.interp(cube[10], borex_lmt_cdf, borex_lmt)
  cube[11] = np.interp(cube[11], borex_rmt_cdf, borex_rmt)

# Map the interval [0,1] to the interval [eps_min, eps_max] as a flat prior for each NSI parameter.
def FlatPrior(cube, ndim, nparams):
  cube[0] = 0.5 * (2 * cube[0] - 1)  # eps_ee L
  cube[1] = 0.5 * (2 * cube[1] - 1)  # eps_ee R
  cube[2] = 0.5 * (2 * cube[2] - 1)  # eps_mumu L
  cube[3] = 0.5 * (2 * cube[3] - 1)  # eps_mumu R
  cube[4] = 0.5 * (2 * cube[4] - 1)  # eps_tata L
  cube[5] = 0.5 * (2 * cube[5] - 1)  # eps_tata R
  cube[6] = 0.5 * (2 * cube[6] - 1)  # eps_emu L
  cube[7] = 0.5 * (2 * cube[7] - 1)  # eps_emu R
  cube[8] = 0.5 * (2 * cube[8] - 1)  # eps_eta L
  cube[9] = 0.5 * (2 * cube[9] - 1)  # eps_eta R
  cube[10] = 0.5 * (2 * cube[10] - 1)  # eps_muta L
  cube[11] = 0.5 * (2 * cube[11] - 1)  # eps_muta R



def main():
  # Set the exposure (Future Xe):
  kTon = 0.1
  years = 10
  days_per_year = 365
  kg_per_kton = 1000000
  exposure = years * days_per_year * kTon * kg_per_kton

  # Set up factories.
  flux_factory = NeutrinoFluxFactory()
  solar_flux = flux_factory.get('solar')
  osc_factory = OscillatorFactory()

  # Construct test data.
  #sm_params = [-0.37, -1.6, 0.23, -0.16, 0.33, -0.27, 0.072, -0.046, 0.057, -0.051, 0.064, -0.063]
  sm_params = [0,0,0,0,0,0,0,0,0,0,0,0]
  n_sm = EventsGenerator(sm_params, exposure, solar_flux, osc_factory)
  width = np.sqrt(n_sm) + 1

  def LogLikelihood(cube, N, D):
    n_signal = EventsGenerator(cube, exposure, solar_flux, osc_factory)
    likelihood = np.zeros(n_signal.shape[0])

    for i in range(n_signal.shape[0]):
      if width[i] == 0:
        continue
      likelihood[i] = -0.5 * np.log(2 * pi * width[i] ** 2) - 0.5 * ((n_signal[i] - n_sm[i]) / width[i]) ** 2
    return np.sum(likelihood)



  # Prepare some sample event rate plots.
  plot = False
  if plot == True:
    print(n_sm)
    e_bins = np.linspace(502.5, 997.5, 100)
    print("nsi1")
    nsi1 = EventsGenerator([0.1, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], exposure, solar_flux, osc_factory)
    print(nsi1)
    plt.errorbar(e_bins, n_sm, yerr=np.sqrt(n_sm), label="SM", color='k', ls="None", marker='.')
    plt.plot(e_bins, nsi1, label=r"$\epsilon^L_{ee} = 0.1$, $\epsilon^R_{ee} = -0.5$",
             drawstyle='steps-mid', ls='dashed', color='r')
    plt.xlim((500,1000))
    plt.xlabel(r'$E_R$ [KeV]')
    plt.ylabel('Events')
    plt.yscale("log")
    plt.legend()
    #plt.title(r'LXe: solar $\nu - e$ counts, 1 kTon-Year Exposure')
    plt.savefig("plots/rates/xenon/png/enon_rates_solar_electron_be7edge.png")
    plt.savefig("plots/rates/xenon/pdf/xenon_rates_solar_electron_be7edge.pdf")




  # Define model parameters
  parameters = ["epsl_ee", "epsr_ee", "epsl_mumu", "epsr_mumu", "epsl_tautau", "epsr_tautau",
                "epsl_emu", "epsr_emu", "epsl_etau", "epsr_etau", "epsl_mutau", "epsr_mutau"]
  n_params = len(parameters)

  file_string = "all_nsi_xenon_borexino_prior"
  text_string = "nsi_multinest/" + file_string + "/" + file_string
  json_string = "nsi_multinest/" + file_string + "/params.json"



  # run Multinest
  pymultinest.run(LogLikelihood, BorexinoPrior, n_params, outputfiles_basename=text_string, resume=False, verbose=True,
                  n_live_points=4000, evidence_tolerance=0.5, sampling_efficiency=0.8)
  json.dump(parameters, open(json_string, 'w'))  # save parameter names
  print("Saving to: \n" + text_string)



if __name__ == "__main__":
  main()
