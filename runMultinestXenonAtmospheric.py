import sys
import json
import numpy as np
from numpy import log, exp, pi, zeros, genfromtxt
import matplotlib.pyplot as plt

import pymultinest
import mpi4py

from pyCEvNS.events import*
from pyCEvNS.oscillation import*

from MNStats import MNStats



coh_posterior = np.genfromtxt("nsi_multinest/coherent_csi_ar/coherent_csi_ar.txt")
coh_stats = MNStats(coh_posterior)
coh_uee, coh_uee_cdf = coh_stats.GetMarginal(0)[0], np.cumsum(coh_stats.GetMarginal(0)[1])
coh_umm, coh_umm_cdf = coh_stats.GetMarginal(1)[0], np.cumsum(coh_stats.GetMarginal(1)[1])
#coh_utt, coh_utt_cdf = coh_stats.GetMarginal(2)[0], np.cumsum(coh_stats.GetMarginal(2)[1])
coh_uem, coh_uem_cdf = coh_stats.GetMarginal(2)[0], np.cumsum(coh_stats.GetMarginal(2)[1])
coh_uet, coh_uet_cdf = coh_stats.GetMarginal(3)[0], np.cumsum(coh_stats.GetMarginal(3)[1])
coh_umt, coh_umt_cdf = coh_stats.GetMarginal(4)[0], np.cumsum(coh_stats.GetMarginal(4)[1])
coh_dee, coh_dee_cdf = coh_stats.GetMarginal(5)[0], np.cumsum(coh_stats.GetMarginal(5)[1])
coh_dmm, coh_dmm_cdf = coh_stats.GetMarginal(6)[0], np.cumsum(coh_stats.GetMarginal(6)[1])
#coh_dtt, coh_dtt_cdf = coh_stats.GetMarginal(8)[0], np.cumsum(coh_stats.GetMarginal(8)[1])
coh_dem, coh_dem_cdf = coh_stats.GetMarginal(7)[0], np.cumsum(coh_stats.GetMarginal(7)[1])
coh_det, coh_det_cdf = coh_stats.GetMarginal(8)[0], np.cumsum(coh_stats.GetMarginal(8)[1])
coh_dmt, coh_dmt_cdf = coh_stats.GetMarginal(9)[0], np.cumsum(coh_stats.GetMarginal(9)[1])




def FrankCopula(u1, v2, theta):
  u2 = - (1/theta) * np.log(1 + (v2 * (1 - np.exp(-theta))) / (v2 * (np.exp(-theta * u1) - 1) -
                                                                        np.exp(-theta * u1)))
  return u2



def InvertMarginal(u1, u2):
  r1[i] = np.interp(u1, cdf1, x1)
  r2[i] = np.interp(u2, cdf2, x2)



# Take in an nsi and return # of events integrated over energy and zenith
def EventsGenerator(nsi_array, expo, flux, osc_factory):
  det = Detector("future-xe")
  nsteps_e = 2
  energy_arr = np.linspace(0.004,0.05,nsteps_e)
  #energy_arr = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04]
  zenith_arr = np.round(np.linspace(-0.975, 0.975, 40), decimals=3)
  observed_events = np.zeros(nsteps_e - 1)

  # Construct the NSI and flux-oscillation-detection pipeline.
  nsi = NSIparameters(0)
  nsi.epu = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2],
             'em': nsi_array[3], 'et': nsi_array[4], 'mt': nsi_array[5]}
  nsi.epd = {'ee': nsi_array[6], 'mm': nsi_array[7], 'tt': nsi_array[8],
             'em': nsi_array[9], 'et': nsi_array[10], 'mt': nsi_array[11]}
  gen = NeutrinoNucleusElasticVector(nsi)

  # Begin event loop.
  flav_arr = np.array(["e", "mu", "tau", "ebar", "mubar", "taubar"])
  for i in range (0, zenith_arr.shape[0]):
    osc = osc_factory.get(oscillator_name='atmospheric', zenith=zenith_arr[i], nsi_parameter=nsi,
                          oscillation_parameter=OSCparameters())
    transformed_flux = osc.transform(flux[i])
    this_obs = 0
    e_a = energy_arr[0]
    for j in range (1, nsteps_e):
      e_b = energy_arr[j]

      for f in range(0, flav_arr.shape[0]):  # Integrate over flavors in each energy bin
        observed_events[this_obs] += gen.events(e_a, e_b, str(flav_arr[f]), transformed_flux, det, expo)

      # Iterate left edge
      this_obs += 1
      e_a = e_b

  return observed_events




# Map the interval [0,1] to the interval [eps_min, eps_max] as a flat prior for each NSI parameter.
def FlatPrior(cube, ndim, nparams):
  cube[0] = 0.5 * (2 * cube[0] - 1)  # eps_u_ee
  cube[1] = 0.5 * (2 * cube[1] - 1)  # eps_u_mumu
  cube[2] = 0.5 * (2 * cube[2] - 1)  # eps_u_tata
  cube[3] = 0.5 * (2 * cube[3] - 1)  # eps_u_emu
  cube[4] = 0.5 * (2 * cube[4] - 1)  # eps_u_eta
  cube[5] = 0.5 * (2 * cube[5] - 1)  # eps_u_muta
  #cube[6] = 0.5 * (2 * cube[6] - 1)  # eps_d_ee
  #cube[7] = 0.5 * (2 * cube[7] - 1)  # eps_d_mumu
  #cube[8] = 0.5 * (2 * cube[8] - 1)  # eps_d_tata
  #cube[9] = 0.5 * (2 * cube[9] - 1)  # eps_d_emu
  #cube[10] = 0.5 * (2 * cube[10] - 1)  # eps_d_eta
  #cube[11] = 0.5 * (2 * cube[11] - 1)  # eps_d_muta


def MarginalPrior(cube, D, N):
  cube[0] = np.interp(cube[0], coh_uee_cdf, coh_uee)
  cube[1] = np.interp(cube[1], coh_umm_cdf, coh_umm)
  cube[2] = (2 * cube[2] - 1)  # eps_u_tata
  cube[3] = np.interp(cube[3], coh_uem_cdf, coh_uem)
  cube[4] = np.interp(cube[4], coh_uet_cdf, coh_uet)
  cube[5] = np.interp(cube[5], coh_umt_cdf, coh_umt)
  cube[6] = np.interp(cube[6], coh_dee_cdf, coh_dee)
  cube[7] = np.interp(cube[7], coh_dmm_cdf, coh_dmm)
  cube[8] = (2 * cube[8] - 1)  # eps_u_tata
  cube[9] = np.interp(cube[9], coh_dem_cdf, coh_dem)
  cube[10] = np.interp(cube[10], coh_det_cdf, coh_det)
  cube[11] = np.interp(cube[11], coh_dmt_cdf, coh_dmt)


def CopulaPrior(cube, D, N):
  cube[6] = np.interp(FrankCopula(cube[0], cube[6], -100), coh_dee_cdf, coh_dee)  # eps_d_ee
  cube[0] = np.interp(cube[0], coh_uee_cdf, coh_uee)

  cube[7] = np.interp(FrankCopula(cube[1], cube[7], -100), coh_dmm_cdf, coh_dmm)  # eps_d_mm
  cube[1] = np.interp(cube[1], coh_umm_cdf, coh_umm)

  #cube[8] = np.interp(cube[8], coh_dtt_cdf, coh_dtt)  # eps_u_tt
  cube[8] = (2 * cube[8] - 1)  # eps_d_tata
  #cube[2] = np.interp(cube[2], coh_utt_cdf, coh_utt)  # eps_u_tt
  cube[2] = (2 * cube[2] - 1)  # eps_u_tata

  cube[9] = np.interp(FrankCopula(cube[3], cube[9], -100), coh_dem_cdf, coh_dem)  # eps_d_em
  cube[3] = np.interp(cube[3], coh_uem_cdf, coh_uem)

  cube[10] = np.interp(FrankCopula(cube[4], cube[10], -100), coh_det_cdf, coh_det)  # eps_d_et
  cube[4] = np.interp(cube[4], coh_uet_cdf, coh_uet)

  cube[11] = np.interp(FrankCopula(cube[5], cube[11], -100), coh_dmt_cdf, coh_dmt)  # eps_d_mt
  cube[5] = np.interp(cube[5], coh_umt_cdf, coh_umt)


def main():
  # Set the exposure (Future Xe):
  kTon = 0.1
  years = 10
  days_per_year = 365
  kg_per_kton = 1000000
  exposure = years * days_per_year * kTon * kg_per_kton

  # Set up factories.
  flux_factory = NeutrinoFluxFactory()
  osc_factory = OscillatorFactory()

  # Prepare flux.
  z_bins = np.round(np.linspace(-0.975, 0.975, 40), decimals=3)
  flux_list = []
  for z in range(0, z_bins.shape[0]):
    this_flux = flux_factory.get('atmospheric_extended', zenith=z_bins[z])
    flux_list.append(this_flux)

  # Construct test data.
  sm_params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  n_sm = EventsGenerator(sm_params, exposure, flux_list, osc_factory)
  width = np.sqrt(n_sm) + 1

  def LogLikelihood(cube, N, D):
    n_signal = EventsGenerator(cube, exposure, flux_list, osc_factory)
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
    e_bins = np.linspace(5,39,18)
    #e_bins = [0.0055,0.0065,0.0075,0.0085,0.0095,0.015,0.025,0.035]
    print("nsi1")
    nsi1 = EventsGenerator([0.2, 0, 0, 0.5, 0.0, 0.0, 1.5*np.pi], exposure, flux_list, osc_factory)
    print(nsi1)
    sm_error = np.sqrt(n_sm)
    nsi_error = np.sqrt(nsi1)
    plt.plot(e_bins, n_sm, label="SM", drawstyle='steps-mid', color='k')
    plt.plot(e_bins, nsi1, label=r"$\epsilon_{ee} = 0.2$, $\epsilon_{e\mu} = 0.5$",
             drawstyle='steps-mid', ls='dashed', color='r')
    plt.fill_between(e_bins, nsi1 - nsi_error, nsi1 + nsi_error, color='lightgray')
    plt.fill_between(e_bins, n_sm - sm_error, n_sm + sm_error, color='lightgray')
    plt.xlim((5,39))
    plt.ylim((1, 100))
    plt.xlabel(r'$E_R$ [KeV]')
    plt.ylabel('Events')
    plt.yscale("log")
    plt.legend()
    #plt.title(r'LXe: solar $\nu - e$ counts, 1 kTon-Year Exposure')
    plt.savefig("xenon_rates_atmospheric.png")
    plt.savefig("xenon_rates_atmospheric.pdf")




  # Define model parameters
  parameters = ["eps_u_ee", "eps_u_mumu", "eps_u_tautau", "eps_u_emu", "eps_u_etau", "eps_u_mutau",
                "eps_d_ee", "eps_d_mumu", "eps_d_tautau", "eps_d_emu", "eps_d_etau", "eps_d_mutau"]
  n_params = len(parameters)

  file_string = "all_nsi_xenon_atmos_coh_prior"
  text_string = "nsi_multinest/" + file_string + "/" + file_string
  json_string = "nsi_multinest/" + file_string + "/params.json"



  # run Multinest
  pymultinest.run(LogLikelihood, CopulaPrior, n_params, outputfiles_basename=text_string, resume=False,
                  verbose=True, n_live_points=5000, evidence_tolerance=0.5, sampling_efficiency=0.8)
  json.dump(parameters, open(json_string, 'w'))  # save parameter names
  print("Saving to: \n" + text_string)



if __name__ == "__main__":
  main()
