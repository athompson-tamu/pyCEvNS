import sys
import json
import numpy as np
from numpy import log, exp, pi, zeros, genfromtxt
import matplotlib.pyplot as plt

import pymultinest
import mpi4py

from pyCEvNS.events import*
from pyCEvNS.oscillation import*


def SolarXeGenerator(nsi_array, expo, flux):
  det = Detector("future-xe")
  nsteps_e = 5
  e_lo = 0
  e_hi = 1.0
  energy_arr = np.linspace(e_lo, e_hi, nsteps_e)
  observed_events = np.zeros(nsteps_e - 1)

  # Construct the NSI and flux-oscillation-detection pipeline.
  nsi = NSIparameters(0)
  nsi.epel = {'ee': nsi_array[12], 'mm': nsi_array[13], 'tt': nsi_array[14], 'em':  nsi_array[15], 'et':  nsi_array[16],
              'mt':  nsi_array[17]}
  nsi.eper = {'ee': nsi_array[18], 'mm': nsi_array[19], 'tt': nsi_array[20], 'em':  nsi_array[21], 'et':  nsi_array[22],
              'mt':  nsi_array[23]}
  gen = NeutrinoElectronElasticVector(nsi)

  # Begin event loop.
  this_obs = 0
  flav_arr = np.array(["e", "mu", "tau"])
  e_a = energy_arr[0]
  for j in range (1, nsteps_e):
    e_b = energy_arr[j]

    for f in range(0, flav_arr.shape[0]):  # Integrate over flavors in each energy bin
      observed_events[this_obs] += gen.events(e_a, e_b, str(flav_arr[f]), flux, det, expo)

    # Iterate left edge
    this_obs += 1
    e_a = e_b

  return observed_events


def AtmosDUNEGenerator(nsi_array, expo, flux, osc_factory):
  det = Detector("dune")
  zenith_arr = np.round(np.linspace(-0.975,-0.025,20), decimals=3)
  energy_arr = np.array([106.00, 119.00, 133.00, 150.00, 168.00, 188.00, 211.00, 237.00, 266.00, 299.00,
                         335.00, 376.00, 422.00, 473.00, 531.00, 596.00, 668.00, 750.00, 841.00, 944.00])
  obs = np.zeros((20,19))  # 18 energy bins, 20 zenith bins

  nsi = NSIparameters(0)
  nsi.epe = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2],
             'em': nsi_array[3], 'et': nsi_array[4], 'mt': nsi_array[5]}

  # Begin event loop.
  for i in range (0, zenith_arr.shape[0]):
    osc = osc_factory.get(oscillator_name='atmospheric', zenith=zenith_arr[i], nsi_parameter=nsi,
                          oscillation_parameter=oscillation_parameters())
    transformed_flux = osc.transform(flux[i])
    gen = NeutrinoNucleonCCQE("mu", transformed_flux)

    e_a = energy_arr[0]
    for j in range (1, energy_arr.shape[0]):
      e_b = energy_arr[j]
      obs[i][j-1] = gen.events(e_a, e_b, det, expo)

      # Iterate left edge
      e_a = e_b

  return obs.ravel()


# Take in an nsi and return # of events integrated over energy and zenith
def AtmosXeGenerator(nsi_array, expo, flux):
  det = Detector("future-xe")
  nsteps_e = 2
  energy_arr = [0.004, 0.05]
  #energy_arr = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04]
  zenith_arr = np.round(np.linspace(-0.975, 0.975, 40), decimals=3)
  observed_events = np.zeros(nsteps_e - 1)

  # Construct the NSI and flux-oscillation-detection pipeline.
  nsi = NSIparameters(0)
  nsi.epu = {'ee': nsi_array[6], 'mm': nsi_array[7], 'tt': nsi_array[8],
             'em': nsi_array[9], 'et': nsi_array[10], 'mt': nsi_array[11]}
  gen = NeutrinoNucleusElasticVector(nsi)

  # Begin event loop.
  flav_arr = np.array(["e", "mu", "tau", "ebar", "mubar", "taubar"])
  for i in range (0, zenith_arr.shape[0]):
    this_obs = 0
    e_a = energy_arr[0]
    for j in range (1, nsteps_e):
      e_b = energy_arr[j]

      for f in range(0, flav_arr.shape[0]):  # Integrate over flavors in each energy bin
        observed_events[this_obs] += gen.events(e_a, e_b, str(flav_arr[f]), flux[i], det, expo)

      # Iterate left edge
      this_obs += 1
      e_a = e_b

  return observed_events




# Map the interval [0,1] to the interval [eps_min, eps_max] as a flat prior for each NSI parameter.
def FlatPrior(cube, D, N):
  # DUNE
  cube[0] = 0.5 * (2 * cube[0] - 1)  # eps_ee
  cube[1] = 0.5 * (2 * cube[1] - 1)  # eps_mumu
  cube[2] = 0.5 * (2 * cube[2] - 1)  # eps_tata
  cube[3] = 0.12 * (2 * cube[3] - 1)  # eps_emu
  cube[4] = 0.3 * (2 * cube[4] - 1)  # eps_eta
  cube[5] = 0.028 * (2 * cube[5] - 1)  # eps_muta
  # Xe(N)
  cube[6] = 0.5 * (2 * cube[6] - 1)  # eps_ee
  cube[7] = 0.5 * (2 * cube[7] - 1)  # eps_mumu
  cube[8] = 0.5 * (2 * cube[8] - 1)  # eps_tata
  cube[9] = 0.1 * (2 * cube[9] - 1)  # eps_emu
  cube[10] = 0.1 * (2 * cube[10] - 1)  # eps_eta
  cube[11] = 0.1 * (2 * cube[11] - 1)  # eps_muta
  # Xe(e)
  cube[12] = 0.5 * (2 * cube[12] - 1)  # eps_ee
  cube[13] = 0.5 * (2 * cube[13] - 1)  # eps_mumu
  cube[14] = 0.5 * (2 * cube[14] - 1)  # eps_tata
  cube[15] = 0.1 * (2 * cube[15] - 1)  # eps_emu
  cube[16] = 0.1 * (2 * cube[16] - 1)  # eps_eta
  cube[17] = 0.1 * (2 * cube[17] - 1)  # eps_muta
  cube[18] = 0.5 * (2 * cube[18] - 1)  # eps_ee
  cube[19] = 0.5 * (2 * cube[19] - 1)  # eps_mumu
  cube[20] = 0.5 * (2 * cube[20] - 1)  # eps_tata
  cube[21] = 0.1 * (2 * cube[21] - 1)  # eps_emu
  cube[22] = 0.1 * (2 * cube[22] - 1)  # eps_eta
  cube[23] = 0.1 * (2 * cube[23] - 1)  # eps_muta




def main():
  # Set the exposure (Future Xe):
  xe_kTon = 0.1
  dune_kTon = 40
  years = 10
  days_per_year = 365
  kg_per_kton = 1000000
  xe_exposure = years * days_per_year * xe_kTon * kg_per_kton
  dune_exposure = years * days_per_year * dune_kTon * kg_per_kton


  # Set up factories.
  flux_factory = NeutrinoFluxFactory()
  osc_factory = OscillatorFactory()

  # Prepare flux.
  solar_flux = flux_factory.get('solar')
  solar_osc = osc_factory.get(oscillator_name='solar', nsi_parameter=NSIparameters(0),
                              oscillation_parameter=oscillation_parameters())
  solar_flux = solar_osc.transform(solar_flux)

  z_bins = np.round(np.linspace(-0.975, 0.975, 40), decimals=3)
  xe_zenith_flux = []
  for z in range(0, 40):
    this_flux = flux_factory.get('atmospheric_extended', zenith=z_bins[z])
    atmos_osc = osc_factory.get(oscillator_name='atmospheric', zenith=z_bins[z], nsi_parameter=NSIparameters(0),
                                oscillation_parameter=oscillation_parameters())
    osc_atmos = atmos_osc.transform(this_flux)
    xe_zenith_flux.append(osc_atmos)

  dune_zenith_flux = []
  for z in range(0, 20):
      this_flux = flux_factory.get('atmospheric', zenith=z_bins[z])
      dune_zenith_flux.append(this_flux)


  # Construct test data.
  sm_params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  null = AtmosXeGenerator(sm_params, xe_exposure, xe_zenith_flux)
  null = np.append(null, AtmosDUNEGenerator(sm_params, dune_exposure, dune_zenith_flux, osc_factory))
  null = np.append(null, SolarXeGenerator(sm_params, xe_exposure, solar_flux))
  width = np.sqrt(null) + 1


  # Define the log likelihood.
  def LogLikelihood(cube, N, D):
    signal = AtmosXeGenerator(cube, xe_exposure, xe_zenith_flux)
    signal = np.append(signal, AtmosDUNEGenerator(cube, dune_exposure, dune_zenith_flux, osc_factory))
    signal = np.append(signal, SolarXeGenerator(cube, xe_exposure, solar_flux))
    likelihood = np.zeros(signal.shape[0])

    for i in range(signal.shape[0]):
      if width[i] == 0:
        continue
      likelihood[i] = -0.5 * np.log(2 * pi * width[i] ** 2) - 0.5 * ((signal[i] - null[i]) / width[i]) ** 2
    return np.sum(likelihood)



  # Define model parameters
  parameters = ["epd_ee", "eped_mumu", "eped_tautau",
                "eped_em", "eped_etau", "eped_mutau",
                "epx_ee", "epx_mumu", "epx_tautau",
                "epx_em", "epx_etau", "epx_mutau",
                "epel_ee", "epel_mumu", "epel_tautau",
                "epel_em", "epel_etau", "epel_mutau",
                "eper_ee", "eper_mumu", "eper_tautau",
                "eper_em", "eper_etau", "eper_mutau"]

  n_params = len(parameters)

  file_string = "pheno_nsi_combined_ALL"
  text_string = "nsi_multinest/" + file_string + "/" + file_string
  json_string = "nsi_multinest/" + file_string + "/params.json"



  # run Multinest
  pymultinest.run(LogLikelihood, FlatPrior, n_params, outputfiles_basename=text_string, resume=False,
                  verbose=True, n_live_points=8000, evidence_tolerance=20, sampling_efficiency=0.8)
  json.dump(parameters, open(json_string, 'w'))  # save parameter names
  print("Saving to: \n" + text_string)



if __name__ == "__main__":
  main()
