import sys
import json
import numpy as np
from numpy import log, exp, pi, zeros, genfromtxt
import matplotlib.pyplot as plt

import pymultinest
import mpi4py

from pyCEvNS.events import*
from pyCEvNS.oscillation import*



# Take in an nsi and return # of events integrated over energy and zenith
def EventsGenerator(nsi_array, expo, flux, osc_factory):
  det = Detector("future-xe")
  nsteps_e = 9
  energy_arr = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04]
  zenith_arr = np.round(np.linspace(-0.975, 0.975, 40), decimals=3)
  observed_events = np.zeros(nsteps_e - 1)

  # Construct the NSI and flux-oscillation-detection pipeline.
  nsi = NSIparameters(0)
  nsi.epu = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2],
             'em': nsi_array[3], 'et': nsi_array[4], 'mt': nsi_array[5]}
  gen = NeutrinoNucleusElasticVector(nsi)

  # Begin event loop.
  flav_arr = np.array(["e", "mu", "tau", "ebar", "mubar", "taubar"])
  for i in range (0, zenith_arr.shape[0]):
    osc = osc_factory.get(oscillator_name='atmospheric', zenith=zenith_arr[i], nsi_parameter=nsi,
                          oscillation_parameter=OSCparameters(delta=nsi_array[6]))
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

  return np.round(observed_events)




# Map the interval [0,1] to the interval [eps_min, eps_max] as a flat prior for each NSI parameter.
def FlatPrior(cube, ndim, nparams):
  cube[0] = 0.5 * (2 * cube[0] - 1)  # eps_ee
  cube[1] = 0.5 * (2 * cube[1] - 1)  # eps_mumu
  cube[2] = 0.5 * (2 * cube[2] - 1)  # eps_tata
  cube[3] = 0.1 * (2 * cube[3] - 1)  # eps_emu
  cube[4] = 0.1 * (2 * cube[4] - 1)  # eps_eta
  cube[5] = 0.1 * (2 * cube[5] - 1)  # eps_muta
  cube[6] = 2 * pi * cube[6]  # delta_CP




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
  sm_params = [0, 0, 0, 0, 0, 0, 1.5 * pi]
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
    e_bins = [0.0055,0.0065,0.0075,0.0085,0.0095,0.015,0.025,0.035]
    print("nsi1")
    nsi1 = EventsGenerator([0.5, 0, 0, -0.2, 0.0, 0.0, np.pi/3], exposure, flux_list, osc_factory)
    print(nsi1)
    plt.plot(e_bins, n_sm, label="SM", drawstyle='steps')
    plt.plot(e_bins, nsi1, label=r"$\epsilon_{ee} = 0.5$, $\epsilon_{e\mu} = -0.2$", drawstyle='steps', ls='dashed')
    plt.xlabel(r'$E_R$ [MeV]')
    plt.ylabel('Events')
    plt.yscale("log")
    plt.legend()
    plt.title(r'LXe: solar $\nu - e$ counts, 1 kTon-Year Exposure')
    plt.savefig("xenon_rates_solar_electron_Emax1MeV.png")
    plt.savefig("xenon_rates_solar_electron_Emax1MeV.pdf")




  # Define model parameters
  parameters = ["eps_ee", "eps_mumu", "eps_tautau", "eps_emu", "eps_etau", "eps_mutau", "delta_cp"]
  n_params = len(parameters)

  file_string = "all_nsi_xenon_atmos"
  text_string = "nsi_multinest/" + file_string + "/" + file_string
  json_string = "nsi_multinest/" + file_string + "/params.json"



  # run Multinest
  pymultinest.run(LogLikelihood, FlatPrior, n_params, outputfiles_basename=text_string, resume=False,
                  verbose=True, n_live_points=4000, evidence_tolerance=0.5, sampling_efficiency=0.3)
  json.dump(parameters, open(json_string, 'w'))  # save parameter names
  print("Saving to: \n" + text_string)



if __name__ == "__main__":
  main()