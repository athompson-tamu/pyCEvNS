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
  det = Detector("dune")
  nsteps_e = 32
  e_lo = 250
  e_hi = 8000
  energy_arr = np.linspace(e_lo, e_hi, nsteps_e)  # N bin edges
  observed_events = np.zeros(nsteps_e - 1)  # N-1 bin centers

  # Construct the NSI and flux-oscillation-detection pipeline.
  nsi = NSIparameters(0)
  nsi.epe = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2],
             'em': nsi_array[3], 'et': nsi_array[4], 'mt': nsi_array[5]}

  osc = osc_factory.get(oscillator_name='beam', nsi_parameter=nsi,
                        oscillation_parameter=OSCparameters(delta=nsi_array[6]), length=1297000)
  transformed_flux = osc.transform(flux)
  gen = NeutrinoNucleonCCQE("e", transformed_flux)

  # Begin event loop.
  e_a = energy_arr[0]
  for j in range (1, nsteps_e):
    e_b = energy_arr[j]
    observed_events[j-1] = gen.events(e_a, e_b, det, expo)

    # Iterate left edge
    e_a = e_b
  print(np.sum(observed_events))
  return observed_events




# Map the interval [0,1] to the interval [eps_min, eps_max] as a flat prior for each NSI parameter.
def FlatPrior(cube, ndim, nparams):
  cube[0] = 0.5 * (2 * cube[0] - 1)  # eps_ee
  cube[1] = 0.5 * (2 * cube[1] - 1)  # eps_mumu
  cube[2] = 0.5 * (2 * cube[2] - 1)  # eps_tautau
  cube[3] = 0.1 * (2 * cube[3] - 1)  # eps_emu
  cube[4] = 0.1 * (2 * cube[4] - 1)  # eps_etau
  cube[5] = 0.1 * (2 * cube[5] - 1)  # eps_mutau
  cube[6] = 2 * pi * cube[6]  # delta_CP




def main():
  # Set the exposure.
  kTon = 40
  years = 5
  days_per_year = 365
  kg_per_kton = 1000000
  exposure = years * days_per_year * kTon * kg_per_kton

  # Set up factories.
  flux_factory = NeutrinoFluxFactory()
  beam_flux = flux_factory.get('far_beam_nu')
  osc_factory = OscillatorFactory()

  # Construct test data.
  sm_params = [0, 0, 0, 0, 0, 0, 1.5*pi]
  n_sm = EventsGenerator(sm_params, exposure, beam_flux, osc_factory)
  width = np.sqrt(n_sm) + 1

  def LogLikelihood(cube, N, D):
    n_signal = EventsGenerator(cube, exposure, beam_flux, osc_factory)
    likelihood = np.zeros(n_signal.shape[0])

    for i in range(n_signal.shape[0]):
      if width[i] == 0:
        continue
      likelihood[i] = -0.5 * np.log(2 * pi * width[i] ** 2) - 0.5 * ((n_signal[i] - n_sm[i]) / width[i]) ** 2
    return np.sum(likelihood)



  # Prepare some sample event rate plots.
  plot = True
  if plot == True:
    e_bins = np.linspace(375, 7875, 31)
    nsi1 = EventsGenerator([0.5, 0, 0, 0, 0, 0.2*np.exp(-1j*pi/2), pi/3], exposure, beam_flux, osc_factory)
    plt.plot(e_bins, n_sm, label="SM", drawstyle='steps')
    plt.plot(e_bins, nsi1, label=r"$\epsilon_{ee} = 0.5$, $\epsilon_{\mu\tau} = 0.2 e^{-i \pi/2}$",
             drawstyle='steps', ls='dashed')
    plt.xlabel(r'$E_\nu$ [MeV]')
    plt.ylabel('Events')
    plt.legend()
    plt.title(r'DUNE FD: $\nu_e$ appearance, 4x5 kTon-Year Exposure')
    #plt.savefig("dune_fd_beam_rate_e_appearance.png")
    #plt.savefig("dune_fd_beam_rate_e_appearance.pdf")




  # Define model parameters
  parameters = ["eps_ee", "eps_mumu", "eps_tautau", "eps_emu", "eps_etau", "eps_mutau", "delta_cp"]
  n_params = len(parameters)

  file_string = "all_nsi_dune_beam_e_appearance"
  text_string = "nsi_multinest/" + file_string + "/" + file_string
  json_string = "nsi_multinest/" + file_string + "/params.json"



  # run Multinest
  #pymultinest.run(LogLikelihood, FlatPrior, n_params, outputfiles_basename=text_string,resume=False, verbose=True,
  #                n_live_points=4000, evidence_tolerance=0.5, sampling_efficiency=0.3)
  #json.dump(parameters, open(json_string, 'w'))  # save parameter names
  #print("Saving to: \n" + text_string)



if __name__ == "__main__":
  main()

