import sys
import json
import numpy as np
from numpy import log, exp, pi, zeros, genfromtxt
import matplotlib.pyplot as plt
import pandas as pd

import pymultinest
import mpi4py

from rebin_flux import RebinAtmosphericFlux

from pyCEvNS.events import*
from pyCEvNS.oscillation import*
from pyCEvNS.plot import CrediblePlot


# Modulate the atmospheric neutrino flux by the survival and transition probabilities.
def ModulateFlux(params, flux_array):
  nsi = NSIparameters(0)
  #nsi.epe = {'ee': params[0], 'mm': params[1], 'tt': params[2], 'em': params[3], 'et': params[4],
   #          'mt': params[5]}
  nsi.epu = {'ee': params[0], 'mm': params[1], 'tt': params[2], 'em': params[3], 'et': params[4],
             'mt': params[5]}
  nsi.epd = {'ee': params[0], 'mm': params[1], 'tt': params[2], 'em': params[3], 'et': params[4],
             'mt': params[5]}

  entries = flux_array.shape[0]
  flux_mod = np.zeros([flux_array.shape[0], flux_array.shape[1]])

  for n in range(0, entries):
    this_e = flux_array[n, 0]
    this_z = flux_array[n, 1]

    # Calculate tau survival and transition probability densities.
    tau_surv = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                             op=oscillation_parameters(delta=params[6]), nui='tau', nuf='tau')
    e_tau_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                op=oscillation_parameters(delta=params[6]), nui='e', nuf='tau')
    mu_tau_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                 op=oscillation_parameters(delta=params[6]), nui='mu',nuf='tau')
    flux_mod[n, 4] = (tau_surv * flux_array[n, 4]) \
                      + (e_tau_trans * flux_array[n, 2]) \
                      + (mu_tau_trans * flux_array[n, 3])

    mu_surv = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                            op=oscillation_parameters(delta=params[6]), nui='mu', nuf='mu')
    e_mu_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                               op=oscillation_parameters(delta=params[6]), nui='e', nuf='mu')
    tau_mu_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                 op=oscillation_parameters(delta=params[6]), nui='tau', nuf='mu')
    flux_mod[n, 3] = (mu_surv * flux_array[n, 3]) \
                      + (e_mu_trans * flux_array[n, 2]) \
                      + (tau_mu_trans * flux_array[n, 4])

    e_surv = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                           op=oscillation_parameters(delta=params[6]), nui='e', nuf='e')
    mu_e_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                               op=oscillation_parameters(delta=params[6]), nui='mu', nuf='e')
    tau_e_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                op=oscillation_parameters(delta=params[6]), nui='tau', nuf='e')
    flux_mod[n, 2] = (e_surv * flux_array[n, 2]) \
                      + (mu_e_trans * flux_array[n, 3]) \
                      + (tau_e_trans * flux_array[n, 4])

    atau_surv = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                              op=oscillation_parameters(delta=params[6]), nui='taubar', nuf='taubar')
    ae_atau_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                  op=oscillation_parameters(delta=params[6]), nui='ebar', nuf='taubar')
    amu_atau_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                   op=oscillation_parameters(delta=params[6]), nui='mubar',nuf='taubar')
    flux_mod[n, 7] = (atau_surv * flux_array[n, 7]) \
                      + (ae_atau_trans * flux_array[n, 5]) \
                      + (amu_atau_trans * flux_array[n, 6])

    amu_surv = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                             op=oscillation_parameters(delta=params[6]), nui='mubar', nuf='mubar')
    ae_amu_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                 op=oscillation_parameters(delta=params[6]), nui='ebar', nuf='mubar')
    atau_amu_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                   op=oscillation_parameters(delta=params[6]), nui='taubar', nuf='mubar')
    flux_mod[n, 6] = (amu_surv * flux_array[n, 6]) \
                      + (ae_amu_trans * flux_array[n, 5]) \
                      + (atau_amu_trans * flux_array[n, 7])

    ae_surv = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                            op=oscillation_parameters(delta=params[6]), nui='ebar', nuf='ebar')
    amu_ae_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                 op=oscillation_parameters(delta=params[6]), nui='mubar', nuf='ebar')
    atau_ae_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                  op=oscillation_parameters(delta=params[6]), nui='taubar', nuf='ebar')
    flux_mod[n, 5] = (ae_surv * flux_array[n, 5]) \
                      + (amu_ae_trans * flux_array[n, 6]) \
                      + (atau_ae_trans * flux_array[n, 7])


  flux_mod[:,0] = flux_array[:,0]
  flux_mod[:,1] = flux_array[:,1]
  return flux_mod



# Take in an nsi and return # of events integrated over energy and zenith
def EventsGenerator(nsi_array, df, expo):
  det = Detector("future-xe")

  nsi = NSIparameters(0)
  #nsi.epe = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2], 'em': nsi_array[3], 'et': nsi_array[4],
   #          'mt': nsi_array[5]}
  nsi.epu = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2], 'em': nsi_array[3], 'et': nsi_array[4],
             'mt': nsi_array[5]}
  nsi.epd = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2], 'em': nsi_array[3], 'et': nsi_array[4],
             'mt': nsi_array[5]}

  #energy_arr = np.linspace(e_low, e_high, nsteps_e)
  nsteps_e = 9
  energy_arr = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04]
  zenith_arr = np.unique(df[:, 1])

  observed_events = np.zeros(nsteps_e - 1)
  this_obs = 0
  flav_arr = np.array(["e", "mu", "tau", "ebar", "mubar", "taubar"])
  for i in range(0, zenith_arr.shape[0]):  # use .shape here to get the whole zenith array, regardless of experiment
    zenith_bin = np.delete(df[df[:,1] == zenith_arr[i]], [1], axis=1)

    e_a = energy_arr[0]

    for j in range (1, nsteps_e):
      e_b = energy_arr[j]

      for f in range(0, flav_arr.shape[0]):
        observed_events[this_obs] += binned_events_nucleus(e_a, e_b, expo, det, Flux(zenith_bin),
                                                           nsip=nsi, flavor=str(flav_arr[f]),
                                                           op=oscillation_parameters(delta=nsi_array[6]))
      #pois_error = np.sqrt(observed_events[this_obs])
      #smear = np.random.random_integers(-0.5 * pois_error, 0.5 * pois_error)
      #observed_events[this_obs] = np.round(observed_events[this_obs]) + smear
      #if observed_events[this_obs] < 0:
      #  observed_events[this_obs] = 0
      this_obs += 1

      # Iterate left edge
      e_a = e_b

  return observed_events




# Map the interval [0,1] to the interval [eps_min, eps_max] as a flat prior for each NSI parameter.
# Constraints from 1905.05203
def FlatPrior(cube, ndim, nparams):
  cube[0] = 1 * (2 * cube[0] - 1)  # eps_ee
  cube[1] = 0.5 * (2 * cube[1] - 1)  # eps_mumu
  cube[2] = 0.5 * (2 * cube[2] - 1)  # eps_tautau
  cube[3] = 0.12 * (2 * cube[3] - 1)  # eps_emu
  cube[4] = 0.3 * (2 * cube[4] - 1)  # eps_etau
  cube[5] = 0.028 * (2 * cube[5] - 1)  # eps_mutau
  cube[6] = 2 * pi * cube[6]  # delta_CP




def main():
  # Set the exposure (Future Xe):
  kTon = 0.1
  years = 20
  days_per_year = 365
  kg_per_kton = 1000000
  negative_zeniths = False

  exposure = years * days_per_year * kTon * kg_per_kton

  # Load the flux file and apply oscillations.
  # We use no nsi for oscillations since the energy is low enough to have no effect from nsi
  atmos = RebinAtmosphericFlux("pyCEvNS/data/atmos_extrapolated.txt", 40, negative_zeniths)
  flux_mod = ModulateFlux([0,0,0,0,0,0,1.5*np.pi], atmos)
  flux_mod = RebinAtmosphericFlux(flux_mod, 1, negative_zeniths)

  # 2D chisquare function
  n_sm = EventsGenerator([0, 0, 0, 0, 0, 0, 1.5*pi], flux_mod, exposure)
  width = np.sqrt(n_sm)
  def LogLikelihood(cube, D, N, df=flux_mod, expo=exposure):
    n_signal = EventsGenerator(cube, df, expo)
    likelihood = np.zeros(n_signal.shape[0])
    for i in range(n_signal.shape[0]):
      if width[i] == 0:
        continue
      likelihood[i] = -0.5 * np.log(2 * pi * width[i] ** 2) - 0.5 * ((n_signal[i] - n_sm[i]) / width[i]) ** 2
    return np.sum(likelihood)



  # Prepare some sample event rate plots.
  plot = False
  if plot:
    dcp = 0
    e_bins = [0.0055, 0.0065, 0.0075, 0.0085, 0.0095, 0.015, 0.025, 0.035]
    print(n_sm)
    nsi1 = EventsGenerator([0.18776773994719264, 0.06045789422901715, 0, 0, 0, 0, dcp], flux_mod, exposure)
    print(nsi1)
    nsi2 = EventsGenerator([0.1, 0, 0, -0.0207, 0, 0, dcp], flux_mod, exposure)
    print(nsi2)
    plt.plot(e_bins, n_sm, "-o", label="SM")
    plt.plot(e_bins, nsi1, "-o",label=r"$\epsilon_{ee} = 0.147$, $\epsilon_{\mu\mu} = 0.06$")
    plt.plot(e_bins, nsi2, "-o",label=r"$\epsilon_{ee} = 0.1$, $\epsilon_{e\mu} = -0.0207$")
    plt.xlabel(r'$E_R$ [MeV]')
    plt.ylabel('Events')
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.title(r'Future-Xe Neutrino Counts, 400 Ton-Year Exposure')
    plt.savefig("xenon_rates_test_nsi_best_fits.png")
    plt.savefig("xenon_rates_test_nsi_best_fits.pdf")



  # Define model parameters
  parameters = ["eps_ee", "eps_mumu", "eps_tautau", "eps_emu", "eps_etau", "eps_mutau", "delta_cp"]
  n_params = len(parameters)

  file_string = "all_nsi_xenon_realistic"
  text_string = "nsi_multinest/" + file_string + "/" + file_string
  json_string = "nsi_multinest/" + file_string + "/params.json"



  # run Multinest
  pymultinest.run(LogLikelihood, FlatPrior, n_params, outputfiles_basename=text_string,resume=False, verbose=True,
                  n_live_points=4000, evidence_tolerance=0.5, sampling_efficiency=0.3)
  json.dump(parameters, open(json_string, 'w'))  # save parameter names
  print("Saving to: \n" + text_string)



if __name__ == "__main__":
  main()
