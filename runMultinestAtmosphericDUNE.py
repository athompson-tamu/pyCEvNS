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

from pyCEvNS.events import*
from pyCEvNS.oscillation import*


n_z = 40
max_z = 0.975


# Take in an nsi and return # of events integrated over energy and zenith
def EventsGenerator(nsi_array, expo, flux, osc_factory):
  det = Detector("dune")
  zenith_arr = np.round(np.linspace(-0.975,max_z,n_z), decimals=3)
  energy_arr = np.array([106.00, 119.00, 133.00, 150.00, 168.00, 188.00, 211.00, 237.00, 266.00, 299.00,
                         335.00, 376.00, 422.00, 473.00, 531.00, 596.00, 668.00, 750.00, 841.00, 944.00])
                         #1059.00, 1189.00, 1334.00, 1496.00, 1679.00, 1884.00, 2113.00, 2371.00, 2661.00, 2985.00])
  obs = np.zeros((n_z,energy_arr.shape[0]-1))  # 18 energy bins, 20 zenith bins

  nsi = NSIparameters(0)
  nsi.epe = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2],
             'em': nsi_array[3] * exp(1j * nsi_array[6]), 'et': nsi_array[4] * exp(1j * nsi_array[7]),
             'mt': nsi_array[5] * exp(1j * nsi_array[8])}
  #nsi.epe = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2],
    #         'em': nsi_array[3] * exp(1j * nsi_array[6]), 'et': nsi_array[4],
     #        'mt': nsi_array[5]}


  # Begin event loop.
  for i in range (0, zenith_arr.shape[0]):
    osc = osc_factory.get(oscillator_name='atmospheric', zenith=zenith_arr[i], nsi_parameter=nsi,
                          oscillation_parameter=OSCparameters(delta=nsi_array[9]))
    transformed_flux = osc.transform(flux[i])
    gen = NeutrinoNucleonCCQE("mu", transformed_flux)

    e_a = energy_arr[0]
    for j in range (1, energy_arr.shape[0]):
      e_b = energy_arr[j]
      obs[i][j-1] = 2*pi*0.05*gen.events(e_a, e_b, det, expo)

      # Iterate left edge
      e_a = e_b

  return obs




# Map the interval [0,1] to the interval [eps_min, eps_max] as a flat prior for each NSI parameter.
def FlatPrior(cube, ndim, nparams):
  cube[0] = 0.5 * (2 * cube[0] - 1)  # eps_ee
  cube[1] = 0.5 * (2 * cube[1] - 1)  # eps_mumu
  cube[2] = 0.5 * (2 * cube[2] - 1)  # eps_tautau
  cube[3] = 0.1 * cube[3]  # eps_emu
  cube[4] = 0.1 * (2 * cube[4] - 1)  # eps_etau
  cube[5] = 0.1 * (2 * cube[5] - 1) # eps_mutau
  cube[6] = 0.5 * pi * (2 * cube[6] - 1)  # emu phase
  cube[7] = 0#0.5 * pi * (2 * cube[7] - 1)  # emu phase
  cube[8] = 0#0.5 * pi * (2 * cube[8] - 1)  # emu phase
  cube[9] = 2 * pi * cube[9]  # delta_CP




def main():
  # Set the exposure.
  kTon = 40
  years = 10
  days_per_year = 365
  kg_per_kton = 1000000
  exposure = years * days_per_year * kTon * kg_per_kton

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
  sm_params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.5 * pi]
  sm_events = EventsGenerator(sm_params, exposure, flux_list, osc_factory)
  null = sm_events.flatten()
  width = np.sqrt(null) + 1

  def LogLikelihood(cube, N, D):
    signal = (EventsGenerator(cube, exposure, flux_list, osc_factory)).flatten()
    likelihood = np.zeros(signal.shape[0])

    for i in range(signal.shape[0]):
      if width[i] == 0:
        continue
      likelihood[i] = -0.5 * np.log(2 * pi * width[i] ** 2) - 0.5 * ((signal[i] - null[i]) / width[i]) ** 2
    return np.sum(likelihood)



  # Prepare some sample event rate plots.
  do_plots = True
  if do_plots == True:
    e_bins = np.array([106.00, 119.00, 133.00, 150.00, 168.00, 188.00, 211.00, 237.00, 266.00, 299.00,
                       335.00, 376.00, 422.00, 473.00, 531.00, 596.00, 668.00, 750.00, 841.00, 944.00])
                       #1059.00, 1189.00, 1334.00, 1496.00, 1679.00, 1884.00, 2113.00, 2371.00, 2661.00, 2985.00])
    z_bins = np.linspace(-1, 1,n_z + 1)
    nsi1 = EventsGenerator([0, 0, 0, 0, 0, 0.05, 0, 0, pi/2, 1.5*pi], exposure, flux_list, osc_factory)
    print("sum NSI: ", np.sum(nsi1))
    print("sum SI: ", np.sum(sm_events))
    plt.rcParams.update({'font.size': 7})
    cmap = plt.get_cmap('inferno')


    fig = plt.figure(figsize=(8,3))
    fig.subplots_adjust(hspace=0.1, wspace=0.45)
    gs = gridspec.GridSpec(1, 3)
    ticks = np.linspace(0,60,6)

    # Plot SM
    ax = fig.add_subplot(gs[0, 0], polar=True)
    ax.set_theta_zero_location('N')
    plt.rgrids((200, 400, 600, 800), color='w')
    R, Z = np.meshgrid(e_bins, np.arccos(z_bins))
    c1 = ax.pcolormesh(Z, R, sm_events, cmap=cmap)
    plt.pcolormesh(-Z, R, sm_events, cmap=cmap)
    fig.colorbar(c1, orientation="horizontal", pad=0.12, ticks=ticks)
    plt.grid()

    # Plot NSI
    ax = fig.add_subplot(gs[0, 1], polar=True)
    ax.set_theta_zero_location('N')
    plt.rgrids((200, 400, 600, 800), color='w')
    R, Z = np.meshgrid(e_bins, np.arccos(z_bins))
    c1 = ax.pcolormesh(Z, R, nsi1, cmap=cmap)
    plt.pcolormesh(-Z, R, nsi1, cmap=cmap)
    fig.colorbar(c1, orientation="horizontal", pad=0.12, ticks=ticks)
    plt.grid()

    # Plot Diff
    ticks = np.linspace(0, 40, 5)
    ax = fig.add_subplot(gs[0, 2], polar=True)
    ax.set_theta_zero_location('N')
    plt.thetagrids((0,45,90,135,180,225,270,315), color='k')
    plt.rgrids((200, 400, 600, 800), color='w')
    R, Z = np.meshgrid(e_bins, np.arccos(z_bins))
    c1 = ax.pcolormesh(Z, R, np.fabs(nsi1 - sm_events), cmap=cmap)
    plt.pcolormesh(-Z, R, np.fabs(nsi1 - sm_events), cmap=cmap)
    fig.colorbar(c1, orientation="horizontal", pad=0.12, ticks=ticks)
    plt.grid()

    plt.savefig("plots/rates/dune/png/polar_plot_dune_atmos_phase_test.png")
    #plt.savefig("plots/rates/dune/pdf/polar_plot_dune_atmos.pdf")


  # Define model parameters
  parameters = ["eps_ee", "eps_mumu", "eps_tautau", "eps_emu", "eps_etau", "eps_mutau",
                "phi_emu", "phi_etau", "phi_mutau", "delta_cp"]
  n_params = len(parameters)

  file_string = "all_single-complex_nsi_dune_atmos_mu"
  text_string = "nsi_multinest/" + file_string + "/" + file_string
  json_string = "nsi_multinest/" + file_string + "/params.json"



  # run Multinest
  #pymultinest.run(LogLikelihood, FlatPrior, n_params, outputfiles_basename=text_string,resume=False, verbose=True,
  #                n_live_points=4000, evidence_tolerance=0.5, sampling_efficiency=0.3)
  #json.dump(parameters, open(json_string, 'w'))  # save parameter names
  #print("Saving to: \n" + text_string)



if __name__ == "__main__":
  main()

