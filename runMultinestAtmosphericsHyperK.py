import sys
import json
import numpy as np
from numpy import log, exp, pi, zeros, genfromtxt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pymultinest
import mpi4py

from pyCEvNS.events import*
from pyCEvNS.oscillation import*



# Take in an nsi and return # of events integrated over energy and zenith
def EventsGenerator(nsi_array, expo, flux, osc_factory):
  det = Detector("hyper-k")
  zenith_arr = np.round(np.linspace(-0.975,-0.025,20), decimals=3)
  energy_arr = np.array([2985.00, 3350.00, 3758.00, 4217.00, 4732.00, 5309.00,
                         5957.00, 6683.00, 7499.00, 8414.00, 9441.00])
  obs = np.zeros((20,10))  # 18 energy bins, 20 zenith bins

  nsi = NSIparameters(0)
  nsi.epe = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2],
             'em': nsi_array[3], 'et': nsi_array[4], 'mt': nsi_array[5]}

  # Begin event loop.
  for i in range (0, zenith_arr.shape[0]):
    osc = osc_factory.get(oscillator_name='atmospheric', zenith=zenith_arr[i], nsi_parameter=nsi,
                          oscillation_parameter=OSCparameters(delta=nsi_array[6]))
    transformed_flux = osc.transform(flux[i])
    gen = NeutrinoNucleonCCQE("mu", transformed_flux)

    e_a = energy_arr[0]
    for j in range (1, energy_arr.shape[0]):
      e_b = energy_arr[j]
      obs[i][j-1] = 2*pi*0.05*np.round(gen.events(e_a, e_b, det, expo))

      # Iterate left edge
      e_a = e_b

  print(obs.shape[0], obs.shape[1])

  return obs




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
  kTon = 5600
  years = 10
  days_per_year = 365
  kg_per_kton = 1000000
  exposure = years * days_per_year * kTon * kg_per_kton

  # Set up factories.
  osc_factory = OscillatorFactory()
  flux_factory = NeutrinoFluxFactory()

  # Prepare flux.
  z_bins = np.round(np.linspace(-0.975, -0.025, 20), decimals=3)
  flux_list = []
  for z in range(0, z_bins.shape[0]):
    this_flux = flux_factory.get('atmospheric', zenith=z_bins[z])
    flux_list.append(this_flux)

  # Construct test data.
  sm_params = [0, 0, 0, 0, 0, 0, 1.5 * pi]
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
  do_plots = False
  if do_plots == True:
    e_bin_centers = np.array([3167.50, 3554.00, 3987.50, 4474.50, 5020.50,
                              5633.00, 6320.00, 7091.00, 7956.50, 8927.50])
    z_bins = np.linspace(-0.975, -0.025,20)
    X, Y = np.meshgrid(e_bin_centers, z_bins)
    nsi1 = EventsGenerator([0.5, 0, 0.4, 0.2*np.exp(-1j*pi/2), 0, 0, pi/3], exposure, flux_list, osc_factory)
    print("sum NSI: ", np.sum(nsi1))
    print("sum SI: ", np.sum(sm_events))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, nsi1, cmap=cm.viridis, alpha=0.7)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r"$E_\nu$")
    ax.set_ylabel(r"$\cos\theta_z$")
    ax.set_zlabel(r"$\nu_\mu$ Counts")
    ax.set_title(r"$\epsilon_{\alpha\beta} = 0$, $\delta_{CP} = 3\pi / 2$")

    plt.savefig("hyperk_atmos_surface_plot_nsi.png")
    plt.savefig("hyperk_atmos_surface_plot_nsi.pdf")
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    R, Z = np.meshgrid(e_bin_centers, np.arccos(z_bins))
    c1 = ax.pcolormesh(Z, R, sm_events)
    plt.pcolormesh(-Z, R, sm_events)
    fig.colorbar(c1, pad=0.2)
    #plt.plot(np.arccos(z_bins), e_bins, color='k', ls='none')
    plt.grid()
    plt.savefig("hyperk_polar_plot_si.pdf")
    plt.savefig("hyperk_polar_plot_si.png")


  # Define model parameters
  parameters = ["eps_ee", "eps_mumu", "eps_tautau", "eps_emu", "eps_etau", "eps_mutau", "delta_cp"]
  n_params = len(parameters)

  file_string = "all_nsi_hyperk_atmos_mu"
  text_string = "nsi_multinest/" + file_string + "/" + file_string
  json_string = "nsi_multinest/" + file_string + "/params.json"



  # run Multinest
  pymultinest.run(LogLikelihood, FlatPrior, n_params, outputfiles_basename=text_string,resume=False, verbose=True,
                  n_live_points=4000, evidence_tolerance=0.5, sampling_efficiency=0.3)
  json.dump(parameters, open(json_string, 'w'))  # save parameter names
  print("Saving to: \n" + text_string)



if __name__ == "__main__":
  main()

