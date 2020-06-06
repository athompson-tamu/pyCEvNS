import sys
import json
import numpy as np
from numpy import log, exp, pi, zeros, genfromtxt
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter, generic_filter1d
from scipy.special import erfinv
from scipy.special import xlogy, gammaln
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline


import pymultinest
import mpi4py

from pyCEvNS.events import*
from pyCEvNS.oscillation import*

from borexFits import BorexFit


def Bkg(name, bins):
  bin_arr = bins
  bkg = BorexFit(name)
  obs = np.zeros(bin_arr.shape[0] - 1)
  left_edge = bin_arr[0]
  for i in range(1, bin_arr.shape[0]):
    right_edge = bin_arr[i]
    obs[i - 1] = bkg.events(left_edge, right_edge)
    left_edge = right_edge
  return obs


def IntegrateSpline(func, bins, cutoff):
  bin_arr = bins
  obs = np.zeros(bin_arr.shape[0] - 1)
  left_edge = bin_arr[0]
  for i in range(1, bin_arr.shape[0]):
    right_edge = bin_arr[i]
    if right_edge > cutoff[1]:
      obs[i-1] = 0
      continue
    elif left_edge < cutoff[0]:
      obs[i-1] = 0
      continue

    obs[i - 1] = quad(func, left_edge, right_edge)[0]
    if obs[i-1] < 0:
      obs[i-1] = 0
    left_edge = right_edge
  return obs


def SmearKernel(x,y):  #ev: this energy. x: energy array. y: event array.
  out = np.zeros(x.shape[0])
  for i in range(0, x.shape[0]):
    sigma = 50 * np.sqrt(x[i]/1000)
    def smear(ec):
      return 1/(np.sqrt(2*pi) * sigma) * np.exp(-0.5*((ec-x[i])/sigma)**2) * np.interp(ec,x,y)
    out[i] = quad(smear, x[0],x[-1])[0]
  return out



def DetResponse(T, T_A):
  sigma = 0.1 * np.sqrt(T)
  return (1 / (np.sqrt(2*pi) * sigma)) * np.exp(-(T_A - T)**2 / (2 * sigma**2))


# Take in an nsi and return # of events integrated over energy and zenith
def EventsGenerator(params, expo, flux_factory, osc_factory, bins):
  det = Detector("borexino")
  nsteps_e = bins.shape[0] #81
  energy_arr = bins
  observed_events = np.zeros(energy_arr.shape[0] - 1)

  # Nuisance fluctuation on the solar flux.
  flux = flux_factory.get('solar', modulation=params[4])

  # Construct the NSI and flux-oscillation-detection pipeline.
  nsi = NSIparameters(0)
  #nsi.epel = {'ee': params[0], 'mm': params[2], 'tt': params[4],
   #           'em': params[6], 'et': params[8], 'mt': params[10]}
  #nsi.eper = {'ee': params[1], 'mm': params[3], 'tt': params[5],
   #           'em': params[7], 'et': params[9], 'mt': params[11]}
  nsi.epel = {'ee': 0, 'mm': 0, 'tt': params[0],
              'em': 0, 'et': params[1], 'mt': 0}
  nsi.eper = {'ee': 0, 'mm': 0, 'tt': params[2],
              'em': 0, 'et': params[3], 'mt': 0}
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
      observed_events[this_obs] += gen.events(e_a, e_b, str(flav_arr[f]), transformed_flux, det,
                                              exposure = None, target_exposure=expo, smearing=None)
    # Iterate left edge
    this_obs += 1
    e_a = e_b

  return observed_events



def Prior4D(cube, N, D):  # for comparison with 1905.03512
  cube[0] = 0.25 * (2 * cube[0] - 1)  # eps_ee L
  cube[1] = 0.25 * (2 * cube[1] - 1)  # eps_ee R
  cube[2] = (2 * cube[2] - 1)  # eps_tautau L
  cube[3] = (2 * cube[3] - 1)  # eps_tautau R

def Prior1D(cube, N, D):  # for comparison with 1905.03512
  cube[0] = (2 * cube[0] - 1)  # eps_ee L
  cube[1] = 1 + 0.06 * np.sqrt(2) * erfinv(2 * cube[1] - 1)  # Be7 flux normalization
  cube[2] = 1 + 0.06 * np.sqrt(2) * erfinv(2 * cube[2] - 1)  # 210 Bi flux normalization
  cube[3] = 1 + 0.06 * np.sqrt(2) * erfinv(2 * cube[3] - 1)  # 210 Po flux normalization

def FitPrior(cube, N, D):
  cube[0] = cube[0] * 2
  cube[1] = 30*cube[1] + 10
  cube[2] = 1 + 0.06 * np.sqrt(2) * erfinv(2 * cube[2] - 1)
  cube[3] = 2*cube[3] + 0.5
  cube[4] = 2*cube[4] + 0.5
  cube[5] = 2*cube[5] + 0.5
  cube[6] = 2*cube[6] + 0.5

def FancyPrior(cube, N, D):
  cube[0] = 1 + 0.06 * np.sqrt(2) * erfinv(2 * cube[0] - 1)  # Be7 flux normalization
  cube[1] = 0.5 * (2 * cube[1] - 1)  # eps_ee L
  cube[2] = 0.5 * (2 * cube[2] - 1)  # eps_ee R
  cube[3] = (2 * cube[3] - 1)  # eps_tautau L
  cube[4] = (2 * cube[4] - 1)  # eps_tautau R

def MinimalPrior(cube, ndim, nparams):
  cube[0] = 0.5 * (2 * cube[0] - 1)  # eps_ee L
  cube[1] = 0.5 * (2 * cube[1] - 1)  # eps_ee R
  cube[2] = 0.5 * (2 * cube[2] - 1)  # eps_etau L
  cube[3] = 0.5 * (2 * cube[3] - 1)  # eps_etau R
  cube[4] = 1 + 0.06 * np.sqrt(2) * erfinv(2 * cube[4] - 1)
  cube[5] = 2 * cube[5]
  cube[6] = 2 * cube[6]
  cube[7] = 4 * cube[7]
  cube[8] = 2 * cube[8]

# Map the interval [0,1] to the interval [eps_min, eps_max] as a flat prior for each NSI parameter.
def FlatPrior(cube, ndim, nparams):
  cube[0] = 0.2 * np.sqrt(2) * erfinv(2 * cube[0] - 1)
  cube[1] = 0.2 * np.sqrt(2) * erfinv(2 * cube[1] - 1)
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
  cube[12] = 1 + 0.06 * np.sqrt(2) * erfinv(2 * cube[12] - 1)
  cube[13] = 2 * cube[13]
  cube[14] = 2 * cube[14]
  cube[15] = 2 * cube[15]
  cube[16] = 2 * cube[16]
  #cube[12] = 1 + 0.06 * np.sqrt(2) * erfinv(2 * cube[12] - 1)



def main():
  # Read in Borexino data
  data = np.genfromtxt("pyCEvNS/data/Nature2018_Fig2a_DATA.txt")
  #po210 = np.loadtxt("pyCEvNS/data/borexino/polonium210.txt", delimiter=",")
  bi210 = np.loadtxt("pyCEvNS/data/borexino/bismuth210.txt", delimiter=",")
  kr85 = np.loadtxt("pyCEvNS/data/borexino/krypton85.txt", delimiter=",")
  c11 = np.loadtxt("pyCEvNS/data/borexino/carbon11.txt", delimiter=",")
  data = data[140:316,:]  # select around Be7 edge, 550 to 1000
  #data = data[0:500,:]  # select 0-1000

  # Set the exposure.
  borex_exp_100tday = 920.84  # exposure for phase 2 in 100t - days
  target_e = 3.307e31  # electrons per 100 ton
  exposure = 0.846*365 * target_e * borex_exp_100tday  # exposure for phase 2 in # of targets - days

  # Set up borexino features.
  borex_bins = data[:,2]
  borex_size = data.shape[0]
  obs = data[:,4] * borex_exp_100tday * data[:,1]
  err = data[:,5] * borex_exp_100tday * data[:,1]
  res = data[:,6] * err
  borex_fit = obs - (res)
  edges = np.zeros(data.shape[0] + 1)
  for b in range(0, borex_size):
    edges[b] = borex_bins[b] - 0.5 * data[b,3]
  edges[borex_size] = borex_bins[borex_size-1] + 0.5 * data[borex_size-1,3]

  # Set up factories.
  flux_factory = NeutrinoFluxFactory()
  osc_factory = OscillatorFactory()

  # Generate backgrounds
  kr_spline = UnivariateSpline(kr85[:, 0], kr85[:, 1])
  bi_spline = UnivariateSpline(bi210[:, 0], bi210[:, 1])
  c11_spline = UnivariateSpline(c11[:, 0], c11[:, 1])
  polonium = Bkg('po', edges) * borex_exp_100tday * data[:,1] / data[:,3]
  bismuth = IntegrateSpline(bi_spline, edges, (0,1250)) * borex_exp_100tday * data[:,1] / data[:,3]
  carbon11 = IntegrateSpline(c11_spline, edges,(0,3000)) * borex_exp_100tday * data[:,1] / data[:,3]
  krypton85 = IntegrateSpline(kr_spline, edges,(0,750)) * borex_exp_100tday * data[:, 1] / data[:, 3]
  backgrounds = 0.51*polonium + 0.81*bismuth + 1.55*carbon11 + 0.52*krypton85

  # Background subtracted mode?
  bkg_sub = 0


  # Set up standard model
  sm_params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  n_sm = EventsGenerator(sm_params, exposure, flux_factory, osc_factory, edges / 1000)
  #signal = SmearKernel(edges, n_sm) + (1 - bkg_sub) * backgrounds
  signal = gaussian_filter(n_sm, sigma=37.5, mode='nearest') + (1 - bkg_sub) * backgrounds


  # Subtract backgrounds
  borex_fit = borex_fit - bkg_sub*backgrounds
  obs = obs - bkg_sub*backgrounds

  # Ad hoc correction.
  corr = gaussian_filter(borex_fit,sigma=5) - signal
  signal += corr

  # Prepare some sample event rate plots.
  plot = False
  if plot == True:
    # Construct test fit
    print("plotting...")
    best_eeL_marginal = -0.35
    simple_fit = [0.3, 0, -0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    best_fit = [-0.485, -1.63, 0.31, 0, 0.3777, -0.131, 0, -0.21, -0.122, 0.409, 0, -0.3, 1.15]
    nsi = EventsGenerator(simple_fit, exposure, flux_factory, osc_factory, edges/1000)
    nsi = gaussian_filter(nsi, sigma=37.5, mode='nearest') + (1 - bkg_sub) * backgrounds + corr
    plt.errorbar(borex_bins, obs, yerr=err, color="k",
                 ls="None", marker='.', label='Borexino Phase II Data')
    #plt.plot(borex_bins, borex_fit, color="r", label="Borexino Fit")
    plt.plot(borex_bins, signal, label="SI", drawstyle='steps-mid',
             ls='solid', color='crimson', linewidth=2)
    plt.plot(borex_bins, nsi, label=r"NSI ($\epsilon^{e,L}_{ee}=0.3$, $\epsilon^{e,R}_{ee}=-0.4$)", drawstyle='steps-mid',
             ls='dashed', color='crimson', linewidth=2)
    plt.plot(borex_bins, polonium, label=r"$^{210}$Po", drawstyle='steps-mid', color='y')
    plt.plot(borex_bins, bismuth, label=r"$^{210}$Bi", drawstyle='steps-mid', color='b')
    plt.plot(borex_bins, carbon11, label=r"$^{11}$C", drawstyle='steps-mid', color='g')
    plt.plot(borex_bins, krypton85, label=r"$^{85}$Kr", drawstyle='steps-mid', color='m')
    plt.xlabel(r'$E_R$ [KeV]', fontsize=13)
    plt.ylabel('Events', fontsize=13)
    plt.yscale('log')
    #plt.xscale('log')
    plt.ylim((5e2,1e6))
    #plt.ylim((1e4,6e4))
    plt.xlim((550,1000))
    plt.legend(loc="upper right", framealpha=1.0)
    plt.tight_layout()
    plt.savefig("plots/rates/borexino/borexino_solar_spectrum.png")
    plt.savefig("plots/rates/borexino/borexino_solar_spectrum.pdf")

    plt.clf()
    plt.plot(borex_bins, res, ls="none",marker='o')
    plt.savefig("plots/rates/borexino/residuals.png")


  # Define likelihoods.
  def LogLikelihood(cube, N, D):
    backgrounds = cube[5] * polonium + cube[6] * bismuth + cube[7] * carbon11 + cube[8] * krypton85
    sig = EventsGenerator(cube, exposure, flux_factory, osc_factory, edges/1000)
    sig = gaussian_filter(sig, sigma=37.5, mode='nearest') + corr + backgrounds
    likelihood = np.zeros(sig.shape[0])

    for i in range(obs.shape[0]):
      if err[i] == 0:
        continue
      likelihood[i] = - 0.5 * np.log(2 * pi * err[i] ** 2) \
                      - 0.5 * ((sig[i] - obs[i]) / err[i]) ** 2
    return np.sum(likelihood)

  def FitLikelihood(cube, N, D):
    exp = exposure * cube[0]
    sig = EventsGenerator([cube[2], 0, 0, 0, 0], exp, flux_factory, osc_factory, edges/1000)
    backgrounds = cube[3] * polonium + cube[4] * bismuth + cube[5] * carbon11 + cube[6] * krypton85
    sig = gaussian_filter(sig, sigma=cube[1], mode='nearest') + backgrounds
    likelihood = np.zeros(sig.shape[0])

    for i in range(obs.shape[0]):
      if err[i] == 0:
        continue
      likelihood[i] = - 0.5 * np.log(2 * pi * err[i] ** 2) \
                      - 0.5 * ((sig[i] - borex_fit[i]) / err[i]) ** 2
    return np.sum(likelihood)

  # Define model parameters
  parameters = ["epsl_ee", "epsr_ee", "epsl_mumu", "epsr_mumu", "epsl_tautau", "epsr_tautau",
                "epsl_emu", "epsr_emu", "epsl_etau", "epsr_etau", "epsl_mutau", "epsr_mutau"]
  params_4d = ["be7flux", "epsl_ee", "epsr_ee","epsl_tautau", "epsr_tautau"]
  params_1d = ["be7flux","epsr_tt"]
  params_fit = ["epsl_ee", "epsr_ee", "epsl_mumu", "epsr_mumu", "epsl_tautau", "epsr_tautau",
                "epsl_emu", "epsr_emu", "epsl_etau", "epsr_etau", "epsl_mutau", "epsr_mutau",
                "be7", "bkg", "bkg", "bkg", "bkg"]
  params_minimal = ["epsl_tautau", "epsl_etau", "epsr_tautau", "epsr_etau",
                    "be7", "po", "bi", "c", "kr"]
  n_params = len(params_minimal)

  file_string = "borexino_minimal"
  text_string = "nsi_multinest/" + file_string + "/" + file_string
  json_string = "nsi_multinest/" + file_string + "/params.json"


  # run Multinest
  pymultinest.run(LogLikelihood, MinimalPrior, n_params, outputfiles_basename=text_string, resume=False, verbose=True,
                  n_live_points=5000, evidence_tolerance=0.5, sampling_efficiency=0.8)
  json.dump(params_minimal, open(json_string, 'w'))  # save parameter names
  print("Saving to: \n" + text_string)



if __name__ == "__main__":
  main()
