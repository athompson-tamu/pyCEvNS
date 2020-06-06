from pyCEvNS.fit import *
from pyCEvNS.flux import *
from pyCEvNS.detectors import *
from pyCEvNS.events import *
from pyCEvNS.experiments import *
from pyCEvNS.helper import _gaussian, _poisson
from numpy import *
import matplotlib.pyplot as plt

import mpi4py

import json

prompt_pdf = genfromtxt('pyCEvNS/data/arrivalTimePDF_promptNeutrinos.txt', delimiter=',')
delayed_pdf = genfromtxt('pyCEvNS/data/arrivalTimePDF_delayedNeutrinos.txt', delimiter=',')

ac_bon = genfromtxt('pyCEvNS/data/data_anticoincidence_beamOn.txt', delimiter=',')
c_bon = genfromtxt('pyCEvNS/data/data_coincidence_beamOn.txt', delimiter=',')
ac_boff = genfromtxt('pyCEvNS/data/data_anticoincidence_beamOff.txt', delimiter=',')
c_boff = genfromtxt('pyCEvNS/data/data_coincidence_beamOff.txt', delimiter=',')



def prompt_time(t):
    if t < 0.25 or t > 11.75:
        return 0
    else:
        return prompt_pdf[int((t-0.25)/0.5), 1]


def delayed_time(t):
    if t < 0.25 or t > 11.75:
        return 0
    else:
        return delayed_pdf[int((t-0.25)/0.5), 1]


def efficiency(pe):
    a = 0.6655
    k = 0.4942
    x0 = 10.8507
    f = a / (1 + exp(-k * (pe - x0)))
    if pe < 5:
        return 0
    if pe < 6:
        return 0.5 * f
    return f

def efficiency_lar(pe):
    a = 0.9
    k = 0.4942
    x0 = 10.8507
    f = a / (1 + np.exp(-k * (pe - x0)))
    return f



#prompt_flux = Flux('prompt')
#delayed_flux = Flux('delayed')
flux_factory = NeutrinoFluxFactory()
prompt_flux = flux_factory.get('coherent_prompt')
delayed_flux = flux_factory.get('coherent_delayed')
csi_detector = Detector('csi')
cenns10_detector = Detector('ar')


# Set constants
pe_per_mev = 0.0878 * 13.348 * 1000
exp_ar = 1.5 * 365 * 28
exp_csi = 4466


# Get Neutrino events (CsI) ###########################################
# Set up energy and timing bins
hi_energy_cut = 32/pe_per_mev #0.030  # mev
lo_energy_cut = 16/pe_per_mev #0.014  # mev
hi_timing_cut = 6.25
lo_timing_cut = 0.0
energy_edges = np.arange(lo_energy_cut, hi_energy_cut, 2/pe_per_mev)
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.arange(lo_timing_cut, hi_timing_cut, 0.5)
timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2

n_prompt_csi = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_delayed_csi = np.zeros(energy_bins.shape[0] * len(timing_bins))

indx = []
# Apply cuts.
for i in range(c_bon.shape[0]):
    if c_bon[i, 0] < lo_energy_cut*pe_per_mev+1 \
       or c_bon[i, 0] >= hi_energy_cut*pe_per_mev-1 \
       or c_bon[i, 1] >= hi_timing_cut - 0.25:
        indx.append(i)
c_bon_meas = np.delete(c_bon, indx, axis=0)
ac_bon_meas = np.delete(ac_bon, indx, axis=0)

#print(c_bon_meas[:,0]/pe_per_mev, c_bon_meas[:,1])
#print(energy_bins, timing_bins)

# Set the observed data = coincidence - anticoincidence (beam on)
n_meas_csi = c_bon_meas.copy()



# Get Neutrino arrays (LAr) ###########################################
# Set up energy and timing bins
hi_energy_cut = 0.120  # mev
lo_energy_cut = 0.02 # mev
hi_timing_cut = 6
lo_timing_cut = 0.0
energy_edges_lar = np.arange(lo_energy_cut, hi_energy_cut, 0.01) # energy resolution ~10keVnr
energy_bins_lar = (energy_edges_lar[:-1] + energy_edges_lar[1:]) / 2
timing_edges_lar = np.arange(lo_timing_cut, hi_timing_cut, 0.5) # 0.5 mus time resolution
timing_bins_lar = (timing_edges_lar[:-1] + timing_edges_lar[1:]) / 2

n_meas_lar = np.zeros((energy_bins_lar.shape[0] * len(timing_bins_lar), 2))
n_prompt_lar = np.zeros(energy_bins_lar.shape[0] * len(timing_bins_lar))
n_delayed_lar = np.zeros(energy_bins_lar.shape[0] * len(timing_bins_lar))
n_bg_lar = np.zeros(energy_bins_lar.shape[0] * len(timing_bins_lar))

flat_index = 0
for i in range(0, energy_bins_lar.shape[0]):
    for j in range(0, timing_bins_lar.shape[0]):
      n_meas_lar[flat_index, 0] = energy_bins_lar[i]
      n_meas_lar[flat_index, 1] = timing_bins_lar[j]
      flat_index += 1

for i in range(0, n_meas_lar.shape[0]):
    n_bg_lar[i] = (226 / (energy_bins.shape[0]) * prompt_time(n_meas_lar[i,1])) \
                   + (10 / (energy_bins.shape[0]) * delayed_time(n_meas_lar[i,1]))

flat_energies_lar = n_meas_lar[:,0]
flat_times_lar = n_meas_lar[:,1]




def prior(cube, n, d):
    cube[0] = (2 * cube[0] - 1)  # eps_u_ee
    cube[1] = (2 * cube[1] - 1)  # eps_u_mumu
    cube[2] = (2 * cube[2] - 1)  # eps_u_emu
    cube[3] = (2 * cube[3] - 1)  # eps_u_eta
    cube[4] = (2 * cube[4] - 1)  # eps_u_muta
    cube[5] = (2 * cube[5] - 1)  # eps_d_ee
    cube[6] = (2 * cube[6] - 1)  # eps_d_mumu
    cube[7] = (2 * cube[7] - 1)  # eps_d_emu
    cube[8] = (2 * cube[8] - 1)  # eps_d_eta
    cube[9] = (2 * cube[9] - 1)  # eps_d_muta



def events_gen(cube):
    nsi = NSIparameters(0)
    nsi.epu = {'ee': cube[0], 'mm': cube[1], 'tt': 0.0,
               'em': cube[2], 'et': cube[3], 'mt': cube[4]}
    nsi.epd = {'ee': cube[5], 'mm': cube[6], 'tt': 0.0,
               'em': cube[7], 'et': cube[8], 'mt': cube[9]}
    nu_gen = NeutrinoNucleusElasticVector(nsi)



    flat_index = 0
    for i in range(0, energy_bins.shape[0]):
        for j in range(0, timing_bins.shape[0]):
            e_a = energy_edges[i]
            e_b = energy_edges[i+1]
            pe = energy_bins[i] * pe_per_mev
            t = timing_bins[j]
            n_prompt_csi[flat_index] = nu_gen.events(e_a, e_b, 'mu',prompt_flux, Detector('csi'), exp_csi) \
                                          * prompt_time(t) * efficiency(pe)
            n_delayed_csi[flat_index] = (nu_gen.events(e_a, e_b, 'e', delayed_flux, Detector('csi'), exp_csi) \
                                         + nu_gen.events(e_a, e_b, 'mubar', delayed_flux, Detector('csi'), exp_csi)) \
                                         * delayed_time(t) * efficiency(pe)
            flat_index += 1

    n_nu_csi = n_prompt_csi + n_delayed_csi

    flat_index = 0
    for i in range(0, energy_bins_lar.shape[0]):
        for j in range(0, timing_bins_lar.shape[0]):
            e_a = energy_edges_lar[i]
            e_b = energy_edges_lar[i + 1]
            pe = energy_bins_lar[i] * pe_per_mev
            t = timing_bins_lar[j]
            n_prompt_lar[flat_index] = nu_gen.events(e_a, e_b, 'mu',prompt_flux, Detector('ar'), exp_ar) \
                                           * prompt_time(t) * efficiency_lar(pe) * 0.46 # ad hoc factor for greater dist
            n_delayed_lar[flat_index] = (nu_gen.events(e_a, e_b, 'e', delayed_flux, Detector('ar'), exp_ar) \
                                         + nu_gen.events(e_a, e_b, 'mubar', delayed_flux, Detector('ar'), exp_ar)) \
                                         * delayed_time(t) * efficiency_lar(pe) * 0.46 # ad hoc factor for greater dist
            #print(nu_gen.events(e_a, e_b, 'e', delayed_flux, Detector('ar'), exp_ar))
            #print("ndelayed ar = ", n_delayed_lar[flat_index])
            flat_index += 1

    n_nu_lar = n_prompt_lar+n_delayed_lar

    #print(np.sum(n_nu_csi), np.sum(n_nu_lar))
    return n_nu_csi, n_nu_lar



if __name__ == '__main__':


    lar_null = events_gen(np.zeros(12))[1]  # projected signal for LAr, 1.5 years.
    #print("lar null = ", np.sum(lar_null), "csi null = ", events_gen(np.zeros(12))[0])



    def loglike(n_signal, n_obs, n_bg, sigma):
        likelihood = np.zeros(n_obs.shape[0])
        for i in range(n_obs.shape[0]):
            n_bg_list = np.arange(max(0, int(n_bg[i] - 2 * np.sqrt(n_bg[i]))),
                                  max(10, int(n_bg[i] + 2 * np.sqrt(n_bg[i]))))
            for nbgi in n_bg_list:
                likelihood[i] += quad(lambda a: _poisson(n_obs[i], (1 + a) * n_signal[i] + nbgi) *
                                                _gaussian(a, 0, sigma), -3 * sigma, 3 * sigma)[0] * \
                                 _poisson(n_bg[i], nbgi)
        #print("likelihood = ",np.sum(np.log(likelihood)))
        return np.sum(np.log(likelihood))

    def combined_likelihood(cube, ndim, nparams):
        n_signal_csi, n_signal_ar = events_gen(cube)
        return loglike(n_signal=n_signal_csi, n_obs=c_bon_meas[:, 2],
                       n_bg=ac_bon_meas[:, 2], sigma=0.28)\
               + loglike(n_signal=n_signal_ar, n_obs=lar_null, n_bg=n_bg_lar, sigma=0.085)



    pymultinest.run(combined_likelihood, prior, 10,
                    outputfiles_basename="nsi_multinest/coherent_csi_ar_jhep/coherent_csi_ar_jhep",
                    resume=False, verbose=True, n_live_points=2000, evidence_tolerance=0.5,
                    sampling_efficiency=0.8)

    #fit(events_gen, 12, c_bon_meas[:, 2], ac_bon_meas[:, 2], 0.28, singlePrior, 
    # 'nsi_multinest/coherent_minimal/coherent_minimal')
    params = ["eps_u_ee", "eps_d_mumu", "eps_u_emu", "eps_u_etau", "eps_u_mutau",
              "eps_d_ee", "eps_d_mumu", "eps_d_emu", "eps_d_etau", "eps_d_mutau"]
    json_string = 'nsi_multinest/coherent_csi_ar_jhep/params.json'
    json.dump(params, open(json_string, 'w'))


