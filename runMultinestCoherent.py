from pyCEvNS.fit import *
from pyCEvNS.flux import *
from pyCEvNS.detectors import *
from pyCEvNS.events import *
from pyCEvNS.experiments import *
from pyCEvNS.helper import _gaussian, _poisson
from numpy import *
from scipy.special import erfinv
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

import mpi4py

import json

# Read in CsI data
prompt_pdf = genfromtxt('pyCEvNS/data/arrivalTimePDF_promptNeutrinos.txt', delimiter=',')
delayed_pdf = genfromtxt('pyCEvNS/data/arrivalTimePDF_delayedNeutrinos.txt', delimiter=',')

ac_bon = genfromtxt('pyCEvNS/data/data_anticoincidence_beamOn.txt', delimiter=',')
c_bon = genfromtxt('pyCEvNS/data/data_coincidence_beamOn.txt', delimiter=',')
ac_boff = genfromtxt('pyCEvNS/data/data_anticoincidence_beamOff.txt', delimiter=',')
c_boff = genfromtxt('pyCEvNS/data/data_coincidence_beamOff.txt', delimiter=',')



# PDFs
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

def efficiency_lar(er):
    a = 0.94617149
    k = 0.2231348
    x0 = 14.98477134
    f = a / (1 + np.exp(-k * (1000*er - x0)))
    return f


flux_factory = NeutrinoFluxFactory()
prompt_flux = flux_factory.get('coherent_prompt')
delayed_flux = flux_factory.get('coherent_delayed')
csi_detector = Detector('csi')
cenns10_detector = Detector('ar')


# Set constants
pe_per_mev = 0.0878 * 13.348 * 1000
exp_ar = 24.4*276*0.51  # kg days
exp_csi = 4466

# % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % v # % # % # % # % # % # % # % # % # % # % # % # % # % # % # %
# Get Neutrino events (CsI) ###########################################
det_csi = Detector('csi')
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

n_meas_csi = c_bon_meas[:,2]
n_bkg_csi = ac_bon_meas[:,2]
# % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % v # % # % # % # % # % # % # % # % # % # % # % # % # % # % # %


# % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % v # % # % # % # % # % # % # % # % # % # % # % # % # % # % # %
# Get Neutrino arrays (LAr) ###########################################
# Read in LAr data
cenns10_obs = np.genfromtxt("cenns10/cenns10_analysisA_data.txt")
cenns10_brn = np.genfromtxt("cenns10/cenns10_analysisA_BRN.txt")
cenns10_error = np.genfromtxt("cenns10/cenns10_errors.txt")


# Read in data
bkgpdf_lar = np.genfromtxt("cenns10/Data/bkgpdf.txt")
brnpdf_lar = np.genfromtxt("cenns10/Data/brnpdf.txt")
brndelayedpdf_lar = np.genfromtxt("cenns10/Data/delbrnpdf.txt")
cevnspdf_lar = np.genfromtxt("cenns10/Data/cevnspdf.txt")
obs_lar = np.genfromtxt("cenns10/Data/datanobkgsub.txt")

# Systematics
brnpdf_m1sigTiming = np.genfromtxt("cenns10/Data/SystErrors/brnpdf-1sigBRNTimingMean.txt")[:,3]
brnpdf_p1sigTiming = np.genfromtxt("cenns10/Data/SystErrors/brnpdf+1sigBRNTimingMean.txt")[:,3]
brnpdf_m1sigEnergy = np.genfromtxt("cenns10/Data/SystErrors/brnpdf-1sigEnergy.txt")[:,3]
brnpdf_p1sigEnergy = np.genfromtxt("cenns10/Data/SystErrors/brnpdf+1sigEnergy.txt")[:,3]
cevnspdf_m1sigF90 = np.genfromtxt("cenns10/Data/SystErrors/cevnspdf-1sigF90.txt")[:,3]
cevnspdf_p1sigF90 = np.genfromtxt("cenns10/Data/SystErrors/cevnspdf+1sigF90.txt")[:,3]
cevnspdfCEvNSTiming = np.genfromtxt("cenns10/Data/SystErrors/cevnspdfCEvNSTimingMeanSyst.txt")[:,3]
brnpdfBRNTimingWidth = np.genfromtxt("cenns10/Data/SystErrors/brnpdfBRNTimingWidthSyst.txt")[:,3]

sigmaBrnEPlus = brnpdf_p1sigEnergy - brnpdf_lar[:,3]
sigmaBrnEMinus = brnpdf_m1sigEnergy - brnpdf_lar[:,3]
skewsBrnE = abs(sigmaBrnEPlus) - abs(sigmaBrnEMinus)
snBrnE = skewnorm(skewsBrnE)

sigmaCEvNSF90Plus = cevnspdf_p1sigF90 - cevnspdf_lar[:,3]
sigmaCEvNSF90Minus = cevnspdf_m1sigF90 - cevnspdf_lar[:,3]
skewCEvNSF90 = abs(sigmaCEvNSF90Plus) - abs(sigmaCEvNSF90Minus)
snCEvNSF90 = skewnorm(skewCEvNSF90)

det_ar = Detector('ar', efficiency=efficiency_lar)

# Set up energy and timing bins
hi_energy_cut_lar = 0.346  # 0.346 MeV = 346 keV
lo_energy_cut_lar = 0.01  # 0.01 MeV = 10 keV
hi_timing_cut_lar = 5.0
lo_timing_cut_lar = 0.0
energy_edges_lar = np.arange(lo_energy_cut_lar, hi_energy_cut_lar + 0.028, 0.028) # energy resolution ~2keV
energy_bins_lar = (energy_edges_lar[:-1] + energy_edges_lar[1:]) / 2
timing_edges_lar = np.linspace(0.0, 5.0, 11)
timing_bins_lar = (timing_edges_lar[1:]+timing_edges_lar[:-1])/2
f90_edges = np.linspace(0.5, 0.9, 9)
f90_bins = (f90_edges[1:]+f90_edges[:-1])/2

# Set up CEvNS, BRN, and Observed arrays
n_prompt_lar = np.zeros(len(energy_bins_lar) * len(timing_bins_lar) * len(f90_bins))
n_delayed_lar = np.zeros(len(energy_bins_lar) * len(timing_bins_lar) * len(f90_bins))
n_bg_lar = brnpdf_lar[:,3] + brndelayedpdf_lar[:,3]
n_meas_lar = obs_lar[:,3] - bkgpdf_lar[:,3]
error_lar = np.sqrt(obs_lar[:,3])

flat_idx = 0
flat_energies = np.zeros_like(cevnspdf_lar[:,0])
flat_times = np.zeros_like(cevnspdf_lar[:,0])
flat_f90 = np.zeros_like(cevnspdf_lar[:,0])
for i in range(energy_bins_lar.shape[0]):
    for j in range(f90_bins.shape[0]):
        for k in range(0, timing_bins_lar.shape[0]):
            flat_energies[flat_idx] = energy_bins_lar[i]
            flat_times[flat_idx] = timing_bins_lar[k]
            flat_f90[flat_idx] = f90_bins[j]
            flat_idx += 1


def F90PDF(f90, MeVnr):
    cevns_pdf_f90_energy = np.histogram2d(cevnspdf_lar[:,0]/1000,cevnspdf_lar[:,1],weights=cevnspdf_lar[:,3],
                                          bins=[12,8],density=True)
    keV_idx = int(np.where(energy_bins_lar == MeVnr)[0])
    f90_idx = int(np.where(f90_bins == f90)[0])
    norm = np.sum(cevns_pdf_f90_energy[0][keV_idx])
    if norm == 0.0:
        return 0.0
    return cevns_pdf_f90_energy[0][keV_idx][f90_idx] / norm

# % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % # % v # % # % # % # % # % # % # % # % # % # % # % # % # % # % # %




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
    cube[10] = cube[10]  # systematics



def events_gen(cube):
    nsi = NSIparameters(0)
    nsi.epu = {'ee': cube[0], 'mm': cube[1], 'tt': 0.0,
               'em': cube[2], 'et': cube[3], 'mt': cube[4]}
    nsi.epd = {'ee': cube[5], 'mm': cube[6], 'tt': 0.0,
               'em': cube[7], 'et': cube[8], 'mt': cube[9]}
    nu_gen = NeutrinoNucleusElasticVector(nsi)

    flat_index = 0
    for i in range(0, energy_bins.shape[0]):
        e_a = energy_edges[i]
        e_b = energy_edges[i+1]
        pe = energy_bins[i] * pe_per_mev
        prompt_rate = nu_gen.events(e_a, e_b, 'mu',prompt_flux, det_csi, exp_csi)
        delayed_rate = (nu_gen.events(e_a, e_b, 'e', delayed_flux, det_csi, exp_csi) \
                      + nu_gen.events(e_a, e_b, 'mubar', delayed_flux, det_csi, exp_csi))
        for j in range(0, timing_bins.shape[0]):
            t = timing_bins[j]
            n_prompt_csi[flat_index] =  prompt_rate * prompt_time(t) * efficiency(pe)
            n_delayed_csi[flat_index] = delayed_rate * delayed_time(t) * efficiency(pe)
            flat_index += 1

    n_nu_csi = n_prompt_csi + n_delayed_csi

    flat_index = 0
    for i in range(0, energy_bins_lar.shape[0]):
        e_a = energy_edges_lar[i]
        e_b = energy_edges_lar[i + 1]
        pe = energy_bins_lar[i] * pe_per_mev
        prompt_rate = nu_gen.events(e_a, e_b, 'mu',prompt_flux, det_ar, exp_ar)
        delayed_rate = (nu_gen.events(e_a, e_b, 'e', delayed_flux, det_ar, exp_ar) \
                      + nu_gen.events(e_a, e_b, 'mubar', delayed_flux, det_ar, exp_ar))
        for k in range(0, f90_bins.shape[0]):
            for j in range(0, timing_bins_lar.shape[0]):
                t = timing_bins_lar[j]
                n_prompt_lar[flat_index] = prompt_rate * prompt_time(t) * F90PDF(f90_bins[k], energy_bins_lar[i])
                n_delayed_lar[flat_index] = delayed_rate * delayed_time(t) * F90PDF(f90_bins[k], energy_bins_lar[i])
                flat_index += 1

    n_nu_lar = (n_prompt_lar+n_delayed_lar) + sigmaCEvNSF90Plus*snCEvNSF90.ppf(cube[10])

    return n_nu_csi, n_nu_lar



def RunPlotting():
    # Plot Dark Matter against Neutrino Spectrum
    from matplotlib.cm import get_cmap
    cmap = get_cmap('tab20b')
    color_prompt= cmap(0.70)
    color_delayed = cmap(0.15)
    color_brn = cmap(0.55)

    n_nu = events_gen(np.zeros(11))[1]
    n_nsi = events_gen([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])[1]

    density = False
    kevnr = flat_energies*1000
    kevnr_bins = energy_bins_lar*1000
    kevnr_edges = energy_edges_lar*1000
    obs_kevnr = np.histogram(kevnr, weights=n_meas_lar, bins=kevnr_edges)[0]
    kevnr_errors = np.sqrt(np.histogram(kevnr, weights=error_lar**2, bins=kevnr_edges)[0])
    plt.errorbar(kevnr_bins, obs_kevnr, yerr=kevnr_errors, color="k",
                dash_capstyle="butt", capsize=4, fmt='o', ls="none", label="Analysis A Data")
    plt.hist([kevnr,kevnr], weights=[n_nu, n_bg_lar],
            bins=kevnr_edges, stacked=True, histtype='stepfilled', density=density,
            color=[color_prompt, color_brn], label=[r"$\nu_\mu$", "BRN"])
    plt.hist(kevnr, weights=n_nsi, bins=kevnr_edges, histtype='step', density=density,
            color='m', label=r'NSI, $\epsilon^{u,V}_{\mu\mu} = -0.05$')

    plt.title(r"CENNS-10 LAr, $t<5.0$ $\mu$s", loc="right", fontsize=15)
    plt.xlabel(r"$E_r$ [keV]", fontsize=20)
    plt.ylabel(r"Events", fontsize=20)
    #plt.xlim((kevnr_edges[0], kevnr_edges[-1]))
    #plt.ylim((0,250))
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot timing spectra
    obs_timing = np.histogram(flat_times, weights=n_meas_lar, bins=timing_edges_lar)[0]
    timing_errors = np.sqrt(np.histogram(flat_times, weights=error_lar**2, bins=timing_edges_lar)[0])
    plt.errorbar(timing_bins_lar, obs_timing, yerr=timing_errors, color="k",
                dash_capstyle="butt", capsize=4, fmt='o', ls="none", label="Analysis A Data")
    plt.hist([flat_times,flat_times], weights=[n_nu, n_bg_lar],
            bins=timing_edges_lar, stacked=True, histtype='stepfilled', density=density,
            color=[color_prompt, color_brn], label=[r"$\nu_\mu$", "BRN"])
    plt.hist(flat_times, weights=n_nsi, bins=timing_edges_lar, histtype='step', density=density,
            color='m', label=r'NSI, $\epsilon^{u,V}_{\mu\mu} = -0.05$')

    plt.title(r"CENNS-10 LAr, $t<5.0$ $\mu$s", loc="right", fontsize=15)
    plt.xlabel(r"$t$ [$\mu$s]", fontsize=20)
    plt.ylabel(r"Events", fontsize=20)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()
    plt.close()



if __name__ == '__main__':
    #RunPlotting()

    def loglike_csi(n_signal, n_obs, n_bg, sigma):
        likelihood = np.ones(n_obs.shape[0])
        for i in range(n_obs.shape[0]):
            n_bg_list = np.arange(max(0, int(n_bg[i] - 2 * np.sqrt(n_bg[i]))),
                                  max(10, int(n_bg[i] + 2 * np.sqrt(n_bg[i]))))
            for nbgi in n_bg_list:
                likelihood[i] += quad(lambda a: _poisson(n_obs[i], (1 + a) * n_signal[i] + nbgi) *
                                                _gaussian(a, 0, sigma), -3 * sigma, 3 * sigma)[0] * \
                                 _poisson(n_bg[i], nbgi)
        return np.sum(np.log(likelihood))
    
    def loglike(n_signal, n_obs, n_bg, err):
        err = err + 1
        likelihood = -0.5*log(2*pi*err**2 + 1) - 0.5*((n_signal + n_bg - n_obs) / err)
        return np.sum(likelihood)

    def combined_likelihood(cube, ndim, nparams):
        # Apply systematics on background
        n_bg_lar_syst = sigmaBrnEPlus*snBrnE.ppf(cube[10]) + n_bg_lar

        # Get signal
        n_signal_csi, n_signal_ar = events_gen(cube)
        return loglike_csi(n_signal=n_signal_csi, n_obs=n_meas_csi, n_bg=n_bkg_csi, sigma=0.28) \
               + loglike(n_signal=n_signal_ar, n_obs=n_meas_lar, n_bg=n_bg_lar_syst, err=error_lar)



    pymultinest.run(combined_likelihood, prior, 11,
                    outputfiles_basename="nsi_multinest/coherent_csi_ar_jhep/coherent_csi_ar_jhep",
                    resume=False, verbose=True, n_live_points=2000, evidence_tolerance=0.5,
                    sampling_efficiency=0.8)

    params = ["eps_u_ee", "eps_d_mumu", "eps_u_emu", "eps_u_etau", "eps_u_mutau",
              "eps_d_ee", "eps_d_mumu", "eps_d_emu", "eps_d_etau", "eps_d_mutau", "syst"]
    json_string = 'nsi_multinest/coherent_csi_ar_jhep/params.json'
    json.dump(params, open(json_string, 'w'))


