import sys

from pyCEvNS.events import *
from pyCEvNS.flux import *
from pyCEvNS.fit import fit

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib.pylab import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# Set up neutrino PDFs and efficiencies.
prompt_pdf = np.genfromtxt('data/coherent/arrivalTimePDF_promptNeutrinos.txt', delimiter=',')
delayed_pdf = np.genfromtxt('data/coherent/arrivalTimePDF_delayedNeutrinos.txt', delimiter=',')
nin_pdf = np.genfromtxt('data/coherent/arrivalTimePDF_promptNeutrons.txt', delimiter=',')
ac_bon = np.genfromtxt('data/coherent/data_anticoincidence_beamOn.txt', delimiter=',')
c_bon = np.genfromtxt('data/coherent/data_coincidence_beamOn.txt', delimiter=',')
ac_boff = np.genfromtxt('data/coherent/data_anticoincidence_beamOff.txt', delimiter=',')
c_boff = np.genfromtxt('data/coherent/data_coincidence_beamOff.txt', delimiter=',')
nin = np.genfromtxt('data/coherent/promptPDF.txt', delimiter=',')


# Get Neutrino Events
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

def nin_time(t):
    if t < 0.25 or t > 11.75:
        return 0
    else:
        return nin_pdf[int((t-0.25)/0.5), 1]

def ffs(q):
    r = 5.5 * (10 ** -15) / meter_by_mev
    s = 0.9 * (10 ** -15) / meter_by_mev
    r0 = np.sqrt(5/3 * (r ** 2) - 5 * (s ** 2))
    return (3 * spherical_jn(1, q * r0) / (q * r0) * np.exp((-(q * s) ** 2) / 2)) ** 2
def efficiency(pe):
    a = 0.6655
    k = 0.4942
    x0 = 10.8507
    f = a / (1 + np.exp(-k * (pe - x0)))
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


# Define constants
pe_per_mev = 0.0878 * 13.348 * 1000
exp_csi = 4466
pim_rate = 0.0457


# Get Neutrino events (CsI)
# Set up energy and timing bins
# [16, 32] PE ~ [14, 26] keVnr
# [0, 52] PE ~ [0, 43] keVnr
# even PE: edges / odd PE: bin centers
hi_energy_cut = 52/pe_per_mev
lo_energy_cut = 0/pe_per_mev
hi_timing_cut = 11.75
lo_timing_cut = 0.0
energy_edges = np.arange(lo_energy_cut, hi_energy_cut, 2/pe_per_mev) # energy resolution ~2keV
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.arange(lo_timing_cut, hi_timing_cut, 0.5) # 0.5 mus time resolution
timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2

indx = []
# energy cut is 14keV ~ 16pe
for i in range(c_bon.shape[0]):
    if c_bon[i, 0] < lo_energy_cut*pe_per_mev + 1 \
       or c_bon[i, 0] >= hi_energy_cut*pe_per_mev - 1 \
       or c_bon[i, 1] >= hi_timing_cut-0.25:
        indx.append(i)
c_bon_meas = np.delete(c_bon, indx, axis=0)
ac_bon_meas = np.delete(ac_bon, indx, axis=0)

# Set the observed data = coincidence - anticoincidence (beam on)
n_meas = c_bon_meas.copy()

# Convert PE to MeV in the data array
n_meas[:,0] *= 1/pe_per_mev

# Set up signal, background, prompt and delayed arrays.
n_obs = c_bon_meas[:, 2]
n_bg = ac_bon_meas[:, 2]
flat_energies = n_meas[:,0]
flat_times = n_meas[:,1]
n_prompt = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_delayed = np.zeros(energy_bins.shape[0] * len(timing_bins))
n_nin = np.zeros(energy_bins.shape[0] * len(timing_bins))


# Get the theoretical prediction for the neutrino events.
flux_factory = NeutrinoFluxFactory()
prompt_flux = flux_factory.get('coherent_prompt')
delayed_flux = flux_factory.get('coherent_delayed')
nsi = NSIparameters(0)
nu_gen = NeutrinoNucleusElasticVector(nsi)
flat_index = 0
print("Generating neutrino spectrum...")
for i in range(0, energy_bins.shape[0]):
    print("E = ", energy_bins[i])
    for j in range(0, timing_bins.shape[0]):
        e_a = energy_edges[i]
        e_b = energy_edges[i + 1]
        pe = energy_bins[i] * pe_per_mev
        t = timing_bins[j]
        n_prompt[flat_index] = nu_gen.events(e_a, e_b, 'mu',prompt_flux, Detector('csi'), exp_csi) \
                               * prompt_time(t) * efficiency(pe)
        n_delayed[flat_index] = (nu_gen.events(e_a, e_b, 'e', delayed_flux, Detector('csi'), exp_csi)
                                 + nu_gen.events(e_a, e_b, 'mubar', delayed_flux, Detector('csi'), exp_csi)) \
                                * delayed_time(t) * efficiency(pe)
        n_nin[i] = (efficiency(pe-0.5) * nin[int(pe-1), 1]+ efficiency(pe+0.5) * nin[int(pe), 1]) * nin_time(t)
        flat_index += 1

n_nu = n_prompt+n_delayed


# Get DM events.
brem_photons = np.genfromtxt("data/coherent/brem.txt")  # binned photon spectrum from
Pi0Info = np.genfromtxt("data/coherent/Pi0_Info.txt")
pion_energy = Pi0Info[:,4] - massofpi0
pion_azimuth = np.arccos(Pi0Info[:,3] / np.sqrt(Pi0Info[:,1]**2 + Pi0Info[:,2]**2 + Pi0Info[:,3]**2))
pion_cos = np.cos(np.pi/180 * Pi0Info[:,0])
pion_flux = np.array([pion_azimuth, pion_cos, pion_energy])
pion_flux = pion_flux.transpose()

# Initialize classes.
m_dp = 75
m_chi = 25
tau = 0.001
dm_gen = DmEventsGen(dark_photon_mass=m_dp, dark_matter_mass=m_chi,
                         life_time=0.001, expo=exp_csi, detector_type='csi')
brem_flux = DMFluxIsoPhoton(brem_photons, dark_photon_mass=m_dp, coupling=1,
                            dark_matter_mass=m_chi, pot_sample=1e5,
                            sampling_size=1000, detector_distance=19.3,
                            brem_suppress=True, verbose=False)
pim_flux = DMFluxFromPiMinusAbsorption(dark_photon_mass=m_dp, coupling_quark=1,
                                        dark_matter_mass=m_chi,
                                        pion_rate=pim_rate, detector_distance=19.3)
pi0_flux = DMFluxFromPi0Decay(pi0_distribution=pion_flux,
                              dark_photon_mass=m_dp, coupling_quark=1,
                              dark_matter_mass=m_chi, detector_distance=19.3)
def GetDMEvents(cube):
    g = 10**cube[1]
    m_med = 10**cube[0]

    dm_gen.dp_mass = m_med
    dm_gen.dm_mass = m_med/3

    brem_flux.dp_m = m_med
    pim_flux.dp_m = m_med
    pi0_flux.dp_m = m_med
    brem_flux.dm_m = m_med/3
    pim_flux.dm_m = m_med/3
    pi0_flux.dm_m = m_med/3

    # Resim all fluxes
    brem_flux.simulate()
    pim_flux.simulate()
    pi0_flux.simulate()

    dm_gen.fx = brem_flux
    brem_events = dm_gen.events(m_med, g, energy_edges, timing_edges,
                                channel="nucleus")[0]

    dm_gen.fx = pim_flux
    pim_events = dm_gen.events(m_med, g, energy_edges, timing_edges,
                               channel="nucleus")[0]

    dm_gen.fx = pi0_flux
    pi0_events = dm_gen.events(m_med, g, energy_edges, timing_edges,
                               channel="nucleus")[0]
    return pim_events + pi0_events + brem_events + n_nu  # signal + neutrino model


def prior(cube):
    cube[0] = cube[0] * 3
    cube[1] = cube[1]*(-6)-4


def main():
    fit(GetDMEvents, 2, n_obs, n_bg, 0.28, prior, './multinest/singlemed_without_cuts/singlemed_without_cuts', n_live_points=9000)


if __name__=="__main__":
    main()
