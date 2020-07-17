import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.stats import norm

ergs = np.arange(1.0,30.1,0.1)

bkg_data_fig4 = np.genfromtxt("reprod/xenon1t_bkg_model_fig4.dat")
total_bkg_signal_fig4 = interp1d(bkg_data_fig4[:,0], bkg_data_fig4[:,1],fill_value=0,bounds_error=False)
data_zenodo_v1 = np.genfromtxt("reprod/data_below_30kev.txt", delimiter=',')
eff_data = np.genfromtxt("reprod/efficiency.txt", delimiter=',')
eff_interp = interp1d(eff_data[:,0],eff_data[:,3],fill_value=0,bounds_error=False)

pe_data_new = np.genfromtxt("reprod/photoelectric_new.dat")
pe_interp_new = interp1d(np.log10(pe_data_new[:,0]),np.log10(pe_data_new[:,1]),fill_value=-np.inf,bounds_error=False)

redondo_data = np.genfromtxt("reprod/2013_redondo_all.dat")
redondo_interp = interp1d(redondo_data[:,0],redondo_data[:,1],fill_value=0,bounds_error=False)

model_inv_P_ABC_data = np.genfromtxt("reprod/200615118_ip_signal_abc.dat")
model_inv_P_P_data = np.genfromtxt("reprod/200615118_ip_signal_P.dat")
model_inv_P_Fe_data = np.genfromtxt("reprod/200615118_ip_signal_fe.dat")

exposure = 0.647309514
binsize = 1.0
m_el = 510.99895000 #keV
alpha_EM = 0.0072973525693
A_Xe = 131.
Z_Xe = 54.
# Here I add the factor of 0.5
r0 = 2.45e-10/1.97327e-10
invGeVincm = 1.97327e-14
E_57Fe = 14.4

def sigma_pe(erg):
    res = 0.0
    if ((erg >= 0.1) and (erg <= 100)):
        res = 1000.0*1.0e-24* 10**pe_interp_new(np.log10(erg))/(A_Xe*1.66053906660e-27)
    return res

def spectral_eff(erg):
    return eff_interp(erg)

def sigma_ae(erg, beta=1.0):
    return 3.0 * (erg/m_el)**2 * (5.0e-12)**2 * (1.0 - beta**(2./3.)/3.0) * sigma_pe(erg) / (16.0*np.pi*alpha_EM*beta)

# Here I define the inverse Primakoff contribution
def inverse_primakoff(k, gagamma=2.0e-10):
    rk2 = (k*r0)**2
    return 0.5*alpha_EM * (1000.0/(A_Xe*1.66053906660e-27)) * Z_Xe**2 * (gagamma * invGeVincm)**2 * ((2.0*rk2 + 1.0)*np.log1p(4.0*rk2)/(4.0*rk2) - 1.0)

# Below I define the solar axion fluxes
def primakoff(erg, gagamma=2.0e-10):
    return 6.0e30 * (365.0*24.0*60.0*60.0) * (gagamma)**2 * erg**2.481 * np.exp(-erg/1.205)

def fe_contrib(gaN=1.0e-6):
    return 4.56e23 * (365.0*24.0*60.0*60.0) * gaN**2

def abc(erg, gae=5.0e-12):
    return 1.0e19 * 365.0 * (gae/0.511e-10)**2 * redondo_interp(erg)

abc_peaks = (0.653029, 0.779074, 0.920547, 0.956836, 1.02042, 1.05343, 1.3497, 1.40807, 1.46949, 1.59487, 1.62314, 1.65075, 1.72461, 1.76286, 1.86037, 2.00007, 2.45281, 2.61233, 3.12669, 3.30616, 3.88237, 4.08163, 5.64394, 5.76064, 6.14217, 6.19863, 6.58874, 6.63942, 6.66482, 7.68441, 7.74104, 7.76785)

# https://arxiv.org/pdf/2003.03825.pdf
def energy_resolution(erg):
    return erg*(31.71/np.sqrt(erg) + 0.15)/100.0


def wrapper_P_2(x, x0):
    return primakoff(x, gagamma=3.0e-10)*inverse_primakoff(x, gagamma=3.0e-10)*norm.pdf(x, x0, energy_resolution(x0))

integrals_P_2 = [quad(wrapper_P_2, 0.0, np.inf, args=(x0), epsabs=0, epsrel=1.0e-8)[0] for x0 in ergs]

def wrapper_ABC_2(x, x0):
    return abc(x, gae=5.0e-12)*inverse_primakoff(x, gagamma=3.0e-10)*norm.pdf(x, x0, energy_resolution(x0))

integrals_ABC_2 = [quad(wrapper_ABC_2, 0.0, 40.0, args=(x0), epsabs=0, epsrel=1.0e-4, limit=500)[0] for x0 in ergs]

rate_P_2 = np.array([integrals_P_2[i]*spectral_eff(ergs[i]) for i in range(len(ergs))])
rate_ABC_2 = np.array([integrals_ABC_2[i]*spectral_eff(ergs[i]) for i in range(len(ergs))])
rate_Fe_2 = np.array([fe_contrib(gaN=5.0e-7)*inverse_primakoff(E_57Fe, gagamma=3.0e-10)*norm.pdf(ergs[i], E_57Fe, energy_resolution(E_57Fe))*spectral_eff(ergs[i]) for i in range(len(ergs))])


bkg_1 = total_bkg_signal_fig4(ergs)
spb_1 = bkg_1 + rate_P_2
spb_2 = bkg_1 + rate_ABC_2
spb_3 = bkg_1 + rate_Fe_2
bkg_at_sig = total_bkg_signal_fig4(data_zenodo_v1[:,0])

fig, ax = plt.subplots()

ax.errorbar(data_zenodo_v1[:,0], data_zenodo_v1[:,1], yerr=np.sqrt(data_zenodo_v1[:,1]/(exposure*binsize)), fmt='o', color='k')
ax.plot(ergs, spb_1, '-', lw=2, color='red')
ax.plot(ergs, spb_2, '-', lw=2, color='blue')
ax.plot(ergs, spb_3, '-', lw=2, color='green')
ax.plot(model_inv_P_P_data[:,0], model_inv_P_P_data[:,1], '--', color='k')
ax.plot(model_inv_P_ABC_data[:,0], model_inv_P_ABC_data[:,1], '--', color='k')
ax.plot(model_inv_P_Fe_data[:,0], model_inv_P_Fe_data[:,1], '--', color='k')

ax.set_xlim([0,30.0])
ax.set_ylim([0,120])
ax.set_ylabel(r'Relative events [1 / t yr keV]')
ax.set_xlabel(r'Energy [keV]')
plt.savefig('xenon1t_inverse_primakoff.pdf')
plt.show()
