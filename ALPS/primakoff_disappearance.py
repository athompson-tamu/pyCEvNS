# Estimate sensitivity for a GANDHI style experiment to photon disappearance due to axion conversion
# in the Primakoff scattering channel in the detector.

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.special import gammaincc, gamma

import getopt, sys

from decimal import Decimal

# define constants
kAlpha = 1/137
mev_per_inv_cm = 1.2e-10



def run_statistics(b):
    def f1(c):
        return gammaincc(c+1,b) / gamma(c+1) - 3
    c_3s = fsolve(f1,np.sqrt(b))
    print("sqrt(b) = ", np.sqrt(b))
    print(c_3s)

    def f2(s):
        return 0.5 - (gammaincc(c_3s[0]+1,s+b) / gamma(c_3s[0]+1))
    return fsolve(f2,np.sqrt(b))



def PrimakoffXS(g, z):
    return kAlpha * (9/4) * (g * z)**2


# Length in cm, rho in cm^-3, xs in cm^2.
def SurvivalProbability(xs, rho, l):
    xs = Decimal(xs)
    rho = Decimal(rho)
    l = Decimal(l)
    return np.exp(-xs*rho*l)

def ConversionProbability(xs, rho, l):
    return 1 - SurvivalProbability(xs, rho, l)


def Significance(s, b):
    return s / np.sqrt(s + b)


def cone_length(m):
    return np.power((3 / (np.pi * 4.51e-6)) * m, 1/3)

def cone_mass(l):
    return 4.51e-6 * (np.pi / 3) * np.power(l,3)


def sphere_mass(l):
    return 4.51e-6 * (4 * np.pi / 3) * np.power(l, 3)

def sphere_length(m):
    return np.power((3 / (4 * np.pi * 4.51e-6)) * m, 1/3)


def main(decay_rate=1e23, det_z=55, det_rho=1.8e22, save_str="output"):
    decay_rate_cone = Decimal(decay_rate / 6.8)
    decay_rate = Decimal(decay_rate)
    csi_photon_cross = Decimal(1.3e-23)  # cm^2

    l_list = np.linspace(10,100,200)
    l_list_cone = np.linspace(10,cone_length(18.89),200)
    m_list_sphere = sphere_mass(l_list)
    m_list_cone = cone_mass(l_list_cone)

    def single_gamma(g, l):
        return SurvivalProbability(((mev_per_inv_cm)**2)*PrimakoffXS(g,det_z),det_rho, l) \
               * ConversionProbability(csi_photon_cross, det_rho, l)

    def miss_gamma(g, l):
        prim_xs = PrimakoffXS(g,det_z)
        return 1 - SurvivalProbability(((mev_per_inv_cm)**2)*prim_xs,det_rho, l) \
               * ConversionProbability(csi_photon_cross, det_rho, l)

    def miss_gamma_v2(g, l):
        prim_xs = Decimal(PrimakoffXS(g, det_z))
        br = prim_xs / (csi_photon_cross + prim_xs)
        return (ConversionProbability(csi_photon_cross, det_rho, l) * br)\
               + SurvivalProbability(csi_photon_cross, det_rho, l)

    def single_gamma_v2(g, l):
        return 1 - miss_gamma_v2(g, l)




    # Sphere limit
    sig_couplings = np.empty_like(l_list)
    for m in range(0,m_list_sphere.shape[0]):
        length = sphere_length(m_list_sphere[m])
        hi = 0
        lo = -20
        mid = (hi + lo) / 2
        while hi - lo > 0.000000001:
            mid = (hi + lo) / 2
            sb = decay_rate * miss_gamma_v2(10**mid, length)
            b = decay_rate * miss_gamma_v2(0, length)
            print("sb, b = ", sb, b)
            sig = (sb-b)/np.sqrt(sb)
            if sig < 2:
                lo = mid
            else:
                hi = mid
        sig_couplings[m] = 10**mid

    cone_couplings = np.empty_like(l_list)
    for m in range(0,m_list_cone.shape[0]):
        length = cone_length(m_list_cone[m])
        hi = 0
        lo = -20
        mid = (hi + lo) / 2
        while hi - lo > 0.000000001:
            mid = (hi + lo) / 2
            sb = decay_rate_cone * miss_gamma_v2(10**mid, length)
            b = decay_rate_cone * miss_gamma_v2(0, length)
            print("sb, b = ", sb, b)
            try:
                sig = (sb-b)/np.sqrt(sb)
            except ValueError:
                sig = 0
            if sig < 2:
                lo = mid
            else:
                hi = mid
        cone_couplings[m] = 10**mid

    e0_couplings = np.empty_like(l_list)
    for m in range(0, m_list_sphere.shape[0]):
        length = sphere_length(m_list_sphere[m])
        hi = 0
        lo = -20
        mid = (hi + lo) / 2
        while hi - lo > 0.000000001:
            mid = (hi + lo) / 2
            sb =  Decimal(1e-5) * decay_rate * (2*single_gamma_v2(10**mid,length) * miss_gamma_v2(10**mid,length)
                                      + (miss_gamma_v2(10**mid,length))**2)
            b = Decimal(1e-5) * decay_rate * (2*single_gamma_v2(0,length) * miss_gamma_v2(0,length)
                                      + (miss_gamma_v2(0,length))**2)
            #print("sb, b = ", sb-b, b)
            try:
                sig = (sb - b) / np.sqrt(sb)
            except ValueError:
                sig = 0
            if sig < 2:
                lo = mid
            else:
                hi = mid
        e0_couplings[m] = 10 ** mid

    # 1e23 decays
    # Sphere limit
    decay_rate = Decimal(1e16)
    sig_couplings_23 = np.empty_like(l_list)
    for m in range(0,m_list_sphere.shape[0]):
        length = sphere_length(m_list_sphere[m])
        hi = 0
        lo = -20
        mid = (hi + lo) / 2
        while hi - lo > 0.000000001:
            mid = (hi + lo) / 2
            sb = decay_rate * miss_gamma_v2(10**mid, length)
            b = decay_rate * miss_gamma_v2(0, length)
            print("sb, b = ", sb, b)
            sig = (sb-b)/np.sqrt(sb)
            if sig < 2:
                lo = mid
            else:
                hi = mid
        sig_couplings_23[m] = 10**mid

    cone_couplings_23 = np.empty_like(l_list)
    for m in range(0,m_list_cone.shape[0]):
        length = cone_length(m_list_cone[m])
        hi = 0
        lo = -20
        mid = (hi + lo) / 2
        while hi - lo > 0.000000001:
            mid = (hi + lo) / 2
            sb = Decimal(1e23 / 6.8) * miss_gamma_v2(10**mid, length)
            b = Decimal(1e23 / 6.8) * miss_gamma_v2(0, length)
            print("sb, b = ", sb, b)
            try:
                sig = (sb-b)/np.sqrt(sb)
            except ValueError:
                sig = 0
            if sig < 2:
                lo = mid
            else:
                hi = mid
        cone_couplings_23[m] = 10**mid

    e0_couplings_23 = np.empty_like(l_list)
    for m in range(0, m_list_sphere.shape[0]):
        length = sphere_length(m_list_sphere[m])
        hi = 0
        lo = -20
        mid = (hi + lo) / 2
        while hi - lo > 0.000000001:
            mid = (hi + lo) / 2
            sb =  Decimal(1e-5) * Decimal(1e23) * (2*single_gamma_v2(10**mid,length) * miss_gamma_v2(10**mid,length)
                                      + (miss_gamma_v2(10**mid,length))**2)
            b = Decimal(1e-5) * Decimal(1e23) * (2*single_gamma_v2(0,length) * miss_gamma_v2(0,length)
                                      + (miss_gamma_v2(0,length))**2)
            #print("sb, b = ", sb-b, b)
            try:
                sig = (sb - b) / np.sqrt(sb)
            except ValueError:
                sig = 0
            if sig < 2:
                lo = mid
            else:
                hi = mid
        e0_couplings_23[m] = 10 ** mid


    mev_per_gev = 1000
    plt.plot(m_list_sphere, sig_couplings * mev_per_gev, color='b', label=r"Spherical Geometry ($10^{9}$ Decays)")
    #plt.plot(m_list_sphere, e0_couplings * mev_per_gev, color='r',
     #        label="Spherical Geometry, (E0 Missing $2\gamma$, $10^{9}$ Decays)")
    #plt.plot(m_list_cone, cone_couplings * mev_per_gev, color='k', label="Conical Geometry (90 deg, $10^{9}$ Decays)")
    plt.plot(m_list_sphere, sig_couplings_23 * mev_per_gev, color='b',
             label="Spherical Geometry ($10^{16}$ Decays)", ls="dashed")
    #plt.plot(m_list_sphere, e0_couplings_23 * mev_per_gev, color='r',
      #       label="Spherical Geometry (E0 Missing $2\gamma$, $10^{16}$ Decays))",  ls="dashed")
    #plt.plot(m_list_cone, cone_couplings_23 * mev_per_gev, color='k',
     #        label="Conical Geometry (90 deg, $10^{16}$ Decays))",  ls="dashed")
    plt.yscale("log")
    plt.xscale("log")
    plt.title(r"CsI, Primakoff Conversion", loc="right")
    plt.legend(fontsize=7)
    plt.xlabel("Detector Mass (tons)")
    #plt.xlabel("Detector Radius [cm]")
    plt.ylabel(r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]")
    plt.savefig("limits_all_loglog_v2.png")
    plt.savefig("limits_all_loglog_v2.pdf")
    plt.clf()




if __name__ == "__main__":

    # read commandline arguments, first
    fullCmdArguments = sys.argv
    argumentList = fullCmdArguments[1:]

    unixOptions = "nd:zr:s"
    gnuOptions = ["decays=", "z=", "density=", "savedir="]

    try:
        arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)

    atomic_number = 55
    density = 1.8e22 # atoms / cm^3
    save_dir = "output"
    n_decays=1e9

    # evaluate given options
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-z", "--z"):
            atomic_number = currentValue
        elif currentArgument in ("-r", "--density"):
            density = currentValue
        elif currentArgument in ("-s", "--savedir"):
            save_dir = currentValue
        elif currentArgument in ("-nd", "--decays"):
            n_decays = currentValue

    main(decay_rate=n_decays, det_z=atomic_number, det_rho=density, save_str=save_dir)