
import matplotlib.pyplot as plt
import numpy as np

from pyCEvNS.boltzmann import *



# constants
mpl = 1.22e22
mp = mpl/np.sqrt(8*np.pi) # reduced planck mass
t0 = 2.75/1.1605e10
tss = 4/1.1605e10
h0 = 2.2e-18/c_light*meter_by_mev
rho0 = 3*h0**2*mp**2

def jzf(epsilon, ma, alphad=0.5, mf=me, mchi_ratio=3):
    def func(z):
        return sigmav(epsilon, ma, z, alphad, mf, mchi_ratio)
    return quad(func, 0, 1/20)[0]

def rho_tilde(t):
    return np.pi**2/30*geff(t)*t**4
def hgr(t):
    return np.sqrt(rho_tilde(t)/3)
def hh(t):
    return hgr(t)/mp


def ss(x, m):
    t = m/x
    return 2*np.pi**2/45*geffs(t)*t**3


s0 = ss(10/t0, 10)

def omegah2(epsilon, ma, alphad=0.5, mf=me, mchi_ratio=3):
    gs0 = 3.91
    gszf = 10.75
    rho0 = 1.66/(1.22e22)*(gs0/np.sqrt(gszf) * (2.35e-10)**3)/jzf(epsilon, ma, alphad, mf, mchi_ratio)
#     print(rho0)
    return 2*rho0/(8.0992e-47*1e12)


def to_omegah2(sol, ma, mchi_ratio):
    m = ma/mchi_ratio
    if not sol.success:
        raise Exception('soloution not success')
    return sol.y[0, -1]/ss(1, m)*hh(m)*s0/rho0*0.678**2*m


print("calculating electron contribution")
mtlist_n = np.logspace(np.log10(me*3+0.5), np.log10(500), 150)

etplist_n = np.zeros_like(mtlist_n)
for i in range(mtlist_n.shape[0]):
    print("m = ", mtlist_n[i])
    lo, hi = -9, 0
    while hi - lo > 0.01:
        omg = omegah2(10**((hi+lo)/2), mtlist_n[i], mchi_ratio=mtlist_n[i]/2)
        if omg > 0.1323:
            lo = (hi+lo)/2
        else:
            hi = (hi+lo)/2
    print(omg)
    etplist_n[i] = (hi+lo)/2



# write new limits to txt
relic_array = np.array([mtlist_n, (10**(etplist_n))**2])
relic_array = relic_array.transpose()
np.savetxt('data/relic/relic_dark_photon_Mchi2MeV_smooth.txt', relic_array, delimiter=",")


fig, ax = plt.subplots(figsize=(4*1.2, 3*1.2))
ax.plot(np.hstack((mtlist_n,)), (10**np.hstack((etplist_n,)))**2, label='Relic (recalculated)')
ax.set_ylabel(r"$Y$")
ax.set_xlabel(r"$m_V$")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()
plt.clf()
