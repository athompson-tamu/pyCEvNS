
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


mtlist_n = np.logspace(np.log10(me*3+0.5), np.log10(mmu*3), 20)
etplist_n = np.zeros_like(mtlist_n)
tmp = np.linspace(-7, -4)
for i in range(mtlist_n.shape[0]):
    lo, hi = -7, -3
    while hi - lo > 0.01:
        omg = omegah2(10**((hi+lo)/2), mtlist_n[i])
        if omg > 0.1323:
            lo = (hi+lo)/2
        else:
            hi = (hi+lo)/2
#     print(omg)
    etplist_n[i] = (hi+lo)/2
    
    
mtlistmu = np.logspace(np.log10(mmu*3), 3, 20)
etplistmu = np.zeros_like(mtlistmu)
tmp = np.linspace(-7, -4)
for i in range(mtlistmu.shape[0]):
    lo, hi = -7, -3
    while hi - lo > 0.01:
        omg = omegah2(10**((hi+lo)/2), mtlistmu[i], mf=np.array([me, mmu]))
        if omg > 0.1323:
            lo = (hi+lo)/2
        else:
            hi = (hi+lo)/2
#     print(omg)
    etplistmu[i] = (hi+lo)/2


# rescale epsilon^2 to Y
y_rescale = 0.5 / (3**4)

# old limit
relic = np.genfromtxt('pyCEvNS/data/dark_photon_limits/relic.txt', delimiter=",")

# write new limits to txt
relic_array = np.array([np.hstack((mtlist_n, mtlistmu)), (10**np.hstack((etplist_n, etplistmu)))**2])
relic_array = relic_array.transpose()
np.savetxt('data/relic/relic_density.txt', relic_array, delimiter=",")


fig, ax = plt.subplots(figsize=(4*1.2, 3*1.2))
ax.plot(np.hstack((mtlist_n, mtlistmu)), y_rescale*(10**np.hstack((etplist_n, etplistmu)))**2, label='Relic (recalculated)')
ax.plot(relic[:,0], y_rescale*relic[:,1], color="k", label="Relic")
ax.set_ylabel(r"$Y$")
ax.set_xlabel(r"$m_V$")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()
plt.clf()
