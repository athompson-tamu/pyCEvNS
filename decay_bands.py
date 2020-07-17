import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def BestDecayCoupling(ma, eg, l):
    hc = 6.58e-22 * 3e8
    return np.sqrt(64 * np.pi * eg * hc / l / ma**4)
 
 
# Read in data.
beam = np.genfromtxt('data/existing_limits/beam.txt')
eeinva = np.genfromtxt('data/existing_limits/eeinva.txt')
lep = np.genfromtxt('data/existing_limits/lep.txt')
lsw = np.genfromtxt('data/existing_limits/lsw.txt')
nomad = np.genfromtxt('data/existing_limits/nomad.txt')

# Astrophyiscal limits
cast = np.genfromtxt("data/existing_limits/cast.txt", delimiter=",")
hbstars = np.genfromtxt("data/existing_limits/hbstars.txt", delimiter=",")
sn1987a_upper = np.genfromtxt("data/existing_limits/sn1987a_upper.txt", delimiter=",")
sn1987a_lower = np.genfromtxt("data/existing_limits/sn1987a_lower.txt", delimiter=",")


# Get decay band to cross-check.
masses_array = np.logspace(-6, 4, 1000)
dune_hi = np.array([BestDecayCoupling(m, 80000, 200) for m in masses_array])
dune_lo = np.array([BestDecayCoupling(m, 500, 574) for m in masses_array])

# Reactors
reactors_hi = np.array([BestDecayCoupling(m, 5, 1.5) for m in masses_array])
reactors_lo = np.array([BestDecayCoupling(m, 0.01, 50) for m in masses_array])

# Solar Axions
solar_hi = np.array([BestDecayCoupling(m, 0.001, 151.95e9) for m in masses_array])
solar_lo = np.array([BestDecayCoupling(m, 1e-9, 151.95e10) for m in masses_array])


# Plot decay check
plt.fill_between(masses_array*1e6, dune_hi*1e3, y2=dune_lo,
                 color="red", alpha=0.9)
plt.fill_between(masses_array*1e6, reactors_hi*1e3, y2=reactors_lo,
                 color="cyan", alpha=0.9)
plt.fill_between(masses_array*1e6, solar_hi*1e3, y2=solar_lo,
                 color="orange", alpha=0.9)

 # Plot astrophysical limits
plt.fill(hbstars[:,0]*1e9, hbstars[:,1]*0.367e-3, label="HB Stars", color="green", alpha=0.3)
plt.fill(cast[:,0]*1e9, cast[:,1]*0.367e-3, label="CAST", color="orchid", alpha=0.3)
plt.fill_between(sn1987a_lower[:,0]*1e9, y1=sn1987a_lower[:,1]*0.367e-3, y2=sn1987a_upper[:,1]*0.367e-3,
                label="SN1987a", color="lightsteelblue", alpha=0.3)


# Plot lab limits
plt.fill(beam[:,0], beam[:,1], label='Beam Dump', color="b", alpha=0.7)
plt.fill(np.hstack((eeinva[:,0], np.min(eeinva[:,0]))), np.hstack((eeinva[:,1], np.max(eeinva[:,1]))),
        color="orange", label=r'$e^+e^-\rightarrow inv.+\gamma$', alpha=0.7)
plt.fill(lep[:,0], lep[:,1], label='LEP', color="green", alpha=0.7)
plt.fill(np.hstack((nomad[:,0], np.min(nomad[:,0]))), np.hstack((nomad[:,1], np.max(nomad[:,1]))),
        color="yellow", label='NOMAD', alpha=0.7)




plt.legend(loc="lower left", framealpha=1, ncol=2, fontsize=9)
plt.xscale('log')
plt.yscale('log')
plt.xlim((1e-3,1e10))
plt.ylim(1e-15,1e-1)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('$m_a$ [eV]', fontsize=15)
plt.ylabel('$g_{a\gamma\gamma}$ [GeV$^{-1}$]', fontsize=15)

plt.tick_params(axis='x', which='minor')
plt.tight_layout()

plt.show()