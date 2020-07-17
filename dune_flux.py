import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# constants
pot_per_year = 1.1e21
pot_sample = 10000
scale = 1/pot_sample


# extract data and split columns
df = pd.read_csv('data/dune/hepmcout.hepmc', sep=' ', header=None, names=list(range(17)))

# get photons
gammas = df.loc[(df[0] == 'P') & (df[2] == '22')]

# declare gamma kinematics
gam_e = gammas[6].to_numpy()
gam_px = gammas[3].to_numpy()
gam_py = gammas[4].to_numpy()
gam_pz = gammas[5].to_numpy()
gam_p = np.sqrt(gam_px**2 + gam_py**2 + gam_pz**2)
gam_theta_z = np.arccos(gam_pz / gam_p)

# get Pi0
pions = df.loc[(df[0] == 'P') & (df[2] == '111')]

# declare Pion kinematics
pi0_e = pions[6].to_numpy()
pi0_px = pions[3].to_numpy()
pi0_py = pions[4].to_numpy()
pi0_pz = pions[5].to_numpy()
pi0_v = np.array([pi0_px/pi0_e, pi0_py/pi0_e, pi0_pz/pi0_e])

# decay pi0 -> 2 photons
gamma_e = 0.001*massofpi0/2 * np.ones_like(pi0_e)
gamma_p = 0.001*massofpi0/2 * np.ones_like(pi0_e)
cs1 = np.random.uniform(-1, 1, pi0_e.shape[0])
phi1 = np.random.uniform(0, 2*np.pi, pi0_e.shape[0])
cs2 = -cs1
phi2 = np.pi + phi1

# Get the p4's in COM and boost to lab frame
zeros = np.zeros_like(gamma_p)
gamma1_p4 = np.array([zeros, zeros, zeros, zeros])
gamma2_p4 = np.array([zeros, zeros, zeros, zeros])
cos_12 = np.empty_like(zeros)
for i in range(0,gamma_e.shape[0]):
    gamma1_p4[:,i] = np.array([gamma_e[i], gamma_p[i]*np.sqrt(1-cs1[i]**2)*np.cos(phi1[i]), gamma_p[i]*np.sqrt(1-cs1[i]**2)*np.sin(phi1[i]), gamma_p[i]*cs1[i]])
    gamma2_p4[:,i] = np.array([gamma_e[i], gamma_p[i]*np.sqrt(1-cs2[i]**2)*np.cos(phi2[i]), gamma_p[i]*np.sqrt(1-cs2[i]**2)*np.sin(phi2[i]), gamma_p[i]*cs2[i]])
    gamma1_p4[:,i] = lorentz_boost(gamma1_p4[:,i], -pi0_v[:,i])
    gamma2_p4[:,i] = lorentz_boost(gamma2_p4[:,i], -pi0_v[:,i])
    cos_12[i] = np.sum(gamma1_p4[1:,i] * gamma2_p4[1:,i]) / (np.sum(gamma1_p4[1:,i]) * np.sum(gamma2_p4[1:,i]))



# Write the gamma info to txt
gamma_data = np.array([gam_e, gam_px, gam_py, gam_pz, gam_theta_z])
gamma_data = gamma_data.transpose()
#np.savetxt("data/dune/hepmc_gamma_flux_from_pi0.txt", gamma_data)

# get number of photons less than 1 degree.
print(gam_theta_z[180*gam_theta_z/np.pi < 1.1].shape[0], gam_theta_z.shape[0])


# plot photon angle spectra
bins = np.linspace(0, 100, 100)
density=True
plt.hist(180*gam_theta_z/np.pi, bins=bins, histtype='step', label=r"$\gamma$ (HepMC)", density=density)
#plt.yscale("log")
plt.xlabel(r"$\theta_z$ [deg]", fontsize=15)
plt.ylabel("a.u.", fontsize=15)
plt.title(r"Pythia8 $\gamma$ Spectrum", loc="right")
#plt.legend(framealpha=1.0, loc="upper right", fontsize=15)
plt.show()



# plot pi0 energy spectra
bins = np.linspace(0, 80, 120)
density=False
wgt_pi0 = np.ones_like(pi0_e) * scale
wgt_gam = np.ones_like(gam_e) * scale
plt.hist(gam_e,  weights=wgt_gam, bins=bins, histtype='step', density=density)
plt.xlabel(r"$E$ [GeV]", fontsize=15)
plt.ylabel("Counts/POT", fontsize=15)
plt.yscale('log')
plt.title(r"Pythia8 $\gamma$", loc="right")
plt.legend(framealpha=1.0, loc="upper right", fontsize=15)
plt.show()

# plot separation cosine spectra
bins = np.linspace(0, 180, 50)
density=True
plt.hist(180*np.arccos(cos_12)/np.pi, bins=bins, histtype='step', density=density)
plt.xlabel(r"$\theta_{12}$ [deg]", fontsize=15)
plt.ylabel("a.u.", fontsize=15)
plt.show()


# plot pz spectra
bins = np.linspace(0, 80, 120)
density=True
plt.hist(pi0_pz, bins=bins, histtype='step', label=r"$\pi^0$", density=density)
plt.hist(gamma1_p4[3,:], bins=bins, histtype='step', label=r"$\gamma_1$", density=density)
plt.hist(gamma2_p4[3,:], bins=bins, histtype='step', label=r"$\gamma_2$", density=density)
plt.xlabel(r"$p_z$ [GeV]", fontsize=15)
plt.ylabel("a.u.", fontsize=15)
plt.yscale('log')
plt.legend(framealpha=1.0, loc="upper right", fontsize=15)
plt.show()


# plot px spectra
bins = np.linspace(-5, 5, 100)
density=True
plt.hist(pi0_px, bins=bins, histtype='step', label=r"$\pi^0$", density=density)
plt.hist(gamma1_p4[1,:], bins=bins, histtype='step', label=r"$\gamma_1$", density=density)
plt.hist(gamma2_p4[1,:], bins=bins, histtype='step', label=r"$\gamma_2$", density=density)
plt.xlabel(r"$p_x$ [GeV]", fontsize=15)
plt.ylabel("a.u.", fontsize=15)
plt.yscale('log')
plt.legend(framealpha=1.0, loc="upper right", fontsize=15)
plt.show()


# plot pz spectra
bins = np.linspace(-5, 5, 100)
density=True
plt.hist(pi0_py, bins=bins, histtype='step', label=r"$\pi^0$", density=density)
plt.hist(gamma1_p4[2,:], bins=bins, histtype='step', label=r"$\gamma_1$", density=density)
plt.hist(gamma2_p4[2,:], bins=bins, histtype='step', label=r"$\gamma_2$", density=density)
plt.xlabel(r"$p_y$ [GeV]", fontsize=15)
plt.ylabel("a.u.", fontsize=15)
plt.yscale('log')
plt.legend(framealpha=1.0, loc="upper right", fontsize=15)
plt.show()
