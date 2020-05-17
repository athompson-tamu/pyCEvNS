import sys
sys.path.append("../")

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

from pyCEvNS.axion import primakoff_scattering_xs, IsotropicAxionFromPrimakoff, primakoff_prod_quant, primakoff_production_cdf


isoprim = IsotropicAxionFromPrimakoff(photon_rates=[[1,1]], axion_mass=0.1, axion_coupling=2.7e-8)

print(r"testing a + Z $\to \gamma$ + Z")
mass_array = np.linspace(0, 1.2, 100)
xs_array = np.zeros_like(mass_array)
for i, m in enumerate(mass_array):
    xs_array[i] = primakoff_scattering_xs(1.115, 32, ma=m, g=2.7e-8)


mev2_to_barn = 0.00257
plt.plot(mass_array, 1e-24 * xs_array/mev2_to_barn)
plt.show()


print("testing primakoff production CDF")
theta_list = np.linspace(0, pi, 1000)
cdf_list = np.empty_like(theta_list)
for i, th in enumerate(theta_list):
    cdf_list[i] = primakoff_production_cdf(th, 20, 23, 0.1)

plt.plot(theta_list, cdf_list)
plt.show()


print(r"testing primakoff ALP production angular distribution")
u = np.random.random(1000)
thetas = primakoff_prod_quant(u, 20, 32, 0.1)

plt.hist(thetas, bins=20)
plt.show()

