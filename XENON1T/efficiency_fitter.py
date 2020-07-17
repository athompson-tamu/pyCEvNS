import numpy as np

# get the efficiency data
eff_points = np.genfromtxt("data/XENON1T_eff.txt", delimiter=",")

def eff(er, a, k, x0):
    return a / (1 + np.exp(-k * (er - x0)))


from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

xdata = eff_points[:,0]
ydata = eff_points[:,1]

popt, pcov = curve_fit(eff, xdata, ydata)

print(popt)
smooth_x = np.linspace(0, 40, 1000)
plt.plot(smooth_x, eff(smooth_x, *popt), 'r--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.scatter(xdata, ydata, label="X1T Efficiency")
plt.legend()
plt.xlabel(r"$E_r$ (keV)")
plt.ylabel(r"$\epsilon$")
plt.xlim((0,15))
plt.show()
plt.close()