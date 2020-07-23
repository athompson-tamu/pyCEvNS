import matplotlib.pyplot as plt
import numpy as np
from numpy import log10, log, exp, sqrt, pi

from scipy.special import gammaln

from matplotlib.cm import get_cmap
from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# Read in data
bkgpdf_data = np.genfromtxt("../Data/bkgpdf.txt")
brnpdf_data = np.genfromtxt("../Data/brnpdf.txt")
brndelayedpdf_data = np.genfromtxt("../Data/delbrnpdf.txt")
efficiency_data = np.genfromtxt("../Data/CENNS10AnlAEfficiency.txt")
cevnspdf_data = np.genfromtxt("../Data/cevnspdf.txt")
obs_data = np.genfromtxt("../Data/datanobkgsub.txt")

# Systematics
brnpdf_m1sigTiming = np.genfromtxt("../Data/SystErrors/brnpdf-1sigBRNTimingMean.txt")[:,3]
brnpdf_p1sigTiming = np.genfromtxt("../Data/SystErrors/brnpdf+1sigBRNTimingMean.txt")[:,3]
brnpdf_m1sigEnergy = np.genfromtxt("../Data/SystErrors/brnpdf-1sigEnergy.txt")[:,3]
brnpdf_p1sigEnergy = np.genfromtxt("../Data/SystErrors/brnpdf+1sigEnergy.txt")[:,3]
cevnspdf_m1sigF90 = np.genfromtxt("../Data/SystErrors/cevnspdf-1sigF90.txt")[:,3]
cevnspdf_p1sigF90 = np.genfromtxt("../Data/SystErrors/cevnspdf+1sigF90.txt")[:,3]
cevnspdfCEvNSTiming = np.genfromtxt("../Data/SystErrors/cevnspdfCEvNSTimingMeanSyst.txt")[:,3]
brnpdfBRNTimingWidth = np.genfromtxt("../Data/SystErrors/brnpdfBRNTimingWidthSyst.txt")[:,3]


# Flat bins
entries = bkgpdf_data.shape[0]
keVee = bkgpdf_data[:,0]
f90 = bkgpdf_data[:,1]
timing = bkgpdf_data[:,2]

# Bins
timing_edges = np.linspace(-0.1, 4.9, 11)
timing_bins = (timing_edges[1:]+timing_edges[:-1])/2
f90_edges = np.linspace(0.5, 0.9, 9)
f90_bins = (f90_edges[1:]+f90_edges[:-1])/2
keVee_edges = np.linspace(0,120,13)
keVee_bins = (keVee_edges[1:]+keVee_edges[:-1])/2


# Counts
bkgpdf = bkgpdf_data[:,3]
brnpdf = brnpdf_data[:,3] + brndelayedpdf_data[:,3]
cevnspdf = cevnspdf_data[:,3]
bkgpdf_bf = bkgpdf_data[:,3] * 3131 / np.sum(bkgpdf_data[:,3])
brnpdf_bf = brnpdf_data[:,3] * 553 / np.sum(brnpdf_data[:,3]) + brndelayedpdf_data[:,3] * 10 / np.sum(brndelayedpdf_data[:,3])
cevnspdf_bf = cevnspdf_data[:,3] * 159 / np.sum(cevnspdf_data[:,3])
obs = obs_data[:,3]
obs_errors = np.sqrt(obs_data[:,3]+1)

# Apply cuts and subtract SS
for i in range(entries):
    _t = timing[i]
    _f90 = f90[i]
    _keVee = keVee[i]
    #obs[i] -= bkgpdf[i]
    if _t > 12:
        obs[i] = 0.0
        bkgpdf[i] = 0.0
        brnpdf[i] = 0.0
        cevnspdf[i] = 0.0
        bkgpdf_bf[i] = 0.0
        brnpdf_bf[i] = 0.0
        cevnspdf_bf[i] = 0.0
        brnpdf_m1sigTiming[i] = 0.0
        brnpdf_p1sigTiming[i] = 0.0
        brnpdf_m1sigEnergy[i] = 0.0
        brnpdf_p1sigEnergy[i] = 0.0
        cevnspdf_m1sigF90[i] = 0.0
        cevnspdf_p1sigF90[i] = 0.0
        cevnspdfCEvNSTiming[i] = 0.0
        cevnspdf_p1sigF90[i] = 0.0


print("Total observations = ", np.sum(obs))
print("BRN = ", np.sum(brnpdf))
print("CEvNS = ", np.sum(cevnspdf))

pass

# Calculate statistis
def gaus(obs, errors, theory):
        ll = -0.5*log(2*pi*errors**2) - 0.5*((theory - obs)**2 / errors**2)
        return np.sum(ll)

def poisson(obs, errors, theory):
        ll = obs * log(theory) - theory - gammaln(obs+1)
        return np.sum(ll)

print("Significance (stat only):")
print(sqrt(abs(2*(gaus(obs, obs_errors, cevnspdf_bf+brnpdf_bf+bkgpdf_bf) - gaus(obs, obs_errors, brnpdf_bf+bkgpdf_bf)))))
print("Significance (stat+syst):")
total_errors = sqrt(obs_errors**2 + (1.085*obs)**2)
print(sqrt(abs(2*(-gaus(obs, total_errors, cevnspdf_bf+brnpdf_bf+bkgpdf_bf) + gaus(obs, total_errors, brnpdf_bf+bkgpdf_bf)))))




obs_keVee = np.histogram(keVee, weights=obs, bins=keVee_edges)[0]
obs_f90 = np.histogram(f90, weights=obs, bins=f90_edges)[0]
obs_timing = np.histogram(timing, weights=obs, bins=timing_edges)[0]

pdf_keVee = np.histogram(keVee, weights=cevnspdf+brnpdf, bins=keVee_edges)[0]
pdf_f90 = np.histogram(f90, weights=cevnspdf+brnpdf, bins=f90_edges)[0]
pdf_timing = np.histogram(timing, weights=cevnspdf+brnpdf, bins=timing_edges)[0]

bkg_keVee = np.histogram(keVee, weights=bkgpdf, bins=keVee_edges)[0]
bkg_f90 = np.histogram(f90, weights=bkgpdf, bins=f90_edges)[0]
bkg_timing = np.histogram(timing, weights=bkgpdf, bins=timing_edges)[0]

stat_errors_keVee = sqrt(obs_keVee + bkg_keVee)
stat_errors_f90 = sqrt(obs_f90 + bkg_f90)
stat_errors_timing = sqrt(obs_timing + bkg_timing)

# Add systematics in quadrature
brnpdf_m1sigTiming_keVee = np.histogram(keVee, weights=brnpdf_m1sigTiming**2, bins=keVee_edges)[0]
brnpdf_p1sigTiming_keVee = np.histogram(keVee, weights=brnpdf_p1sigTiming**2, bins=keVee_edges)[0]
brnpdf_m1sigEnergy_keVee = np.histogram(keVee, weights=brnpdf_m1sigEnergy**2, bins=keVee_edges)[0]
brnpdf_p1sigTiming_keVee = np.histogram(keVee, weights=brnpdf_p1sigEnergy**2, bins=keVee_edges)[0]
cevnspdf_m1sigF90_keVee = np.histogram(keVee, weights=cevnspdf_m1sigF90**2, bins=keVee_edges)[0]
cevnspdf_p1sigF90_keVee = np.histogram(keVee, weights=cevnspdf_p1sigF90**2, bins=keVee_edges)[0]
cevnspdfCEvNSTiming_keVee = np.histogram(keVee, weights=cevnspdfCEvNSTiming**2, bins=keVee_edges)[0]
brnpdfBRNTimingWidth_keVee = np.histogram(keVee, weights=cevnspdf_p1sigF90**2, bins=keVee_edges)[0]
syst_keVee_upper = sqrt(brnpdf_p1sigTiming_keVee + brnpdf_p1sigTiming_keVee + cevnspdf_p1sigF90_keVee + cevnspdfCEvNSTiming_keVee + brnpdfBRNTimingWidth_keVee)/2
syst_keVee_lower = sqrt(brnpdf_m1sigTiming_keVee + brnpdf_m1sigTiming_keVee + cevnspdf_m1sigF90_keVee + cevnspdfCEvNSTiming_keVee + brnpdfBRNTimingWidth_keVee)/2

brnpdf_m1sigTiming_f90 = np.histogram(f90, weights=brnpdf_m1sigTiming**2, bins=f90_edges)[0]
brnpdf_p1sigTiming_f90 = np.histogram(f90, weights=brnpdf_p1sigTiming**2, bins=f90_edges)[0]
brnpdf_m1sigEnergy_f90 = np.histogram(f90, weights=brnpdf_m1sigEnergy**2, bins=f90_edges)[0]
brnpdf_p1sigTiming_f90 = np.histogram(f90, weights=brnpdf_p1sigEnergy**2, bins=f90_edges)[0]
cevnspdf_m1sigF90_f90 = np.histogram(f90, weights=cevnspdf_m1sigF90**2, bins=f90_edges)[0]
cevnspdf_p1sigF90_f90 = np.histogram(f90, weights=cevnspdf_p1sigF90**2, bins=f90_edges)[0]
cevnspdfCEvNSTiming_f90 = np.histogram(f90, weights=cevnspdfCEvNSTiming**2, bins=f90_edges)[0]
brnpdfBRNTimingWidth_f90 = np.histogram(f90, weights=cevnspdf_p1sigF90**2, bins=f90_edges)[0]
syst_f90_upper = sqrt(brnpdf_p1sigTiming_f90 + brnpdf_p1sigTiming_f90 + cevnspdf_p1sigF90_f90 + cevnspdfCEvNSTiming_f90 + brnpdfBRNTimingWidth_f90)/2
syst_f90_lower = sqrt(brnpdf_m1sigTiming_f90 + brnpdf_m1sigTiming_f90 + cevnspdf_m1sigF90_f90 + cevnspdfCEvNSTiming_f90 + brnpdfBRNTimingWidth_f90)/2

brnpdf_m1sigTiming_timing = np.histogram(timing, weights=brnpdf_m1sigTiming**2, bins=timing_edges)[0]
brnpdf_p1sigTiming_timing = np.histogram(timing, weights=brnpdf_p1sigTiming**2, bins=timing_edges)[0]
brnpdf_m1sigEnergy_timing = np.histogram(timing, weights=brnpdf_m1sigEnergy**2, bins=timing_edges)[0]
brnpdf_p1sigTiming_timing = np.histogram(timing, weights=brnpdf_p1sigEnergy**2, bins=timing_edges)[0]
cevnspdf_m1sigF90_timing = np.histogram(timing, weights=cevnspdf_m1sigF90**2, bins=timing_edges)[0]
cevnspdf_p1sigF90_timing = np.histogram(timing, weights=cevnspdf_p1sigF90**2, bins=timing_edges)[0]
cevnspdfCEvNSTiming_timing = np.histogram(timing, weights=cevnspdfCEvNSTiming**2, bins=timing_edges)[0]
brnpdfBRNTimingWidth_timing = np.histogram(timing, weights=cevnspdf_p1sigF90**2, bins=timing_edges)[0]
syst_timing_upper = sqrt(brnpdf_p1sigTiming_timing + brnpdf_p1sigTiming_timing + cevnspdf_p1sigF90_timing + cevnspdfCEvNSTiming_timing + brnpdfBRNTimingWidth_timing)/2
syst_timing_lower = sqrt(brnpdf_m1sigTiming_timing + brnpdf_m1sigTiming_timing + cevnspdf_m1sigF90_timing + cevnspdfCEvNSTiming_timing + brnpdfBRNTimingWidth_timing)/2



# Plot energy spectra
plt.errorbar(keVee_bins, obs_keVee, yerr=stat_errors_keVee, color="k", ls="none", marker="o")
plt.hist([keVee,keVee], weights=[cevnspdf,brnpdf], bins=keVee_edges, stacked=True,
        histtype='step', label=["CEvNS", "BRN"], color=['blue', 'orange'])
plt.fill_between(keVee_bins, pdf_keVee+syst_keVee_upper,pdf_keVee-syst_keVee_lower, facecolor="orange", alpha=0.2, step='mid')
plt.xlabel(r"$E$ [keVee]")
plt.ylabel(r"Counts")
plt.title(r"$t < 1.5$ $\mu$s, $E > 10$ keVee, $E < 20$ keVee", fontsize=15, loc='right')
plt.legend()
plt.show()
plt.close()

# Plot timing spectra
plt.errorbar(timing_bins, obs_timing, yerr=stat_errors_timing, color="k", ls="none", marker="o")
plt.hist([timing,timing], weights=[cevnspdf,brnpdf], bins=timing_edges, stacked=True,
        histtype='step', label=["CEvNS", "BRN"], color=['blue', 'orange'])
plt.fill_between(timing_bins, pdf_timing+syst_timing_upper,pdf_timing-syst_timing_lower, facecolor="orange", alpha=0.2, step='mid')
plt.xlabel(r"$t$ [$\mu$s]")
plt.ylabel(r"Counts")
plt.legend()
plt.show()
plt.close()

# Plot f90 spectra
plt.errorbar(f90_bins, obs_f90, yerr=stat_errors_f90, color="k", ls="none", marker="o")
plt.hist([f90,f90], weights=[cevnspdf,brnpdf], bins=f90_edges, stacked=True,
        histtype='step', label=["CEvNS", "BRN"])
plt.fill_between(f90_bins, pdf_f90+syst_f90_upper,pdf_f90-syst_f90_lower, facecolor="orange", alpha=0.2, step='mid')
plt.title(r"$t < 1.5$ $\mu$s, $E > 10$ keVee, $E < 20$ keVee", fontsize=15, loc='right')
plt.xlabel(r"F90")
plt.ylabel(r"Counts")
plt.legend()
plt.show()
plt.close()


# Systematic PDFs
plt.hist(f90, weights=cevnspdf, bins=f90_edges, histtype='step', label="CEvNS")
plt.hist(f90, weights=cevnspdf_m1sigF90, bins=f90_edges, histtype='step', label="CEvNS -1 sigma")
plt.hist(f90, weights=cevnspdf_p1sigF90, bins=f90_edges, histtype='step', label="CEvNS +1 sigma")
plt.xlabel(r"F90")
plt.ylabel(r"Counts")
plt.legend()
plt.show()
plt.close()

# Systematic PDFs
plt.hist(timing, weights=cevnspdf, bins=timing_edges, histtype='step', label="CEvNS")
plt.hist(timing, weights=cevnspdfCEvNSTiming, bins=timing_edges, histtype='step', label="CEvNS timing syst")
plt.xlabel(r"Time")
plt.ylabel(r"Counts")
plt.legend()
plt.show()
plt.close()

# Systematic PDFs
plt.hist(timing, weights=brnpdf, bins=timing_edges, histtype='step', label="BRN")
plt.hist(timing, weights=brnpdfBRNTimingWidth, bins=timing_edges, histtype='step', label=r"BRN timing width syst $+1\sigma$")
plt.xlabel(r"Time")
plt.ylabel(r"Counts")
plt.legend()
plt.show()
plt.close()


# 2D hist:
plt.hist2d(keVee, f90, weights=bkgpdf, bins=[keVee_edges, f90_edges])
plt.xlabel("E (keVee)")
plt.ylabel("F90")
plt.title("SS", loc="right")
plt.show()

plt.hist2d(keVee, f90, weights=brnpdf, bins=[keVee_edges, f90_edges])
plt.xlabel("E (keVee)")
plt.ylabel("F90")
plt.title("BRN", loc="right")
plt.show()

plt.hist2d(keVee, f90, weights=cevnspdf, bins=[keVee_edges, f90_edges])
plt.xlabel("E (keVee)")
plt.ylabel("F90")
plt.title("CEvNS", loc="right")
plt.show()