from pyCEvNS.plot import CrediblePlot
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

namelist_nsi = [r'$\epsilon^{e,V}_{ee}$',r'$\epsilon^{e,V}_{\mu\mu}$', r'$\epsilon^{e,V}_{\tau\tau}$',
            r'$\epsilon^{e,V}_{e\mu}$',r'$\epsilon^{e,V}_{e\tau}$',
            r'$\epsilon^{e,V}_{\mu\tau}$', r'$\epsilon^{q,V}_{ee}$',r'$\epsilon^{q,V}_{\mu\mu}$', r'$\epsilon^{q,V}_{\tau\tau}$',
            r'$\epsilon^{q,V}_{e\mu}$',r'$\epsilon^{q,V}_{e\tau}$',
            r'$\epsilon^{q,V}_{\mu\tau}$']

lr_names = [r'$\epsilon^{e,L}_{ee}$',r'$\epsilon^{e,R}_{ee}$',
            r'$\epsilon^{e,L}_{\mu\mu}$',r'$\epsilon^{e,R}_{\mu\mu}$',
            r'$\epsilon^{e,L}_{\tau\tau}$',r'$\epsilon^{e,R}_{\tau\tau}$',
            r'$\epsilon^{e,L}_{e\mu}$',r'$\epsilon^{e,R}_{e\mu}$',
            r'$\epsilon^{e,L}_{e\tau}$',r'$\epsilon^{e,R}_{e\tau}$',
            r'$\epsilon^{e,L}_{\mu\tau}$',r'$\epsilon^{e,R}_{\mu\tau}$',
]

all_namelist = [r'$\epsilon^{e,V}_{ee}$', r'$\epsilon^{e,V}_{\mu\mu}$', r'$\epsilon^{e,V}_{\tau\tau}$',
                r'$\epsilon^{e,V}_{e\mu}$', r'$\epsilon^{e,V}_{e\tau}$', r'$\epsilon^{e,V}_{\mu\tau}$',
                r'$\epsilon^{u,V}_{ee}$', r'$\epsilon^{u,V}_{\mu\mu}$', r'$\epsilon^{u,V}_{\tau\tau}$',
                r'$\epsilon^{u,V}_{e\mu}$', r'$\epsilon^{u,V}_{e\tau}$', r'$\epsilon^{u,V}_{\mu\tau}$',
                r'$\epsilon^{d,V}_{ee}$', r'$\epsilon^{d,V}_{\mu\mu}$', r'$\epsilon^{d,V}_{\tau\tau}$',
                r'$\epsilon^{d,V}_{e\mu}$', r'$\epsilon^{d,V}_{e\tau}$', r'$\epsilon^{d,V}_{\mu\tau}$']

# STAGE 1
borex_vector_solars = CrediblePlot("nsi_multinest/borexino_12dim_nsi_nuissance/borexino_vector_nsi_nuisance.txt")
coherent_ar_csi = CrediblePlot("nsi_multinest/coherent_csi_ar/coherent_csi_ar.txt")

# STAGE 2
xenon_atmos = CrediblePlot("nsi_multinest/all_nsi_xenon_atmos_coh_prior/all_nsi_xenon_atmos_coh_prior.txt")
xenon_vector_solars = CrediblePlot("nsi_multinest/all_nsi_xenon_borexino_prior/vector_nsi_xenon_borexino_prior.txt")

# STAGE 3
priorflow = CrediblePlot("nsi_multinest/prior_flow_ud_combined/prior_flow_ud_combined.txt")
priorflow_18d = CrediblePlot("nsi_multinest/prior_flow_18D_v2/prior_flow_18D_v2_TTFIXED.txt")


# Inferno
color_3 = "#42039D"
color_2 = "#C5407D"
color_1 = "#FDAF31"
# Viridis
#color_3 = "#39568CFF"
#color_2 = "#1F968BFF"
#color_1 = "#FDE725FF"


# Plot L/R Priorflow
print("Plotting Priorflow L/R")
filelist = ["nsi_multinest/borexino_12dim_nsi_nuissance/borexino_12dim_nsi_nuissance.txt"]
idx_list = [((0,1,2,3,4,5,6,7,8,9,10,11),(0,1,2,3,4,5,6,7,8,9,10,11))]
colors=[color_2, color_1]

left_right_12d = CrediblePlot("nsi_multinest/all_nsi_xenon_borexino_prior/all_nsi_xenon_borexino_prior.txt")

fig, ax = left_right_12d.special_grid(filelist, (0,1,2,3,4,5,6,7,8,9,10,11), idx_list,
                                              (0,0,0,0,0,0,0,0,0,0,0,0), lr_names, colors, nbins=40)
fig.savefig("plots/credible/priorflow/priorflow_LR_grid_overlay.png")
fig.savefig("plots/credible/priorflow/priorflow_LR_grid_overlay.pdf")





# 2D comparison with 12 NSI (U+D)
print("Plotting Priorflow U+D summed 2d grid special overlay")
filelist = ["nsi_multinest/coherent_csi_ar/coherent_csi_ar_sumUD.txt",
            "nsi_multinest/all_nsi_xenon_atmos_coh_prior/all_nsi_xenon_atmos_coh_prior_sumUD.txt",
            "nsi_multinest/borexino_12dim_nsi_nuissance/borexino_vector_nsi_nuisance.txt",
            "nsi_multinest/all_nsi_xenon_borexino_prior/vector_nsi_xenon_borexino_prior.txt"]
idx_list = [((0,1,2,3,4),(6,7,9,10,11)),
            ((0,1,2,3,4,5),(6,7,8,9,10,11)),
            ((0,1,2,3,4,5),(0,1,2,3,4,5)),
            ((0,1,2,3,4,5),(0,1,2,3,4,5))]
colors=[color_3, color_1, color_2, color_1, color_2]

#priorflow_12d = CrediblePlot("special_posteriors/priorflow_ud_summed_12d.txt")
priorflow_12d = CrediblePlot("nsi_multinest/prior_flow_18D_v2/prior_flow_18D_v2_sumUD_TTFIXED.txt")
fig, ax = priorflow_12d.special_grid(filelist, (0,1,2,3,4,5,6,7,8,9,10,11), idx_list,
                                     (0,0,0,0,0,0,0,0,0,0,0,0), namelist_nsi, colors,
                                     nbins=30, credible_level=(0.96,))
fig.savefig("plots/credible/priorflow/priorflow_12d_grid_overlay.png")
fig.savefig("plots/credible/priorflow/priorflow_12d_grid_overlay.pdf")




# 2D comparison with 18 NSI (U+D)
print("Plotting Priorflow 18d grid special overlay")
filelist = ["nsi_multinest/coherent_csi_ar/coherent_csi_ar.txt",
            "nsi_multinest/all_nsi_xenon_atmos_coh_prior/all_nsi_xenon_atmos_coh_prior.txt",
            "nsi_multinest/borexino_12dim_nsi_nuissance/borexino_vector_nsi_nuisance.txt",
            "nsi_multinest/all_nsi_xenon_borexino_prior/vector_nsi_xenon_borexino_prior.txt"]
idx_list = [((0,1,2,3,4,5,6,7,8,9),(6,7,9,10,11,12,13,15,16,17)),
            ((0,1,2,3,4,5,6,7,8,9,10,11),(6,7,8,9,10,11,12,13,14,15,16,17)),
            ((0,1,2,3,4,5),(0,1,2,3,4,5)),
            ((0,1,2,3,4,5),(0,1,2,3,4,5))]
colors=[color_3, color_1, color_2, color_1, color_2]

#priorflow_12d = CrediblePlot("special_posteriors/priorflow_ud_summed_12d.txt")
fig, ax = priorflow_18d.special_grid(filelist, (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17), idx_list,
                                     (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), all_namelist, colors,
                                     nbins=30, credible_level=(0.96,))
fig.savefig("plots/credible/priorflow/priorflow_18d_grid_overlay.png")
fig.savefig("plots/credible/priorflow/priorflow_18d_grid_overlay.pdf")




"""
print("---- SUM UD CREDIBLES ---")
plt.clf()
coherent_UD = CrediblePlot("nsi_multinest/coherent_csi_ar/coherent_csi_ar_sumUD.txt")
priorflow_UD = CrediblePlot("nsi_multinest/prior_flow_18D_v2/prior_flow_18D_v2_sumUD_TTFIXED.txt")
print("u+d: EE")
coherent_UD.credible_1d(idx=0, smooth=True, nbins=40, color=color_3)
priorflow_UD.credible_1d(idx=6, smooth=True, nbins=40, color=color_3)
print("u+d: MM")
coherent_UD.credible_1d(idx=1, smooth=True, nbins=40, color=color_3)
priorflow_UD.credible_1d(idx=7, smooth=True, nbins=40, color=color_3)
print("u+d: TT")
priorflow_UD.credible_1d(idx=8, smooth=True, nbins=40, color=color_3)
print("u+d: EM")
coherent_UD.credible_1d(idx=2, smooth=True, nbins=40, color=color_3)
priorflow_UD.credible_1d(idx=9, smooth=True, nbins=40, color=color_3)
print("u+d: ET")
coherent_UD.credible_1d(idx=3, smooth=True, nbins=40, color=color_3)
priorflow_UD.credible_1d(idx=10, smooth=True, nbins=40, color=color_3)
print("u+d: MT")
coherent_UD.credible_1d(idx=4, smooth=True, nbins=40, color=color_3)
priorflow_UD.credible_1d(idx=11, smooth=True, nbins=40, color=color_3)
plt.clf()

"""

print("Plotting priorflow comparison")
x = np.linspace(-1,1,10)
y = 0.5*np.ones(10)
fig = plt.figure(figsize=(12,6))
#fig.subplots_adjust(wspace=0.)
gs = gridspec.GridSpec(3, 6)

# E NSI
ax = fig.add_subplot(gs[0, 0])
print("Priorflow ee")
priorflow_18d.credible_1d(idx=0, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_vector_solars.credible_1d(idx=0, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("Borex ee")
borex_vector_solars.credible_1d(idx=0, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{e,V}_{ee}$", fontsize=15)
plt.ylabel("Posterior Density")
ax = fig.add_subplot(gs[0, 1])
print("Priorflow mm")
priorflow_18d.credible_1d(idx=1, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_vector_solars.credible_1d(idx=1, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("Borex mm")
borex_vector_solars.credible_1d(idx=1, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{e,V}_{\mu\mu}$", fontsize=15)
ax = fig.add_subplot(gs[0, 2])
print("Priorflow tt")
priorflow_18d.credible_1d(idx=2, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_vector_solars.credible_1d(idx=2, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("Borex tt")
borex_vector_solars.credible_1d(idx=2, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{e,V}_{\tau\tau}$", fontsize=15)
ax = fig.add_subplot(gs[0, 3])
print("Priorflow em")
priorflow_18d.credible_1d(idx=3, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_vector_solars.credible_1d(idx=3, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("Borex em")
borex_vector_solars.credible_1d(idx=3, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{e,V}_{e\mu}$", fontsize=15)
ax = fig.add_subplot(gs[0, 4])
print("Priorflow et")
priorflow_18d.credible_1d(idx=4, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_vector_solars.credible_1d(idx=4, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("Borex et")
borex_vector_solars.credible_1d(idx=4, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{e,V}_{e\tau}$", fontsize=15)
ax = fig.add_subplot(gs[0, 5])
print("Priorflow mt")
priorflow_18d.credible_1d(idx=5, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_vector_solars.credible_1d(idx=5, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("Borex mt")
borex_vector_solars.credible_1d(idx=5, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{e,V}_{\mu\tau}$", fontsize=15)


## UP NSI
ax = fig.add_subplot(gs[1, 0])
print("Priorflow u ee")
priorflow_18d.credible_1d(idx=6, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_atmos.credible_1d(idx=0, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("COHERENT u ee")
coherent_ar_csi.credible_1d(idx=0, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{u,V}_{ee}$", fontsize=15)
plt.ylabel("Posterior Density")
ax = fig.add_subplot(gs[1, 1])
print("Priorflow u mm")
priorflow_18d.credible_1d(idx=7, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_atmos.credible_1d(idx=1, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("COHERENT u MM")
coherent_ar_csi.credible_1d(idx=1, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{u,V}_{\mu\mu}$", fontsize=15)
ax = fig.add_subplot(gs[1, 2])
print("Priorflow u tt")
priorflow_18d.credible_1d(idx=8, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_atmos.credible_1d(idx=2, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
ax.plot(x,y, color=color_1, ls="dashed")
ax.set_aspect("equal")
ax.set_aspect(1./ax.get_data_ratio())
plt.xlabel(r"$\epsilon^{u,V}_{\tau\tau}$", fontsize=15)
ax = fig.add_subplot(gs[1, 3])
print("Priorflow u em")
priorflow_18d.credible_1d(idx=9, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_atmos.credible_1d(idx=3, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("COHERENT u eM")
coherent_ar_csi.credible_1d(idx=2, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{u,V}_{e\mu}$", fontsize=15)
ax = fig.add_subplot(gs[1, 4])
print("Priorflow u et")
priorflow_18d.credible_1d(idx=10, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_atmos.credible_1d(idx=4, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("COHERENT u et")
coherent_ar_csi.credible_1d(idx=3, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{u,V}_{e\tau}$", fontsize=15)
ax = fig.add_subplot(gs[1, 5])
print("Priorflow u mt")
priorflow_18d.credible_1d(idx=11, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_atmos.credible_1d(idx=5, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("COHERENT u mt")
coherent_ar_csi.credible_1d(idx=4, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{u,V}_{\mu\tau}$", fontsize=15)

## DOWN NSI
ax = fig.add_subplot(gs[2, 0])
print("Priorflow d ee")
priorflow_18d.credible_1d(idx=12, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_atmos.credible_1d(idx=6, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("COHERENT d ee")
coherent_ar_csi.credible_1d(idx=5, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{d,V}_{ee}$", fontsize=15)
plt.ylabel("Posterior Density")
ax = fig.add_subplot(gs[2, 1])
print("Priorflow d mm")
priorflow_18d.credible_1d(idx=13, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_atmos.credible_1d(idx=7, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("COHERENT d mm")
coherent_ar_csi.credible_1d(idx=6, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{d,V}_{\mu\mu}$", fontsize=15)
ax = fig.add_subplot(gs[2, 2])
print("Priorflow d tt")
priorflow_18d.credible_1d(idx=14, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_atmos.credible_1d(idx=8, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
ax.plot(x,y, color=color_1, ls="dashed")
ax.set_aspect("equal")
ax.set_aspect(1./ax.get_data_ratio())
plt.xlabel(r"$\epsilon^{d,V}_{\tau\tau}$", fontsize=15)
ax = fig.add_subplot(gs[2, 3])
print("Priorflow d em")
priorflow_18d.credible_1d(idx=15, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_atmos.credible_1d(idx=9, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("COHERENT d em")
coherent_ar_csi.credible_1d(idx=7, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{d,V}_{e\mu}$", fontsize=15)
ax = fig.add_subplot(gs[2, 4])
print("Priorflow d et")
priorflow_18d.credible_1d(idx=16, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_atmos.credible_1d(idx=10, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("COHERENT d et")
coherent_ar_csi.credible_1d(idx=8, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{d,V}_{e\tau}$", fontsize=15)
ax = fig.add_subplot(gs[2, 5])
print("Priorflow d mt")
priorflow_18d.credible_1d(idx=17, smooth=True, nbins=40, ax=ax, color=color_3)
xenon_atmos.credible_1d(idx=11, smooth=True, nbins=40, ax=ax, color=color_2, ls="dashdot")
print("COHERENT d mt")
coherent_ar_csi.credible_1d(idx=9, smooth=True, nbins=40, ax=ax, color=color_1, ls="dashed")
plt.xlabel(r"$\epsilon^{d,V}_{\mu\tau}$", fontsize=15)

plt.tight_layout()
plt.savefig("plots/credible/priorflow/marginals_comparison_18d.png")
plt.savefig("plots/credible/priorflow/marginals_comparison_18d.pdf")
plt.clf()










# 18D priorflow
print("plotting 18d grid")
fig, ax = priorflow_18d.credible_grid((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17), (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), all_namelist, nbins=30)
fig.savefig("plots/credible/priorflow/priorflow_18d_borexino.png")
fig.savefig("plots/credible/priorflow/priorflow_18d_borexino.pdf")




