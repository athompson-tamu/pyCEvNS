from pyCEvNS.plot import CrediblePlot
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec


namelist = [r'$\epsilon_{ee}$',r'$\epsilon_{\mu\mu}$', r'$\epsilon_{\tau\tau}$',
            r'$\epsilon_{e\mu}$',r'$\epsilon_{e\tau}$',
            r'$\epsilon_{\mu\tau}$', r'$\delta_{CP}$']
namelist_nodcp = [r'$\epsilon_{ee}$',r'$\epsilon_{\mu\mu}$', r'$\epsilon_{\tau\tau}$',
            r'$\epsilon_{e\mu}$',r'$\epsilon_{e\tau}$',
            r'$\epsilon_{\mu\tau}$']
namelist_nsi = [r'$\epsilon^e_{ee}$',r'$\epsilon^e_{\mu\mu}$', r'$\epsilon^e_{\tau\tau}$',
            r'$\epsilon^e_{e\mu}$',r'$\epsilon^e_{e\tau}$',
            r'$\epsilon^e_{\mu\tau}$', r'$\epsilon^q_{ee}$',r'$\epsilon^q_{\mu\mu}$', r'$\epsilon^q_{\tau\tau}$',
            r'$\epsilon^q_{e\mu}$',r'$\epsilon^q_{e\tau}$',
            r'$\epsilon^q_{\mu\tau}$']
single_namelist = [r'$\epsilon^{e,L}_{ee}$', r'$\epsilon^{e,R}_{ee}$',
                   r'$\epsilon^u_{ee}$', r'$\epsilon^d_{ee}$']
namelist_complex = [r'$\epsilon_{ee}$',r'$\epsilon_{\mu\mu}$', r'$\epsilon_{\tau\tau}$',
                    r'$\epsilon_{e\mu}$',r'$\epsilon_{e\tau}$', r'$\epsilon_{\mu\tau}$',
                    r'$\phi_{e\mu}$', r'$\delta_{CP}$']
solar_namelist_dcp = [r'$\epsilon^L_{ee}$',r'$\epsilon^R_{ee}$',r'$\epsilon^L_{\mu\mu}$',r'$\epsilon^R_{\mu\mu}$',
                  r'$\epsilon^L_{\tau\tau}$', r'$\epsilon^R_{\tau\tau}$',r'$\epsilon^L_{e\mu}$',r'$\epsilon^R_{e\mu}$',
                  r'$\epsilon^L_{e\tau}$',r'$\epsilon^R_{e\tau}$', r'$\epsilon^L_{\mu\tau}$', r'$\epsilon^R_{\mu\tau}$',
                  r'$\delta_{CP}$']
solar_namelist = [r'$\epsilon^L_{ee}$',r'$\epsilon^R_{ee}$',r'$\epsilon^L_{\mu\mu}$',r'$\epsilon^R_{\mu\mu}$',
                  r'$\epsilon^L_{\tau\tau}$', r'$\epsilon^R_{\tau\tau}$',r'$\epsilon^L_{e\mu}$',r'$\epsilon^R_{e\mu}$',
                  r'$\epsilon^L_{e\tau}$',r'$\epsilon^R_{e\tau}$', r'$\epsilon^L_{\mu\tau}$', r'$\epsilon^R_{\mu\tau}$']
solar_4d = ['Be7Flux',r'$\epsilon^L_{ee}$',r'$\epsilon^R_{ee}$',
            r'$\epsilon^L_{\tau\tau}$', r'$\epsilon^R_{\tau\tau}$']
solar_1d = [r'$\epsilon^L_{ee}$']
all_namelist = [r'$\epsilon^u_{ee}$',r'$\epsilon^u_{\mu\mu}$', r'$\epsilon^u_{\tau\tau}$',
                r'$\epsilon^u_{e\mu}$',r'$\epsilon^u_{e\tau}$',
                r'$\epsilon^u_{\mu\tau}$', r'$\epsilon^d_{ee}$',r'$\epsilon^d_{\mu\mu}$', r'$\epsilon^d_{\tau\tau}$',
                r'$\epsilon^d_{e\mu}$',r'$\epsilon^d_{e\tau}$',
                r'$\epsilon^d_{\mu\tau}$', r'$\epsilon^e_{ee}$',r'$\epsilon^e_{\mu\mu}$', r'$\epsilon^e_{\tau\tau}$',
                r'$\epsilon^e_{e\mu}$',r'$\epsilon^e_{e\tau}$',
                r'$\epsilon^e_{\mu\tau}$', r'$\delta_{CP}$']
ud_namelist = [r'$\epsilon^u_{ee}$',r'$\epsilon^u_{\mu\mu}$', r'$\epsilon^u_{\tau\tau}$',
                r'$\epsilon^u_{e\mu}$',r'$\epsilon^u_{e\tau}$',
                r'$\epsilon^u_{\mu\tau}$', r'$\epsilon^d_{ee}$',r'$\epsilon^d_{\mu\mu}$', r'$\epsilon^d_{\tau\tau}$',
                r'$\epsilon^d_{e\mu}$',r'$\epsilon^d_{e\tau}$',
                r'$\epsilon^d_{\mu\tau}$']
ud_namelist_nott = [r'$\epsilon^u_{ee}$',r'$\epsilon^u_{\mu\mu}$',
                r'$\epsilon^u_{e\mu}$',r'$\epsilon^u_{e\tau}$',
                r'$\epsilon^u_{\mu\tau}$', r'$\epsilon^d_{ee}$',r'$\epsilon^d_{\mu\mu}$',
                r'$\epsilon^d_{e\mu}$',r'$\epsilon^d_{e\tau}$',
                r'$\epsilon^d_{\mu\tau}$']
pheno_namelist = [r'$\epsilon^O_{ee}$',r'$\epsilon^O_{\mu\mu}$', r'$\epsilon^O_{\tau\tau}$',
                r'$\epsilon^O_{e\mu}$',r'$\epsilon^O_{e\tau}$',
                r'$\epsilon^O_{\mu\tau}$', r'$\epsilon^N_{ee}$',r'$\epsilon^N_{\mu\mu}$', r'$\epsilon^N_{\tau\tau}$',
                r'$\epsilon^N_{e\mu}$',r'$\epsilon^N_{e\tau}$',
                r'$\epsilon^N_{\mu\tau}$']
diag_namelist = ["epel_ee", "epel_mumu", "epel_tautau",
                "eper_ee", "eper_mumu", "eper_tautau",
                "epu_ee", "epu_mumu", "epu_tautau",
                "epd_ee", "epd_mumu", "epd_tautau"]
vec_diag_namelist = [r'$\epsilon^e_{ee}$',r'$\epsilon^e_{\mu\mu}$', r'$\epsilon^e_{\tau\tau}$',
                     r'$\epsilon^u_{ee}$',r'$\epsilon^u_{\mu\mu}$', r'$\epsilon^u_{\tau\tau}$',
                     r'$\epsilon^d_{ee}$',r'$\epsilon^d_{\mu\mu}$', r'$\epsilon^d_{\tau\tau}$',]
vec_diag_namelist_ud = [r'$\epsilon^e_{ee}$',r'$\epsilon^e_{\mu\mu}$', r'$\epsilon^e_{\tau\tau}$',
                        r'$\epsilon^{u-d}_{ee}$',r'$\epsilon^{u-d}_{\mu\mu}$', r'$\epsilon^{u-d}_{\tau\tau}$']
idx = (0,1,2,3,4,5,6)
idx_single = (0,1,2,3)
idx_solar = (0,1,2,3,4,5,6,7,8,9,10,11,12)
idx_xen_dune = (0,1,2,3,4,5,6,7,8,9,10,11)
idx_complex = (0,1,2,3,4,5,6,7)
idx_diag = (0,1,2,3,4,5,6,7,8,9,10,11)
idx_vec_diag = (0,1,2,3,4,5,6,7,8)
idx_vec_diag_ud = (0,1,2,3,4,5)
idx_all = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18)
test_pt = (0.0,0.0,0.0,0.0,0.0,0.0,1.5*np.pi)
test_pt_single = (0,0,0,0)
test_pt_diag = (0,0,0,0,0,0,0,0,0,0,0,0)
test_pt_vec_diag = (0,0,0,0,0,0,0,0,0)
test_pt_vec_diag_ud = (0,0,0,0,0,0)
test_pt_complex = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.5*np.pi)
test_pt_solar = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.5*np.pi)
test_pt_all = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.5*np.pi)
rect1 = patches.Rectangle((0,0),20,20,facecolor='#FF605E')
rect2 = patches.Rectangle((0,0),20,20,facecolor='#64B2DF')




dune_beam = CrediblePlot("nsi_multinest/all_nsi_dune_beam_e_appearance/all_nsi_dune_beam_e_appearance.txt")
dune_atmos = CrediblePlot("nsi_multinest/all_nsi_dune_atmos_mu/all_nsi_dune_atmos_mu.txt")
dune_atmos_complex = CrediblePlot("nsi_multinest/all_single-complex_nsi_dune_atmos_mu/modified.txt")
hyperk = CrediblePlot("nsi_multinest/all_nsi_hyperk_atmos_mu/all_nsi_hyperk_atmos_mu.txt")
xe_atmos = CrediblePlot("nsi_multinest/all_nsi_xenon_atmos/all_nsi_xenon_atmos.txt")
xe_solar_vect = CrediblePlot("python/multinest_posteriors/all_nsi_vector_solar_05prior.txt")
xe_solars = CrediblePlot("git/pyCEvNS/multinest/all_nsi_left_right_solar/all_nsi_left_right_solar.txt")
xe_borex_prior = CrediblePlot("nsi_multinest/all_nsi_xenon_borexino_prior/all_nsi_xenon_borexino_prior.txt")
xe_borex_prior_vector = CrediblePlot("nsi_multinest/all_nsi_xenon_borexino_prior/vector_nsi_xenon_borexino_prior.txt")
borex_solars = CrediblePlot("nsi_multinest/borexino_12dim_nsi_nuissance/borexino_12dim_nsi_nuissance.txt")
borex_vector_solars = CrediblePlot("nsi_multinest/borexino_12dim_nsi_fluxMod/borexino_6dim_vector_nsi_fluxMod.txt")
borex_4d = CrediblePlot("nsi_multinest/borexino_4dim_nsi_fluxMod/borexino_4dim_nsi_fluxMod.txt")
borex_eeL = CrediblePlot("nsi_multinest/borexino_1dim_nsi_eeL/borexino_1dim_nsi_eeL.txt")
borex_eeR = CrediblePlot("nsi_multinest/borexino_1dim_nsi_eeR/borexino_1dim_nsi_eeR.txt")
borex_ttL = CrediblePlot("nsi_multinest/borexino_1dim_nsi_ttL/borexino_1dim_nsi_ttL.txt")
borex_ttR = CrediblePlot("nsi_multinest/borexino_1dim_nsi_ttR/borexino_1dim_nsi_ttR.txt")
coherent = CrediblePlot("nsi_multinest/coherent_ud/coherent_ud_summed.txt")
coherent_ud = CrediblePlot("nsi_multinest/coherent_ud/coherent_ud_.txt")
coherent_ud_nott = CrediblePlot("nsi_multinest/coherent_ud_nott/coherent_ud_nott_.txt")
xen_ud_summed = CrediblePlot("python/multinest_posteriors/all_nsi_xenon_atmos_coh-prior_ud-summed.txt")
xen_ud = CrediblePlot("nsi_multinest/all_ud_nsi_xenon_atmos/all_ud_nsi_xenon_atmos.txt")
xen_priorflow = CrediblePlot("nsi_multinest/all_nsi_xenon_atmos_coh_prior/all_nsi_xenon_atmos_coh_prior.txt")
xen_priorflow_no_cop = CrediblePlot("nsi_multinest/all_nsi_xenon_atmos_coh-marg_prior/all_nsi_xenon_atmos_coh-marg_prior.txt")
priorflow = CrediblePlot("nsi_multinest/prior_flow_ud_combined/prior_flow_ud_combined.txt")
priorflow_marginals = CrediblePlot("nsi_multinest/prior_flow_marginals_ud_combined/prior_flow_marginals_ud_combined.txt")

print("plotting Xenon solars with borexino prior")
fig, ax = xe_borex_prior.credible_grid((0,1,2,3,4,5,6,7,8,9,10,11),
                                     (0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.), solar_namelist, nbins=40)
fig.savefig("plots/credible/all_nsi_xenon_solars_borex_prior.png")
fig.savefig("plots/credible/all_nsi_xenon_solars_borex_prior.pdf")

print("plotting Xenon solars vector sum")
fig, ax = xe_borex_prior_vector.credible_grid((0,1,2,3,4,5), (0.0,0.0,0.0,0.0,0.0,0.0), namelist_nodcp, nbins=40)
fig.savefig("plots/credible/all_nsi_xenon_solars_borex_prior_vector.png")
fig.savefig("plots/credible/all_nsi_xenon_solars_borex_prior_vector.pdf")

# Borexino solars
print("plotting Borex solars")
fig, ax = borex_solars.credible_grid((0,1,2,3,4,5,6,7,8,9,10,11),
                                     (0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.), solar_namelist, nbins=40)
fig.suptitle(r"$\epsilon^{e,L}_{\alpha\beta}, \epsilon^{e,R}_{\alpha\beta}$, Borexino Solar Neutrino Appearance",
             x=0.6, y=0.8,
             fontsize=80)
fig.savefig("plots/credible/all_nsi_borexino_solars_nuissance.png")
fig.savefig("plots/credible/all_nsi_borexino_solars_nuissance.pdf")

print("plotting Borex solars vector sum")
fig, ax = borex_vector_solars.credible_grid((0,1,2,3,4,5), (0.0,0.0,0.0,0.0,0.0,0.0), namelist_nodcp, nbins=40)
fig.suptitle(r"$\epsilon^{e,V}_{\alpha\beta} = \epsilon^{e,L}_{\alpha\beta} + \epsilon^{e,R}_{\alpha\beta}$, "
             r"Borexino Solar Neutrino Appearance",
             x=0.6, y=0.8,
             fontsize=40)
fig.savefig("plots/credible/all_nsi_borexino_vector-NSI_solars.png")
fig.savefig("plots/credible/all_nsi_borexino_vector-NSI_solars.pdf")

print("plotting Borex 4d")
fig, ax = borex_4d.credible_grid((0,1,2,3,4), (1.0,0.0,0.0,0.0,0.0), solar_4d, nbins=40)
fig.savefig("plots/credible/borexino_solars_4d.png")
fig.savefig("plots/credible/borexino_solars_4d.pdf")
print("plotting Borex 1D")
plt.clf()
fig = plt.figure(figsize=(8,4))
fig.subplots_adjust(wspace=0.1)
gs = gridspec.GridSpec(2, 4)
ax = fig.add_subplot(gs[0, 0])
borex_eeL.credible_1d(idx=1, smooth=True, nbins=40, ax=ax, color='b')
plt.xlabel(r"$\epsilon^{e,L}_{ee}$")
plt.ylabel("Posterior Density")
ax = fig.add_subplot(gs[0, 1])
borex_eeR.credible_1d(idx=1, smooth=True, nbins=40, ax=ax, color='b')
plt.xlabel(r"$\epsilon^{e,R}_{ee}$")
ax = fig.add_subplot(gs[0, 2])
borex_ttL.credible_1d(idx=1, smooth=True, nbins=40, ax=ax, color='b')
plt.xlabel(r"$\epsilon^{e,L}_{\tau\tau}$")
ax = fig.add_subplot(gs[0, 3])
borex_ttR.credible_1d(idx=1, smooth=True, nbins=40, ax=ax, color='b')
plt.xlabel(r"$\epsilon^{e,R}_{\tau\tau}$")
ax = fig.add_subplot(gs[1, 0])
borex_eeL.credible_1d(idx=0, smooth=True, nbins=40, ax=ax, color='b')
plt.xlabel("Be7 Modulation")
plt.ylabel("Posterior Density")
ax = fig.add_subplot(gs[1, 1])
borex_eeR.credible_1d(idx=0, smooth=True, nbins=40, ax=ax, color='b')
plt.xlabel("Be7 Modulation")
ax = fig.add_subplot(gs[1, 2])
borex_ttL.credible_1d(idx=0, smooth=True, nbins=40, ax=ax, color='b')
plt.xlabel("Be7 Modulation")
ax = fig.add_subplot(gs[1, 3])
borex_ttR.credible_1d(idx=0, smooth=True, nbins=40, ax=ax, color='b')
plt.xlabel("Be7 Modulation")
plt.tight_layout()
plt.savefig("plots/credible/borexino_1d_posteriors.png")
plt.clf()

print("plotting Prior Flow")
fig, ax = priorflow.credible_grid((0,1,2,3,4,5,6,7,8,9,10,11), (0,0,0,0,0,0,0,0,0,0,0,0), namelist_nsi, nbins=50)
fig.savefig("plots/credible/priorflow_ud_combined.png")
fig.savefig("plots/credible/priorflow_ud_combined.pdf")

print("plotting Prior Flow versus marginals only comparison")
fig, ax = priorflow.credible_grid_overlay("nsi_multinest/prior_flow_marginals_ud_combined/prior_flow_marginals_ud_combined.txt",
                                          (0,1,2,3,4,5,6,7,8,9,10,11), (0,0,0,0,0,0,0,0,0,0,0,0), namelist_nsi, nbins=50)
fig.savefig("plots/credible/priorflow_copula_vs_marginals.png")
fig.savefig("plots/credible/priorflow_copula_vs_marginals.pdf")


print("plotting COHERENT ud")
fig, ax = coherent_ud.credible_grid((0,1,2,3,4,5,6,7,8,9,10,11), (0,0,0,0,0,0,0,0,0,0,0,0), ud_namelist, nbins=40)
fig.savefig("plots/credible/coh_ud.png")
fig.savefig("plots/credible/coh_ud.pdf")

print("plotting COHERENT tt test")
fig, ax = coherent_ud_nott.credible_grid((0,1,2,3,4,5,6,7,8,9), (0,0,0,0,0,0,0,0,0,0), ud_namelist_nott, nbins=40)
fig.savefig("plots/credible/coh_nott.png")
fig.savefig("plots/credible/coh_nott.pdf")

# DUNE vs. Xe overlay
print("Plotting COHERENT prior comparison")
fig = plt.figure(figsize=(8,3))
fig.subplots_adjust(wspace=0.6)
gs = gridspec.GridSpec(1, 3)
ax = fig.add_subplot(gs[0, 0])
xen_ud.credible_2d(idx=(5,11), credible_level=(0.6827, 0.9545), nbins=40, ax=ax, center=(0,0), color='b')
plt.xlabel(r"$\epsilon^u_{\mu\tau}$")
plt.ylabel(r"$\epsilon^d_{\mu\tau}$")
ax = fig.add_subplot(gs[0, 1])
xen_priorflow_no_cop.credible_2d(idx=(5,11), credible_level=(0.6827, 0.9545), nbins=40, ax=ax, center=(0,0), color='b')
plt.xlabel(r"$\epsilon^u_{\mu\tau}$")
plt.ylabel(r"$\epsilon^d_{\mu\tau}$")
ax = fig.add_subplot(gs[0, 2])
xen_priorflow.credible_2d(idx=(5,11), credible_level=(0.6827, 0.9545), nbins=40, ax=ax, center=(0,0), color='b')
plt.xlabel(r"$\epsilon^u_{\mu\tau}$")
plt.ylabel(r"$\epsilon^d_{\mu\tau}$")
plt.tight_layout()

plt.savefig("plots/credible/xenon_prior_comparison_mt.png")
plt.savefig("plots/credible/xenon_prior_comparison_mt.pdf")
plt.clf()

# LXe Solars (Vector sum)
print("plotting LXe solars vector sum")
fig, ax = xe_solar_vect.credible_grid((0,1,2,3,4,5), (0.0,0.0,0.0,0.0,0.0,0.0), namelist_nodcp, nbins=40)
fig.suptitle(r"$\epsilon^{e,V}_{\alpha\beta} = \epsilon^{e,L}_{\alpha\beta} + \epsilon^{e,R}_{\alpha\beta}$, "
             r"Future-LXe Solar Neutrino Appearance",
             x=0.6, y=0.8,
             fontsize=40)
fig.savefig("plots/credible/all_nsi_xenon_solars_vector_sum_05prior.png")
fig.savefig("plots/credible/all_nsi_xenon_solars_vector_sum_05prior.pdf")


print("plotting DUNE Atmospherics")
fig, ax = dune_atmos.credible_grid((0,1), (0,0), namelist, nbins=50)
fig.savefig("plots/credible/dune_ee_mm_test.png")
fig.savefig("plots/credible/dune_ee_mm_test.pdf")


print("plotting COHERENT")
fig, ax = coherent.credible_grid((0,1,2,3,4,5), (0,0,0,0,0,0), namelist, nbins=40)
fig.savefig("plots/credible/xenon_summed_ud_coh-marg_prior.png")
fig.savefig("plots/credible/xenon_summed_ud_coh-marg_prior.pdf")

print("plotting Xenon UD")
fig, ax = xen_priorflow.credible_grid((0,1,2,3,4,5,6,7,8,9,10,11), (0,0,0,0,0,0,0,0,0,0,0,0), ud_namelist, nbins=30)
fig.savefig("plots/credible/xenon_ud_with_coh_prior_marginals.png")
fig.savefig("plots/credible/xenon_ud_with_coh_prior_marginals.pdf")


# LXe Atmospherics
print("plotting LXe atmospherics")
fig, ax = xe_atmos.credible_grid((0,1,2,3,4,5), (0,0,0,0,0,0), namelist_nsi, nbins=25)
fig.suptitle(r"Future-LXe Atmospheric Neutrino Appearance", x=0.6, y=0.8, fontsize=60)
fig.savefig("plots/credible/all_nsi_xenon_atmospherics.png")
fig.savefig("plots/credible/all_nsi_xenon_atmospherics.pdf")


# DUNE vs. Xe overlay
fig = plt.figure(figsize=(8,3))
fig.subplots_adjust(wspace=0.6)
gs = gridspec.GridSpec(1, 3)
ax = fig.add_subplot(gs[0, 0])
dune_atmos.credible_2d(idx=(0,1), credible_level=(0.6827, 0.9545), nbins=40, ax=ax, center=(0,0), color='r')
xe_atmos.credible_2d(idx=(0,1), credible_level=(0.6827, 0.9545), nbins=30, ax=ax, center=(0,0), color='b')
plt.xlabel(r"$\epsilon^u_{ee}$")
plt.ylabel(r"$\epsilon^u_{\mu\mu}$")
ax = fig.add_subplot(gs[0, 1])
dune_atmos.credible_2d(idx=(0,3), credible_level=(0.6827, 0.9545), nbins=40, ax=ax, center=(0,0), color='r')
xe_atmos.credible_2d(idx=(0,3), credible_level=(0.6827, 0.9545), nbins=40, ax=ax, center=(0,0), color='b')
plt.xlabel(r"$\epsilon^u_{ee}$")
plt.ylabel(r"$\epsilon^u_{e\mu}$")
ax = fig.add_subplot(gs[0, 2])
dune_atmos.credible_2d(idx=(3,4), credible_level=(0.6827, 0.9545), nbins=40, ax=ax, center=(0,0), color='r')
xe_atmos.credible_2d(idx=(3,4), credible_level=(0.6827, 0.9545), nbins=40, ax=ax, center=(0,0), color='b')
plt.xlabel(r"$\epsilon^u_{e\mu}$")
plt.ylabel(r"$\epsilon^u_{e\tau}$")
plt.tight_layout()

plt.savefig("plots/credible/dune_vs_xenon_overlay.png")
plt.savefig("plots/credible/dune_vs_xenon_overlay.pdf")
plt.clf()




# DUNE Beam
print("plotting dune beam")
fig, ax = dune_beam.credible_grid(idx, test_pt, namelist, nbins=40)
fig.suptitle(r"DUNE LBNE Beam $\nu_e$ Appearance (5 yr exposure)", x=0.6, y=0.8, fontsize=60)
fig.savefig("plots/credible/all_nsi_dune_beam_e_5yr.png")
fig.savefig("plots/credible/all_nsi_dune_beam_e_5yr.pdf")

# DUNE Complex Atmos
print("plotting complex dune atmos")
fig, ax = dune_atmos_complex.credible_grid(idx_complex, test_pt_complex, namelist_complex, nbins=40)
fig.suptitle(r"DUNE Atmospheric $\nu_\mu$ Appearance", x=0.6, y=0.8, fontsize=60)
fig.savefig("plots/credible/one_complex_nsi_dune_atmos.png")
fig.savefig("plots/credible/one_complex_nsi_dune_atmos.pdf")


# LXe Solars
print("plotting LXe solars")
fig, ax = xe_solars.credible_grid(idx_solar, test_pt_solar, solar_namelist_dcp, nbins=40)
fig.suptitle(r"$\epsilon^{e,L}_{\alpha\beta}, \epsilon^{e,R}_{\alpha\beta}$, Future-LXe Solar Neutrino Appearance",
             x=0.6, y=0.8,
             fontsize=80)
fig.savefig("plots/credible/all_nsi_xenon_solars.png")
fig.savefig("plots/credible/all_nsi_xenon_solars.pdf")



# DUNE Atmospheric
print("plotting DUNE Atmospherics")
fig, ax = dune_atmos.credible_grid(idx, test_pt, namelist, nbins=30)
fig.suptitle(r"DUNE Atmospheric $\nu_\mu$ Appearance", x=0.6, y=0.8, fontsize=60)
fig.savefig("plots/credible/all_nsi_dune_atmospherics_mu.png")
fig.savefig("plots/credible/all_nsi_dune_atmospherics_mu.pdf")

# Hyper-K Atmospheric
print("plotting Hyper-K Atmospherics")
fig, ax = hyperk.credible_grid(idx, test_pt, namelist, nbins=30)
fig.suptitle(r"Hyper-K Atmospheric $\nu_\mu$ Appearance", x=0.6, y=0.8, fontsize=60)
fig.savefig("plots/credible/all_nsi_hyper-k_atmospherics_mu.png")
fig.savefig("plots/credible/all_nsi_hyper-k_atmospherics_mu.pdf")



