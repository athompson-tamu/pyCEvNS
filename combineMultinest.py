# Set up constants for Xenon.
n = 78
z = 54

# Define transformation matrices
u = [
  [1,0,0],
  [0, 2*z + n, 2*n + z],
  [1, 3, 3]
]

u_invs = [
  [1, 0, 0],
  [(2*n+z)/(3*(z-n)), 1/(z-n), (2*n+z)/(3*(n-z))],
  [(2*z+n)/(3*(n-z)), 1/(n-z), (2*z+n)/(3*(z-n))]
]


# Read in posterior distributions.
pheno_0 = np.genfromtxt("multinest/all_nsi_dune_beam_e_appearance", delimiter=',')
pheno_1 = np.genfromtxt("multinest/all_nsi_dune_beam_e_appearance", delimiter=',')
pheno_2 = np.genfromtxt("multinest/all_nsi_dune_beam_e_appearance", delimiter=',')



a = u_invs[1][0]
b = u_invs[1][1]
c = u_invs[1][2]

# Compute maxima and minima for each epsilon

# Perform the convolution integral.
for x in range(0, pheno_0.shape[0]):
  for y in range(0, pheno_1.shape[0]):
    for z in range(0, pheno_2.shape[0]):
      convolution = (1 / c) * pheno_0[x:0] * pheno_1[y:0] * pheno_2[z:0]
      eps = b*pheno_1[y:,2:9] - a*pheno_0[x:,2:9] - c*pheno_2[z:,2:9]
      posterior += convolution * np.histogramdd(eps, bins=10,
                                                range=((-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)))[0]
