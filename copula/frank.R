# Read in the data.
library(copula)
library(VineCopula)
library(distr)

# Read in coherent samples and marginals.
coh <- read.table("multinest/coherent_ud_post_equal_weights.dat", header=FALSE)
post = read.table("multinest/coh_marginals.txt", header=FALSE)
val = read.table("multinest/coh_values.txt", header=FALSE)
df_post <- data.frame(uee = coh[,1], dee = -coh[,7],
                      umm = coh[,2], dmm = -coh[,8],
                      uem = coh[,4], dem = -coh[,10],
                      uet = coh[,5], det = -coh[,11],
                      umt = coh[,6], dmt = -coh[,12])
marginals <- data.frame(uee = val[,1], ueep = post[,1],
                        dee = -val[,7], deep = post[,7],
                        umm = val[,2], ummp = post[,2],
                        dmm = -val[,8], dmmp = post[,8],
                        uem = val[,4], uemp = post[,4],
                        dem = -val[,10], demp = post[,10],
                        uet = val[,5], uetp = post[,5],
                        det = -val[,11], detp = post[,11],
                        umt = val[,6], umtp = post[,6],
                        dmt = -val[,12], dmtp = post[,12])

# Define marginals
# EE
uee <- DiscreteDistribution(supp = marginals$uee , prob = marginals$ueep)
q_uee <- q(uee)  ## Quantile function
dee <- DiscreteDistribution(supp = marginals$dee , prob = marginals$deep)
q_dee <- q(dee)  ## Quantile function
# MM
umm <- DiscreteDistribution(supp = marginals$umm , prob = marginals$ummp)
q_umm <- q(umm)  ## Quantile function
dmm <- DiscreteDistribution(supp = marginals$dmm , prob = marginals$dmmp)
q_dmm <- q(dmm)  ## Quantile function
# EM
uem <- DiscreteDistribution(supp = marginals$uem , prob = marginals$uemp)
q_uem <- q(uem)  ## Quantile function
dem <- DiscreteDistribution(supp = marginals$dem , prob = marginals$demp)
q_dem <- q(dem)  ## Quantile function
# ET
uet <- DiscreteDistribution(supp = marginals$uet , prob = marginals$uetp)
q_uet <- q(uet)  ## Quantile function
det <- DiscreteDistribution(supp = marginals$det , prob = marginals$detp)
q_det <- q(det)  ## Quantile function
# MT
umt <- DiscreteDistribution(supp = marginals$umt , prob = marginals$umtp)
q_umt <- q(umt)  ## Quantile function
dmt <- DiscreteDistribution(supp = marginals$dmt , prob = marginals$dmtp)
q_dmt <- q(dmt)  ## Quantile function

# Read in pseudo observations.
zee <- pobs(as.matrix(cbind(df_post$uee, df_post$dee)))
zmm <- pobs(as.matrix(cbind(df_post$umm, df_post$dmm)))
zem <- pobs(as.matrix(cbind(df_post$uem, df_post$dem)))
zet <- pobs(as.matrix(cbind(df_post$uet, df_post$det)))
zmt <- pobs(as.matrix(cbind(df_post$umt, df_post$dmt)))

# Estimate copula parameters
cop_model <- frankCopula(dim = 2)
fit <- fitCopula(cop_model, zmt, method = 'itau')
coef(fit)

##    param 
## -1.469658

# Build the fitted copula.
fitEE <- frankCopula(param = 7.03, dim = 2)
fitMM <- frankCopula(param = 15.3, dim = 2)
fitEM <- frankCopula(param = 18.6, dim = 2)
fitET <- frankCopula(param = 7.387, dim = 2)
fitMT <- frankCopula(param = 17.819, dim = 2)

cdf_ee <- rCopula(5000,fitEE)
sim_ee <- cbind(q_uee(cdf_ee[,1]), q_dee(cdf_ee[,2]))

cdf_mm <- rCopula(5000,fitMM)
sim_mm <- cbind(q_umm(cdf_mm[,1]), q_dmm(cdf_mm[,2]))

cdf_em <- rCopula(5000,fitEM)
sim_em <- cbind(q_uem(cdf_em[,1]), q_dem(cdf_em[,2]))

cdf_et <- rCopula(5000,fitET)
sim_et <- cbind(q_uet(cdf_et[,1]), q_det(cdf_et[,2]))

cdf_mt <- rCopula(5000,fitMT)
sim_mt <- cbind(q_umt(cdf_mt[,1]), q_dmt(cdf_mt[,2]))

# Draw from the copula.
plot(df_post$uee, df_post$dee, main = '', col = "blue")
points(sim_ee[,1], sim_ee[,2], col = 'red')

plot(df_post$umm, df_post$dmm, main = '', col = "blue")
points(sim_mm[,1], sim_mm[,2], col = 'red')

plot(df_post$uem, df_post$dem, main = '', col = "blue")
points(sim_em[,1], sim_em[,2], col = 'red')

plot(df_post$uet, df_post$det, main = '', col = "blue")
points(sim_et[,1], sim_et[,2], col = 'red')

plot(df_post$umt, df_post$dmt, main = '', col = "blue")
points(sim_mt[,1], sim_mt[,2], col = 'red')



# Run goodness of fit.
gf <- gofCopula(fitEE, zee, N = 50, estim.method = "itau")
gf

gf <- gofCopula(fitMM, zmm, N = 50, estim.method = "itau")
gf

gf <- gofCopula(fitEM, zem, N = 50, estim.method = "itau")
gf

gf <- gofCopula(fitET, zet, N = 50, estim.method = "itau")
gf

gf <- gofCopula(fitMT, zmt, N = 50, estim.method = "itau")
gf
# Compare to normal
gf <- gofCopula(normalCopula(dim = 2), z, N = 50, estim.method = "itau")
gf


