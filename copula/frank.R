# Read in the data.
library(copula)
library(VineCopula)

coh <- read.table("coherent_ud_post_equal_weights.dat", header=FALSE)

# Generate plots. We rotate the down-type NSI so that the correlation is positive.
plot(coh[,1], -coh[,7])
plot(coh[,2], -coh[,8])
plot(coh[,3], -coh[,9])
plot(coh[,4], -coh[,10])
plot(coh[,5], -coh[,11])
plot(coh[,6], -coh[,12])

# Read in pseudo observations. Again, flip the values for d-type so that the fit will work.
z <- pobs(as.matrix(cbind(coh[,1],coh[,2],coh[,4],coh[,5],coh[,6],-coh[,7],-coh[,8],-coh[,10],-coh[,11],-coh[,12])))

# Estimate copula parameters
cop_model <- frankCopula(dim = 10)
fit <- fitCopula(cop_model, z, method = 'itau')
coef(fit)

##    param 
## ~1.488821

# Run goodness of fit.
gf <- gofCopula(frankCopula(dim = 10, param=1.488821), z, N = 50, estim.method = "itau")
gf


gf <- gofCopula(normalCopula(dim = 10), z, N = 50, estim.method = "itau")
gf
