# Read in the data.
library(copula)
library(VineCopula)

coh <- read.table("coherent_ud_post_equal_weights.dat", header=FALSE)

#plot(coh[,1], 1-coh[,7])

# Read in pseudo observations.
z <- pobs(as.matrix(cbind(coh[,1],coh[,2],coh[,4],coh[,5],coh[,6],coh[,7],coh[,8],coh[,10],coh[,11],coh[,12])))

# Estimate copula parameters
cop_model <- frankCopula(dim = 10)
#m <- pobs(as.matrix(z))
fit <- fitCopula(cop_model, z, method = 'itau')
coef(fit)

##    param 
## -6.647493

# Run goodness of fit.
gf <- gofCopula(frankCopula(dim = 10, param=-1.469), z, N = 50, estim.method = "itau")
gf


gf <- gofCopula(normalCopula(dim = 10), z, N = 50, method = "itau")
gf

## 	Parametric bootstrap goodness-of-fit test with 'method'="Sn", 'estim.method'="mpl"
## 
## data:  x
## statistic = 0.29449, parameter = 0.59983, p-value = 0.009804
