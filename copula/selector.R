# Copula family estimator.

# Copula fitting.
library(copula)
library(VineCopula)

xee <- read.table("all_nsi_vector_solar_equal_weights.txt", header=FALSE)

u <- pobs(as.matrix(cbind(xee[,1],xee[,2])))[,1]
v <- pobs(as.matrix(cbind(xee[,1],xee[,2])))[,2]

plot(xee[,1], xee[,2])

#selectedCopula <- BiCopSelect(u,v,familyset=NA)
#selectedCopula
