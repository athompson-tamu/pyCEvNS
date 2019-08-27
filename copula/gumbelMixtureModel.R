# Copula fitting.
library(copula)
library(VineCopula)

xee <- read.table("all_nsi_vector_solar_equal_weights.txt", header=FALSE)

u <- pobs(as.matrix(cbind(xee[,1],xee[,2])))[,1]
v <- pobs(as.matrix(cbind(xee[,1],xee[,2])))[,2]

plot(xee[,1], xee[,2])

#selectedCopula <- BiCopSelect(u,v,familyset=NA)
#selectedCopula

gumbel.cop <- archmCopula("gumbel", 2)
set.seed(500)
m <- pobs(as.matrix(cbind(xee[,1],xee[,2])))
fit <- fitCopula(gumbel.cop,m,method='ml')
alpha <- coef(fit)

copula_dist <- mvdc(copula=gumbelCopula(alpha,dim=2), margins=c(xee[,1],xee[,2]),
                    paramMargins=list(list(mean=xee[,1], sd=xee[,2]),
                                      list(mean=xee[,1], sd=xee[,2])))
sim <- rmvdc(copula_dist, 3965)
