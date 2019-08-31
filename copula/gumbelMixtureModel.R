library(copula)
library(VineCopula)

# Read in the sampled distribution (equal weights) and the sample probability (posterior).
samples <- read.table("multinest/all_nsi_vector_solar_equal_weights.txt", header=FALSE)
post <- read.table("multinest/all_nsi_vector_solar.txt", header=FALSE)
marginals <- read.table("multinest/xee_marginals.txt")

z <- pobs(as.matrix(cbind(samples[,1],samples[,2])))

plot(xee[,1], samples[,2])
plot(xee[,1], samples[,3])
plot(xee[,2], samples[,3])


# Set up the mixture.
mGG <- mixCopula(list(tCopula(0.7,dim=2), tCopula(-0.5,dim=2)))
mGG

# Set theta
getTheta(mGG, freeOnly = TRUE, attr = TRUE)
getTheta(mGG, named=TRUE)
copula:::isFree(mGG)

# Decide which parameters to fix.
fixedParam(mGG) <- fx <- c(FALSE, FALSE, FALSE, FALSE, TRUE, TRUE)
#stopifnot(identical(copula:::isFree(mGG), !fx))

fit <- fitCopula(mGG, data = z)
print( fit )
print( summary(fit) )



# Now compute the marginals of our original joint distribution and simulate the fitted copula.
df_post <- data.frame(ee = samples[,1], mm = samples[,2])
xee_marginals <- data.frame(ee = marginals[,1], eep = marginals[,2], mm = marginals[,3],
                            mmp = marginals[,4], tt = marginals[,5], ttp = marginals[,6])

# Define shape variables for observed marginals.
x_mean <- mean(xee_marginals$ee)
x_var <- var(xee_marginals$ee)
x_rate <- x_mean / x_var
x_shape <- ( (x_mean)^2 ) / x_var
y_mean <- mean(xee_marginals$mm)
y_var <- var(xee_marginals$mm)
y_rate <- y_mean / y_var
y_shape <- ( (y_mean)^2 ) / y_var

# Define marginals
library(distr)
fancy_ee <- DiscreteDistribution(supp = xee_marginals$ee , prob = xee_marginals$eep)
dfancy_ee <- d(fancy_ee)  ## Density function
pfancy_ee <- p(fancy_ee)  ## Distribution function
qfancy_ee <- q(fancy_ee)  ## Quantile function
rfancy_ee <- r(fancy_ee)  ## Random number generation
fancy_mm <- DiscreteDistribution(supp = xee_marginals$mm , prob = xee_marginals$mmp)
dfancy_mm <- d(fancy_mm)  ## Density function
pfancy_mm <- p(fancy_mm)  ## Distribution function
qfancy_mm <- q(fancy_mm)  ## Quantile function
rfancy_mm <- r(fancy_mm)  ## Random number generation

u <- rCopula(5000,mGG)
sim <- cbind(qfancy_ee(u[,1]), qfancy_mm(u[,2]))

# Draw from the copula.
# <- mvdc(frankCopula(param = 1.48, dim = 2), margins = c("fancy_ee","fancy_mm"), paramMargins = )
#sim <- rMvdc(my_dist, 1000)
plot(df_post$ee, df_post$mm, main = 'Test', col = "blue")

points(sim[,1], sim[,2], col = 'red')
