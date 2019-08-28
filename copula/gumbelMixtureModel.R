library(copula)
library(VineCopula)

# Read in the sampled distribution (equal weights) and the sample probability (posterior).
samples <- read.table("multinest/all_nsi_vector_solar_equal_weights.txt", header=FALSE)
post <- read.table("multinest/all_nsi_vector_solar.txt", header=FALSE)

z <- pobs(as.matrix(cbind(samples[,1],samples[,2])))

plot(xee[,1], samples[,2])
plot(xee[,1], samples[,3])
plot(xee[,2], samples[,3])


# Set up the mixture.
mGG <- mixCopula(list(gumbelCopula(5), frankCopula(2)))
mGG

# Set theta
getTheta(mGG, freeOnly = TRUE, attr = TRUE)
getTheta(mGG, named=TRUE)
copula:::isFree(mGG)

# Decide which parameters to fix.
fixedParam(mGG) <- fx <- c(FALSE, FALSE, FALSE, FALSE)
#stopifnot(identical(copula:::isFree(mGG), !fx))

fit <- fitCopula(mGG, data = z)
print( fit )
print( summary(fit) )


# Now compute the marginals of our original joint distribution and simulate the fitted copula.
df_post <- data.frame(p = post[,1], ee = post[,3])
b <- seq(-0.5, 0.5, length.out=50)
df_post$cdf <- cumsum(post[,1])
df_post$grp <- cut(ee_cdf, b)



