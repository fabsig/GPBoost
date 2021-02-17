library(gpboost)

#############################################################
# This script contains various examples on how to do inference and prediction for
#   (i) grouped (or clustered) random effects models
#   (ii) Gaussian process (GP) models
#   (iii) models that combine GP and grouped random effects
# and on how to save models
#############################################################

# --------------------Simulate data----------------
n <- 1000 # number of samples
# Simulate single level grouped random effects data
m <- 100 # number of categories / levels for grouping variable
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
sigma2_1 <- 1^2 # random effect variance
sigma2 <- 0.5^2 # error variance
set.seed(1)
b <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
eps <- b[group]
xi <- sqrt(sigma2) * rnorm(n) # simulate error term
y <- eps + xi # observed data for single level grouped random effects model
# Simulate data for linear mixed effects model
X <- cbind(rep(1,n),runif(n)) # design matrix / covariate data for fixed effects
beta <- c(3,3) # regression coefficients
y_lin <- eps + xi + X%*%beta # add fixed effects to observed data
# Simulate data for two crossed random effects and a random slope
x <- runif(n) # covariate data for random slope
n_obs_gr <- n/m # number of samples per group
group_crossed <- rep(1,n) # grouping variable for second crossed random effect
for(i in 1:m) group_crossed[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
sigma2_2 <- 0.5^2 # variance of second random effect
sigma2_3 <- 0.75^2 # variance of random slope for first random effect
b_crossed <- sqrt(sigma2_2) * rnorm(n_obs_gr) # second random effect
b_random_slope <- sqrt(sigma2_3) * rnorm(m) # simulate random effects
y_crossed_random_slope <- b[group] + # observed data = sum of all random effects
  b_crossed[group_crossed] + x * b_random_slope[group] + xi
# Simulate data for two nested random effects
m_nested <- 200 # number of categories / levels for the second nested grouping variable
group_nested <- rep(1,n)  # grouping variable for nested lower level random effects
for(i in 1:m_nested) group_nested[((i-1)*n/m_nested+1):(i*n/m_nested)] <- i
b_nested <- 1. * rnorm(m_nested) # nested lower level random effects
y_nested <- b[group] + b_nested[group_nested] + xi # observed data


#--------------------Grouped random effects model: single-level random effect----------------
# --------------------Training----------------
gp_model <- GPModel(group_data = group) # Create random effects model
fit(gp_model, y = y, params = list(std_dev = TRUE)) # Fit model
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(group_data = group, y = y, params = list(std_dev = TRUE))
summary(gp_model)
# Use other optimization technique (gradient descent with Nesterov acceleration 
#   instead of Fisher scoring) and monitor convergence of optimization (trace=TRUE)
gp_model <- fitGPModel(group_data = group, y = y,
                       params = list(optimizer_cov = "gradient_descent",
                                     lr_cov = 0.1, use_nesterov_acc = TRUE,
                                     maxit = 100, std_dev = TRUE, trace=TRUE))
summary(gp_model)

# Evaluate negative log-likelihood
gp_model$neg_log_likelihood(cov_pars=c(sigma2,sigma2_1),y=y)

# Do optimization using optim and e.g. Nelder-Mead
gp_model <- GPModel(group_data = group)
optim(par=c(1,1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")

# --------------------Prediction----------------
gp_model <- fitGPModel(group_data = group, y = y, params = list(std_dev = TRUE))
group_test <- 1:m
pred <- predict(gp_model, group_data_pred = group_test)
# Compare true and predicted random effects
plot(b, pred$mu, xlab="truth", ylab="predicted",
     main="Comparison of true and predicted random effects")
abline(a=0,b=1)
# Also predict covariance matrix
group_test = c(1,1,2,2,-1,-1)
pred <- predict(gp_model, group_data_pred = group_test, predict_cov_mat = TRUE)
pred$mu# Predicted mean
pred$cov# Predicted covariance

#--------------------Saving a GPModel and loading it from a file----------------
gp_model <- fitGPModel(group_data = group, y = y)
group_test = c(1,1,2,2,-1,-1)
pred <- predict(gp_model, group_data_pred = group_test, predict_var = TRUE)
# Save model to file
filename <- tempfile(fileext = ".json")
saveGPModel(gp_model,filename = filename)
# Load from file and make predictions again
gp_model_loaded <- loadGPModel(filename = filename)
pred_loaded <- predict(gp_model_loaded, group_data_pred = group_test, predict_var = TRUE)
# Check equality
pred$mu - pred_loaded$mu
pred$var - pred_loaded$var

#--------------------Mixed effects model: random effects and linear fixed effects----------------
# Create random effects model
gp_model <- GPModel(group_data = group)
# Fit model
fit(gp_model, y = y_lin, X = X, params = list(std_dev = TRUE))
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(group_data = group,
                       y = y_lin, X = X, params = list(std_dev = TRUE))
summary(gp_model)

#--------------------Two crossed random effects and a random slope----------------
# Create random effects model
gp_model <- GPModel(group_data = cbind(group,group_crossed),
                    group_rand_coef_data = x,
                    ind_effect_group_rand_coef = 1)# indicate that the random slope is for the first random effect
# Fit model
fit(gp_model, y = y_crossed_random_slope, params = list(std_dev = TRUE))
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(group_data = cbind(group,group_crossed),
                       group_rand_coef_data = x,
                       ind_effect_group_rand_coef = 1,
                       y = y_crossed_random_slope, params = list(std_dev = TRUE))
summary(gp_model)

# --------------------Two nested random effects----------------
# Estimate model
group_data <- cbind(group, group_nested)
gp_model <- fitGPModel(group_data=group_data, y=y_nested, params = list(std_dev = TRUE))
summary(gp_model)


#--------------------Gaussian process model----------------
library(gpboost)
#--------------------Simulate data----------------
n <- 200 # number of samples
set.seed(2)
coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
sigma2_1 <- 1^2 # marginal variance of GP
rho <- 0.1 # range parameter
sigma2 <- 0.5^2 # error variance
D <- as.matrix(dist(coords))
Sigma <- sigma2_1*exp(-D/rho)+diag(1E-20,n)
# Sigma <- sigma2_1*exp(-(D/rho)^2)+diag(1E-20,n)# different covariance function
C <- t(chol(Sigma))
b_1 <- rnorm(n) # simulate random effect
xi <- sqrt(sigma2) * rnorm(n) # simulate error term
y <- C %*% b_1 + xi
# Add linear regression term
X <- cbind(rep(1,n),runif(n)) # design matrix / covariate data for fixed effect
beta <- c(3,3) # regression coefficients
y_lin <- C %*% b_1 + xi + X%*%beta # add fixed effect to observed data
# Simulate spatially varying coefficient (random coefficient) model data
Z_SVC <- cbind(runif(n),runif(n)) # covariate data for random coeffients
colnames(Z_SVC) <- c("var1","var2")
# simulate SVC random effect
b_2 <- rnorm(n)
b_3 <- rnorm(n)
# Note: for simplicity, we assume that all GPs have the same covariance parameters
y_svc <- C %*% b_1 + Z_SVC[,1] * C %*% b_2 + Z_SVC[,2] * C %*% b_3 + xi

#--------------------Training----------------
# Create Gaussian process model
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
## Other covariance functions:
# gp_model <- GPModel(gp_coords = coords, cov_function = "gaussian")
# gp_model <- GPModel(gp_coords = coords,
#                     cov_function = "matern", cov_fct_shape=1.5)
# gp_model <- GPModel(gp_coords = coords,
#                     cov_function = "powered_exponential", cov_fct_shape=1.1)
# Fit model
fit(gp_model, y = y, params = list(std_dev = TRUE))
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                       y = y, params = list(std_dev = TRUE))
summary(gp_model)

# Use other optimization technique: Nesterov accelerated gradient descent instead of Fisher scoring
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", y = y,
                       params = list(optimizer_cov = "gradient_descent",
                                     lr_cov = 0.05, use_nesterov_acc = TRUE,
                                     std_dev = TRUE))
summary(gp_model)

# Evaluate negative log-likelihood
gp_model$neg_log_likelihood(cov_pars=c(sigma2,sigma2_1,rho),y=y)

# Do optimization using optim and e.g. Nelder-Mead
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
optim(par=c(1,1,0.2), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")

#--------------------Prediction----------------
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                       y = y, params = list(std_dev = TRUE))
set.seed(1)
ntest <- 5
# prediction locations (=features) for Gaussian process
coords_test <- cbind(runif(ntest),runif(ntest))/10
pred <- predict(gp_model, gp_coords_pred = coords_test,
                predict_cov_mat = TRUE)
print("Predicted (posterior/conditional) mean of GP")
pred$mu
print("Predicted (posterior/conditional) covariance matrix of GP")
pred$cov

#--------------------Gaussian process model with linear mean function----------------
# Include a liner regression term instead of assuming a zero-mean a.k.a. "universal Kriging"
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                       y = y_lin, X=X, params = list(std_dev = TRUE))
summary(gp_model)

#--------------------Gaussian process model with Vecchia approximation----------------
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", 
                       vecchia_approx = TRUE, num_neighbors = 30, y = y)
summary(gp_model)

#--------------------Gaussian process model with random coefficents----------------
# Estimate model
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                       gp_rand_coef_data = Z_SVC,
                       y = y_svc, params = list(std_dev = TRUE))
summary(gp_model)
# Note: this is a small sample size for this type of model
#   -> covariance parameters estimates can have high variance


#--------------------GP model with two independent observations of the GP----------------
# Simulate data
n <- 200 # number of samples per cluster
set.seed(1)
coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
coords <- rbind(coords,coords) # locations for second observation of GP (same locations)
# indices that indicate the GP sample to which an observations belong
cluster_ids <- c(rep(1,n),rep(2,n))
sigma2_1 <- 1^2 # marginal variance of GP
rho <- 0.1 # range parameter
sigma2 <- 0.5^2 # error variance
D <- as.matrix(dist(coords[1:n,]))
Sigma <- sigma2_1*exp(-D/rho)+diag(1E-20,n)
C <- t(chol(Sigma))
b_1 <- rnorm(2 * n) # simulate random effect
eps <- c(C %*% b_1[1:n], C %*% b_1[1:n + n])
xi <- sqrt(sigma2) * rnorm(2 * n) # simulate error term
y <- eps + xi

# Create Gaussian process model
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                    cluster_ids = cluster_ids)
# Fit model
fit(gp_model, y = y, params = list(std_dev = TRUE))
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                       cluster_ids = cluster_ids,
                       y = y, params = list(std_dev = TRUE))
summary(gp_model)


#--------------------Combine Gaussian process with grouped random effects----------------
# Simulate data
n <- 200 # number of samples
m <- 25 # number of categories / levels for grouping variable
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
set.seed(1)
coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
sigma2_1 <- 1^2 # random effect variance
sigma2_2 <- 1^2 # marginal variance of GP
rho <- 0.1 # range parameter
sigma2 <- 0.5^2 # error variance
# incidence matrix relating grouped random effects to samples
Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
set.seed(156)
b1 <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
D <- as.matrix(dist(coords))
Sigma <- sigma2_2*exp(-D/rho)+diag(1E-20,n)
C <- t(chol(Sigma))
b_2 <- rnorm(n) # simulate random effect
eps <- Z1 %*% b1 + C %*% b_2
xi <- sqrt(sigma2) * rnorm(n) # simulate error term
y <- eps + xi

# Create Gaussian process model
gp_model <- GPModel(group_data = group,
                    gp_coords = coords, cov_function = "exponential")
# Fit model
fit(gp_model, y = y, params = list(std_dev = TRUE))
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(group_data = group,
                       gp_coords = coords, cov_function = "exponential",
                       y = y, params = list(std_dev = TRUE))
summary(gp_model)
