#############################################################
# Examples on how to do inference and prediction for generalized linear 
# mixed effects models with various likelihoods:
#   - "gaussian" (=regression)
#   - "bernoulli" (=classification)
#   - "poisson" and "gamma" (=Poisson and gamma regression)
# and various random effects models:
#   - grouped (aka clustered) random effects models including random slopes
#   - Gaussian process (GP) models
#   - combined GP and grouped random effects
# 
# Author: Fabio Sigrist
#############################################################

library(gpboost)

simulate_response_variable <- function (lp, rand_eff, likelihood) {
  ## Function that simulates response variable for various likelihoods
  n <- length(rand_eff)
  if (likelihood == "gaussian") {
    xi <- 0.25 * rnorm(n) # error term
    y <- lp + rand_eff + xi
  } else if (likelihood == "bernoulli_probit") {
    probs <- pnorm(lp + rand_eff)
    y <- as.numeric(runif(n) < probs)
  } else if (likelihood == "bernoulli_logit") {
    probs <- 1/(1+exp(-(lp + rand_eff)))
    y <- as.numeric(runif(n) < probs)
  } else if (likelihood == "poisson") {
    mu <- exp(lp + rand_eff)
    y <- qpois(runif(n), lambda = mu)
  } else if (likelihood == "gamma") {
    mu <- exp(lp + rand_eff)
    y <- qgamma(runif(n), scale = mu, shape = 1)
  }
  return(y)
}

# Choose likelihood: either "gaussian" (=regression), 
#                     "bernoulli_probit", "bernoulli_logit", (=classification)
#                     "poisson", or "gamma"
likelihood <- "gaussian"

#################################
# Grouped random effects
#################################
# --------------------Simulate data----------------
# Single-level grouped random effects
n <- 1000 # number of samples
m <- 100 # number of categories / levels for grouping variable
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
set.seed(1)
b <- 1 * rnorm(m) # simulate random effects
rand_eff <- b[group]
rand_eff <- rand_eff - mean(rand_eff)
# Simulate linear regression fixed effects
X <- cbind(rep(1,n),runif(n)-0.5) # design matrix / covariate data for fixed effects
beta <- c(0,3) # regression coefficients
lp <- X %*% beta
y <- simulate_response_variable(lp=lp, rand_eff=rand_eff, likelihood=likelihood)
hist(y, breaks=20)  # visualize response variable
# Crossed grouped random effects and a random slope
x <- runif(n) # covariate data for random slope
n_obs_gr <- n/m # number of samples per group
group_crossed <- rep(1,n) # grouping variable for second crossed random effect
for(i in 1:m) group_crossed[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
b_crossed <- 0.5 * rnorm(n_obs_gr) # second random effect
b_random_slope <- 0.75 * rnorm(m) # simulate random effects
rand_eff <- b[group] + b_crossed[group_crossed] + x * b_random_slope[group]
rand_eff <- rand_eff - mean(rand_eff)
y_crossed_random_slope <- simulate_response_variable(lp=lp, rand_eff=rand_eff, likelihood=likelihood)
# Nested grouped random effects
group_inner <- rep(1,n)  # grouping variable for nested lower level random effects
for(i in 1:m) {
  group_inner[((i-1)*n/m+1):((i-0.5)*n/m)] <- 1
  group_inner[((i-0.5)*n/m + 1):((i)*n/m)] <- 2
}
group_nested <- get_nested_categories(group, group_inner)
b_nested <- 1. * rnorm(length(group_nested)) # nested lower level random effects
rand_eff <- b[group] + b_nested[group_nested]
rand_eff <- rand_eff - mean(rand_eff)
y_nested <- simulate_response_variable(lp=lp, rand_eff=rand_eff, likelihood=likelihood)

# --------------------Training----------------
gp_model <- fitGPModel(group_data = group, y = y, X = X, likelihood = likelihood)
summary(gp_model)
# Get coefficients and variance/covariance parameters separately
gp_model$get_coef()
gp_model$get_cov_pars()
# Obtaining standard deviations and p-values for fixed effects coefficients ('std_dev = TRUE')
gp_model <- fitGPModel(group_data = group, y = y, X = X, likelihood = likelihood,
                       params = list(std_dev = TRUE))
summary(gp_model)

# Optional arguments for the 'params' argument of the 'fit' function:
# - monitoring convergence: 'trace = TRUE'
# - calculate standard deviations: 'std_dev = TRUE'
# - change optimization algorithm options (see below)
# For available optimization options, see
#   https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#optimization-parameters
# gp_model <- fitGPModel(group_data = group, y = y, X = X,
#                        params = list(trace=TRUE,
#                                      std_dev = TRUE,
#                                      optimizer_cov= "gradient_descent",
#                                      lr_cov = 0.1, use_nesterov_acc = TRUE, maxit = 100))

# --------------------Prediction----------------
group_test <- c(1,2,-1)
X_test <- cbind(rep(1,length(group_test)),runif(length(group_test)))
# Predict latent variable
pred <- predict(gp_model, X_pred = X_test, group_data_pred = group_test,
                predict_var = TRUE, predict_response = FALSE)
pred$mu # Predicted latent mean
pred$var # Predicted latent variance
# Predict response variable (for Gaussian data, latent and response variable predictions are the same)
pred_resp <- predict(gp_model, X_pred = X_test, group_data_pred = group_test,
                     predict_var = TRUE, predict_response = TRUE)
pred_resp$mu # Predicted response variable (label)
pred_resp$var # Predicted variance of response variable

# --------------------Predict ("estimate") training data random effects----------------
# The following shows how to obtain predicted (="estimated") random effects for the training data
all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
# The function 'predict_training_data_random_effects' returns predicted random effects for all data points.
# Unique random effects for every group can be obtained as follows
first_occurences <- match(unique(group), group)
training_data_random_effects <- all_training_data_random_effects[first_occurences] 
# Compare true and predicted random effects
plot(b, training_data_random_effects, xlab="truth", ylab="predicted",
     main="Comparison of true and predicted random effects")
# Adding the overall intercept gives the group-wise intercepts
group_wise_intercepts <- gp_model$get_coef()[1] + training_data_random_effects
# Alternatively, this can also be done as follows
# group_unique <- unique(group)
# X_zero <- cbind(rep(0,length(group_unique)),rep(0,length(group_unique)))
# pred_random_effects <- predict(gp_model, group_data_pred = group_unique, X_pred = X_zero)
# sum(abs(training_data_random_effects - pred_random_effects$mu))

#--------------------Saving a GPModel and loading it from a file----------------
# Save model to file
filename <- tempfile(fileext = ".json")
saveGPModel(gp_model,filename = filename)
# Load from file and make predictions again
gp_model_loaded <- loadGPModel(filename = filename)
pred_loaded <- predict(gp_model_loaded, group_data_pred = group_test, 
                       X_pred = X_test, predict_var = TRUE, predict_response = FALSE)
pred_resp_loaded <- predict(gp_model_loaded, group_data_pred = group_test, 
                            X_pred = X_test, predict_var = TRUE, predict_response = TRUE)
# Check equality
sum(abs(pred$mu - pred_loaded$mu))
sum(abs(pred$var - pred_loaded$var))
sum(abs(pred_resp$mu - pred_resp_loaded$mu))
sum(abs(pred_resp$var - pred_resp_loaded$var))

#--------------------Two crossed random effects and a random slope----------------
gp_model <- fitGPModel(group_data = cbind(group,group_crossed), group_rand_coef_data = x,
                       ind_effect_group_rand_coef = 1, likelihood = likelihood,
                       y = y_crossed_random_slope, X = X, params = list(std_dev = TRUE))
# 'ind_effect_group_rand_coef = 1' indicates that the random slope is for the first random effect
summary(gp_model)
# Prediction
pred <- predict(gp_model, group_data_pred=cbind(group,group_crossed), 
                group_rand_coef_data_pred=x, X_pred=X)

# Obtain predicted (="estimated") random effects for the training data
all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
# The function 'predict_training_data_random_effects' returns predicted random effects for all data points.
# Unique random effects for every group can be obtained as follows
first_occurences_1 <- match(unique(group), group)
first_occurences_2 <- match(unique(group_crossed), group_crossed)
pred_random_effects <- all_training_data_random_effects[first_occurences_1,1]
pred_random_slopes <- all_training_data_random_effects[first_occurences_1,3]
pred_random_effects_crossed <- all_training_data_random_effects[first_occurences_2,2]
# Compare true and predicted random effects
plot(b, pred_random_effects, xlab="truth", ylab="predicted",
     main="Comparison of true and predicted random effects")
points(b_random_slope, pred_random_slopes, col=2, pch=2, lwd=1.5)
points(b_crossed, pred_random_effects_crossed, col=4, pch=4, lwd=1.5)

# Random slope model in which an intercept random effect is dropped / not included
gp_model <- fitGPModel(group_data = cbind(group,group_crossed), group_rand_coef_data = x,
                       ind_effect_group_rand_coef = 1, drop_intercept_group_rand_effect = c(TRUE,FALSE),
                       likelihood = likelihood, y = y_crossed_random_slope, X = X, 
                       params = list(std_dev = TRUE))
# 'drop_intercept_group_rand_effect = c(TRUE,FALSE)' indicates that the first categorical variable 
#   in group_data has no intercept random effect
summary(gp_model)

# --------------------Two nested random effects----------------
# First create nested random effects variable
group_nested <- get_nested_categories(group, group_inner)
group_data <- cbind(group, group_nested)
gp_model <- fitGPModel(group_data = group_data, y = y_nested, 
                       likelihood = likelihood, params = list(std_dev = TRUE))
summary(gp_model)

# --------------------Using cluster_ids for independent realizations of random effects----------------
cluster_ids = rep(0,n)
cluster_ids[(n/2+1):n] = 1
gp_model <- fitGPModel(group_data = group, y = y, cluster_ids = cluster_ids, 
                       likelihood = likelihood, params = list(std_dev = TRUE))
summary(gp_model)
#Note: gives sames result in this example as when not using cluster_ids
#   since the random effects of different groups are independent anyway

#--------------------Evaluate negative log-likelihood and do optimization using optim----------------
gp_model <- GPModel(group_data = group, likelihood = likelihood)
if (likelihood == "gaussian") {
  init_cov_pars <- c(1,1)
} else {
  init_cov_pars <- 1
}
eval_nll <- function(pars, gp_model, y, X, likelihood) {
  if (likelihood == "gaussian") {
    coef <- pars[-c(1,2)]
    cov_pars <- exp(pars[c(1,2)])
  } else {
    coef <- pars[-1]
    cov_pars <- exp(pars[1])
  }
  fixed_effects <- as.numeric(X%*%coef)
  if (likelihood == "gaussian") {
    y <- y - fixed_effects
    fixed_effects <- NULL
  }
  gp_model$neg_log_likelihood(cov_pars=cov_pars, y=y, fixed_effects=fixed_effects)
}
pars <- c(init_cov_pars, rep(0,dim(X)[2]))
eval_nll(pars = pars, gp_model = gp_model, X = X, y=y, likelihood = likelihood)
# Do optimization using optim and e.g. Nelder-Mead
opt <- optim(par = pars, fn = eval_nll, gp_model = gp_mod, y = y, X = X, 
             likelihood = likelihood, method = "Nelder-Mead")


#################################
# Gaussian processes
#################################
#--------------------Simulate data----------------
ntrain <- 500 # number of training samples
set.seed(1)
# training and test locations (=features) for Gaussian process
coords_train <- matrix(runif(2)/2,ncol=2)
# exclude upper right corner
while (dim(coords_train)[1]<ntrain) {
  coord_i <- runif(2) 
  if (!(coord_i[1]>=0.6 & coord_i[2]>=0.6)) {
    coords_train <- rbind(coords_train,coord_i)
  }
}
nx <- 30 # test data: number of grid points on each axis
x2 <- x1 <- rep((1:nx)/nx,nx)
for(i in 1:nx) x2[((i-1)*nx+1):(i*nx)]=i/nx
coords_test <- cbind(x1,x2)
coords <- rbind(coords_train, coords_test)
ntest <- nx * nx
n <- ntrain + ntest
# Simulate spatial Gaussian process
sigma2_1 <- 1 # marginal variance of GP
rho <- 0.1 # range parameter
D <- as.matrix(dist(coords))
Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
C <- t(chol(Sigma))
b_1 <- as.vector(C %*% rnorm(n=n))
b_1 <- b_1 - mean(b_1)
y <- simulate_response_variable(lp=0, rand_eff=b_1, likelihood=likelihood)
# Split into training and test data
y_train <- y[1:ntrain]
y_test <- y[1:ntest+ntrain]
b_1_train <- b_1[1:ntrain]
b_1_test <- b_1[1:ntest+ntrain]
hist(y_train,breaks=50)# visualize response variable
# Including linear regression fixed effects
X <- cbind(rep(1,ntrain),runif(ntrain)-0.5) # design matrix / covariate data for fixed effects
beta <- c(0,3) # regression coefficients
lp <- X %*% beta
y_lin <- simulate_response_variable(lp=lp, rand_eff=b_1_train, likelihood=likelihood)
# Spatially varying coefficient (random coefficient) model
Z_SVC <- cbind(runif(ntrain),runif(ntrain)) # covariate data for random coefficients
colnames(Z_SVC) <- c("var1","var2")
# simulate SVC GP
b_2 <- C[1:ntrain, 1:ntrain] %*% rnorm(ntrain)
b_3 <- C[1:ntrain, 1:ntrain] %*% rnorm(ntrain)
# Note: for simplicity, we assume that all GPs have the same covariance parameters
rand_eff <- b_1_train + Z_SVC[,1] * b_2 + Z_SVC[,2] * b_3
rand_eff <- rand_eff - mean(rand_eff)
y_svc <- simulate_response_variable(lp=0, rand_eff=rand_eff, likelihood=likelihood)

#--------------------Training----------------
gp_model <- fitGPModel(gp_coords = coords_train, cov_function = "exponential",
                       likelihood = likelihood, y = y_train)
summary(gp_model)

## Other covariance functions:
# gp_model <- fitGPModel(gp_coords = coords, cov_function = "gaussian",
#                        likelihood = likelihood, y = y_train)
# gp_model <- fitGPModel(gp_coords = coords,
#                        cov_function = "matern", cov_fct_shape=1.5,
#                        likelihood = likelihood, y = y_train)
# gp_model <- fitGPModel(gp_coords = coords,
#                        cov_function = "powered_exponential", cov_fct_shape=1.1,
#                        likelihood = likelihood, y = y_train)

# Optional arguments for the 'params' argument of the 'fit' function:
# - monitoring convergence: trace=TRUE
# - obtain standard deviations: std_dev = TRUE
# - change optimization algorithm options (see below)
# For available optimization options, see
#   https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#optimization-parameters
# gp_model <- fitGPModel(group_data = group, y = y, X = X,
#                        params = list(trace=TRUE,
#                                      std_dev = TRUE,
#                                      optimizer_cov= "gradient_descent",
#                                      lr_cov = 0.1, use_nesterov_acc = TRUE, maxit = 100))

#--------------------Prediction----------------
# Prediction of latent variable
pred <- predict(gp_model, gp_coords_pred = coords_test,
                predict_var = TRUE, predict_response = FALSE)
# Predict response variable (label)
pred_resp <- predict(gp_model, gp_coords_pred = coords_test,
                     predict_var = TRUE, predict_response = TRUE)
if (likelihood %in% c("bernoulli_probit","bernoulli_logit")) {
  print("Test error:")
  mean(as.numeric(pred_resp$mu>0.5) != y_test)
} else {
  print("Test root mean square error:")
  sqrt(mean((pred_resp$mu - y_test)^2))
}

# Visualize predictions and compare to true values
library(ggplot2)
library(viridis)
library(gridExtra)
plot1 <- ggplot(data = data.frame(s_1=coords_test[,1],s_2=coords_test[,2],b=b_1_test),aes(x=s_1,y=s_2,color=b)) +
  geom_point(size=4, shape=15) + scale_color_viridis(option = "B") + 
  ggtitle("True latent GP and training locations") + 
  geom_point(data = data.frame(s_1=coords_train[,1],s_2=coords_train[,2],y=y_train),aes(x=s_1,y=s_2),size=3, col="white", alpha=1, shape=43)
plot2 <- ggplot(data = data.frame(s_1=coords_test[,1],s_2=coords_test[,2],b=pred$mu),aes(x=s_1,y=s_2,color=b)) +
  geom_point(size=4, shape=15) + scale_color_viridis(option = "B") + ggtitle("Predicted latent GP mean")
plot3 <- ggplot(data = data.frame(s_1=coords_test[,1],s_2=coords_test[,2],b=sqrt(pred$var)),aes(x=s_1,y=s_2,color=b)) +
  geom_point(size=4, shape=15) + scale_color_viridis(option = "B") + 
  labs(title="Predicted latent GP standard deviation", subtitle=" = prediction uncertainty")
grid.arrange(plot1, plot2, plot3, ncol=2)

# Predict latent GP at training data locations (=smoothing)
GP_smooth <- predict_training_data_random_effects(gp_model)
# Compare true and predicted random effects
plot(b_1_train, GP_smooth, xlab="truth", ylab="predicted",
     main="Comparison of true and predicted random effects")
# The above is equivalent to the following
# GP_smooth2 = predict(gp_model, gp_coords_pred=coords_train)
# sum(abs(GP_smooth - GP_smooth2$mu))

#--------------------Gaussian process model with linear mean function----------------
# Include a liner regression term instead of assuming a zero-mean a.k.a. "universal Kriging"
gp_model <- fitGPModel(gp_coords = coords_train, cov_function = "exponential",
                       y = y_lin, X=X, likelihood = likelihood, params = list(std_dev = TRUE))
summary(gp_model)

#--------------------Gaussian process model with Vecchia approximation----------------
gp_model <- fitGPModel(gp_coords = coords_train, cov_function = "exponential", 
                       vecchia_approx = TRUE, num_neighbors = 30, y = y_train,
                       likelihood = likelihood)
summary(gp_model)
# Prediction: setting 'num_neighbors_pred' to a larger value than 'num_neighbors' for training
#   can lead to better predictions
pred_vecchia <- predict(gp_model, gp_coords_pred = coords_test, num_neighbors_pred = 100,
                predict_var = TRUE, predict_response = FALSE)
ggplot(data = data.frame(s_1=coords_test[,1],s_2=coords_test[,2],b=pred_vecchia$mu),aes(x=s_1,y=s_2,color=b)) +
  geom_point(size=8, shape=15) + scale_color_viridis(option = "B") + ggtitle("Predicted latent GP mean with Vecchia approxmation")


#--------------------Gaussian process model with Wendland covariance function----------------
gp_model <- fitGPModel(gp_coords = coords_train, cov_function = "wendland", 
                       cov_fct_shape=1, cov_fct_taper_range=0.1, y = y_train,
                       likelihood = likelihood)
summary(gp_model)

#--------------------Gaussian process model with random coefficients----------------
gp_model <- fitGPModel(gp_coords = coords_train, cov_function = "exponential",
                       gp_rand_coef_data = Z_SVC,
                       y = y_svc, likelihood = likelihood)
summary(gp_model)
# Note: this is a small sample size for this type of model
#   -> covariance parameters estimates can have high variance
GP_smooth <- predict_training_data_random_effects(gp_model)
# Compare true and predicted random effects
plot(b_1_train, GP_smooth[,1], xlab="truth", ylab="predicted",
     main="Comparison of true and predicted random effects")
points(b_2, GP_smooth[,2], col=2, pch=2, lwd=1.5)
points(b_3, GP_smooth[,3], col=4, pch=4, lwd=1.5)

# --------------------Using cluster_ids for independent realizations of GPs----------------
cluster_ids = rep(0,ntrain)
cluster_ids[(ntrain/2+1):ntrain] = 1
gp_model <- fitGPModel(gp_coords = coords_train, cov_function = "exponential",
                       cluster_ids = cluster_ids, likelihood = likelihood,
                       y = y_train)
summary(gp_model)

# --------------------Evaluate negative log-likelihood and do optimization using optim----------------
gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                    likelihood = likelihood)
if (likelihood == "gaussian") {
  init_cov_pars <- c(1,1,0.2)
} else {
  init_cov_pars <- c(1,0.2)
}
gp_model$neg_log_likelihood(cov_pars = init_cov_pars, y = y_train)
# Do optimization using optim and e.g. Nelder-Mead
optim(par = init_cov_pars, fn = gp_model$neg_log_likelihood, y = y_train,
      method = "Nelder-Mead")


#################################
# Combined Gaussian process and grouped random effects
#################################
# Simulate data
n <- 500 # number of samples
m <- 50 # number of categories / levels for grouping variable
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
set.seed(1)
coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
sigma2_1 <- 1^2 # random effect variance
sigma2_2 <- 1^2 # marginal variance of GP
rho <- 0.1 # range parameter
b1 <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
D <- as.matrix(dist(coords))
Sigma <- sigma2_2*exp(-D/rho)+diag(1E-20,n)
C <- t(chol(Sigma))
b_2 <- C %*% rnorm(n) # simulate GP
rand_eff <- b1[group] + b_2
rand_eff <- rand_eff - mean(rand_eff)
y <- simulate_response_variable(lp=0, rand_eff=rand_eff, likelihood=likelihood)

# Define and train model
gp_model <- fitGPModel(group_data = group,
                       gp_coords = coords, cov_function = "exponential",
                       y = y, likelihood = likelihood)
summary(gp_model)
