#############################################################
# Examples on how to do use the GPBoost and LaGaBoost algorithms 
# for various likelihoods:
#   - "gaussian" (=regression)
#   - "bernoulli" (=classification)
#   - "poisson", "gamma", "negative_binomial" (= Poisson, gamma, and negative binomial regression)
# and various random effects models:
#   - grouped (aka clustered) random effects models
#   - Gaussian process (GP) models
# 
# Author: Fabio Sigrist
#############################################################

library(gpboost)

f1d <- function(x) {
  ## Non-linear fixed effects function for simulation
  return( 1 / (1 + exp(-(x - 0.5) * 10)) - 0.5)
}
simulate_response_variable <- function (lp, rand_eff, likelihood) {
  ## Function that simulates response variable for various likelihoods
  n <- length(rand_eff)
  if (likelihood == "gaussian") {
    xi <- sqrt(0.05) * rnorm(n) # error term
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
    shape <- 10
    y <- qgamma(runif(n), scale = mu / shape, shape = shape)
  } else if (likelihood == "negative_binomial") {
    mu <- exp(lp + rand_eff)
    y <- qnbinom(runif(n), mu = mu, size = 1.5)
  }
  return(y)
}

# Choose likelihood: either "gaussian" (=regression), 
#                     "bernoulli_probit", "bernoulli_logit", (=classification)
#                     "poisson", "gamma", or "negative_binomial"
# For a list of all currently supported likelihoods, see https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#likelihood
likelihood <- "gaussian"

#################################
# Combine tree-boosting and grouped random effects model
#################################
# --------------------Simulate data----------------
n <- 5000 # number of samples
m <- 500 # number of groups
set.seed(1)
# Simulate random and fixed effects
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
b1 <- sqrt(0.25) * rnorm(m)
rand_eff <- b1[group]
rand_eff <- rand_eff - mean(rand_eff)
# Simulate fixed effects
p <- 5 # number of predictor variables
X <- matrix(runif(p*n), ncol=p)
f <- f1d(X[,1])
y <- simulate_response_variable(lp=f, rand_eff=rand_eff, likelihood=likelihood)
hist(y, breaks=20)  # visualize response variable

#--------------------Training----------------
# Define random effects model
gp_model <- GPModel(group_data = group, likelihood = likelihood)
# The default optimizer for covariance parameters (hyperparameters) is 
# Nesterov-accelerated gradient descent.
# This can be changed to, e.g., Nelder-Mead as follows:
# set_optim_params(gp_model, params=list(optimizer_cov="nelder_mead"))
# Use the option trace=TRUE to monitor convergence of hyperparameter estimation of the gp_model. E.g.:
# set_optim_params(gp_model, params=list(trace=TRUE))

# Specify boosting parameters
# Note: these parameters are by no means optimal for all data sets but 
#       need to be chosen appropriately, e.g., using 'gpb.grid.search.tune.parameters'
nrounds <- 250
if (likelihood=="gaussian") {
  nrounds <- 50
} else if (likelihood %in% c("bernoulli_probit","bernoulli_logit")) {
  nrounds <- 500
}
params <- list(learning_rate = 0.01, max_depth = 3, num_leaves = 2^10)

bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = nrounds, 
               params = params, verbose = 0)
summary(gp_model) # Estimated random effects model
# Same thing using the gpb.train function
dataset <- gpb.Dataset(data = X, label = y)
bst <- gpb.train(data = dataset, gp_model = gp_model, nrounds = nrounds,
                 params = params, verbose = 0)

#--------------------Prediction----------------
group_test <- 1:m # Predictions for existing groups
group_test_new <- rep(-1,m) # Can also do predictions for new/unobserved groups
x_test <- seq(from=0, to=1, length.out=m)
Xtest <- cbind(x_test, matrix(0, ncol=p-1 , nrow=m))
# 1. Predict latent variable (pred_latent=TRUE) and variance
pred <- predict(bst, data = Xtest, group_data_pred = group_test, 
                predict_var = TRUE, pred_latent = TRUE)
# pred[["fixed_effect"]]: predictions from the tree-ensemble
# pred[["random_effect_mean"]]: predicted means of the gp_model
# pred["random_effect_cov"]]: predicted (co-)variances of the gp_model
# 2. Predict response variable (pred_latent=FALSE)
pred_resp <- predict(bst, data = Xtest, group_data_pred = group_test_new, 
                     predict_var = TRUE, pred_latent = FALSE)
# pred_resp[["response_mean"]]: mean predictions of the response variable 
#   which combines predictions from the tree ensemble and the random effects
# pred_resp[["response_var"]]: predictive (co-)variances (if predict_var=True)

# Visualize fitted response variable
plot(X[,1], y, col=rgb(0,0,0,alpha=0.1), main="Data and predicted response variable")
lines(Xtest[,1], pred_resp$response_mean, col=3, lwd=3)
# Visualize fitted (latent) fixed effects function
x <- seq(from=0, to=1, length.out=200)
plot(x, f1d(x), type="l",lwd=3, col=2, main="True and predicted latent function F")
lines(Xtest[,1], pred$fixed_effect, col=4, lwd=3)
legend(legend=c("True F","Pred F"), "bottomright", bty="n", lwd=3, col=c(2,4))
# Compare true and predicted random effects
plot(b1, pred$random_effect_mean, xlab="truth", ylab="predicted",
     main="Comparison of true and predicted random effects")

#--------------------Choosing tuning parameters using Bayesian optimization and the 'mlrMBO' R package ----------------
packakes_to_load <- c("mlrMBO", "DiceKriging", "rgenoud") # load required packages (non-standard way of loading to avoid CRAN warnings)
for (package in packakes_to_load) do.call(require,list(package, character.only=TRUE))
source("https://raw.githubusercontent.com/fabsig/GPBoost/master/helpers/R_package_tune_pars_bayesian_optimization.R")# Load required function
# Define search space
# Note: if the best combination found below is close to the bounday for a paramter, you might want to extend the corresponding range
search_space <- list("learning_rate" = c(0.001, 10), 
                     "min_data_in_leaf" = c(1, 1000),
                     "max_depth" = c(-1, -1), # -1 means no depth limit as we tune 'num_leaves'. Can also additionally tune 'max_depth', e.g., "max_depth" = c(-1, 1, 2, 3, 5, 10)
                     "num_leaves" = c(2, 2^10),
                     "lambda_l2" = c(0, 100),
                     "max_bin" = c(63, min(n,10000)),
                     "feature_fraction" = c(0.5, 1),
                     "line_search_step_length" = c(TRUE, FALSE))
metric = "mse" # Define metric
if (likelihood %in% c("bernoulli_probit","bernoulli_logit")) {
  metric = "binary_logloss"
}
# Note: can also use metric = "test_neg_log_likelihood". For more options, see https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
gp_model <- GPModel(group_data = group, likelihood = likelihood)
data_train <- gpb.Dataset(data = X, label = y)
# Run parameter optimization using Bayesian optimization and k-fold CV 
crit = makeMBOInfillCritCB() # other criterion options: makeMBOInfillCritEI()
opt_params <- tune.pars.bayesian.optimization(search_space = search_space, n_iter = 100,
                                              data = dataset, gp_model = gp_model,
                                              nfold = 5, nrounds = 1000, early_stopping_rounds = 20,
                                              metric = metric, crit = crit,
                                              cv_seed = 4, verbose_eval = 1)
print(paste0("Best parameters: ", paste0(unlist(lapply(seq_along(opt_params$best_params), 
                                                       function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=opt_params$best_params, 
                                                       n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", round(opt_params$best_score, digits=3)))

# Alternatively and faster: using manually defined validation data instead of cross-validation
valid_tune_idx <- sample.int(length(y), as.integer(0.2*length(y))) # use 20% of the data as validation data
folds <- list(valid_tune_idx)
opt_params <- tune.pars.bayesian.optimization(search_space = search_space, n_iter = 100,
                                              data = dataset, gp_model = gp_model,
                                              folds = folds, nrounds = 1000, early_stopping_rounds = 20,
                                              metric = metric, crit = crit, 
                                              cv_seed = 4, verbose_eval = 1)

#--------------------Choosing tuning parameters using random grid search----------------
param_grid <- list("learning_rate" = c(0.001, 0.01, 0.1, 1, 10), 
                   "min_data_in_leaf" = c(1, 10, 100, 1000),
                   "max_depth" = c(-1), # -1 means no depth limit as we tune 'num_leaves'. Can also additionally tune 'max_depth', e.g., "max_depth" = c(-1, 1, 2, 3, 5, 10)
                   "num_leaves" = 2^(1:10),
                   "lambda_l2" = c(0, 1, 10, 100),
                   "max_bin" = c(250, 500, 1000, min(n,10000)),
                   "feature_fraction" = c(0.5, 0.75, 1),
                   "line_search_step_length" = c(TRUE, FALSE))
metric = "mse" # Define metric
if (likelihood %in% c("bernoulli_probit","bernoulli_logit")) {
  metric = "binary_logloss"
}
# Note: can also use metric = "test_neg_log_likelihood". For more options, see https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
gp_model <- GPModel(group_data = group, likelihood = likelihood)
data_train <- gpb.Dataset(data = X, label = y)
set.seed(1)
# Run parameter optimization using random grid search and k-fold CV
# Note: deterministic grid search can be done by setting 'num_try_random=NULL'
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid,
                                              data = data_train, gp_model = gp_model,
                                              num_try_random = 100, nfold = 5,
                                              nrounds = 1000, early_stopping_rounds = 20,
                                              verbose_eval = 1, metric = metric, cv_seed = 4)
print(paste0("Best parameters: ", paste0(unlist(lapply(seq_along(opt_params$best_params), 
                                                       function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=opt_params$best_params, 
                                                       n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", round(opt_params$best_score, digits=3)))

# Alternatively and faster: using manually defined validation data instead of cross-validation
valid_tune_idx <- sample.int(length(y), as.integer(0.2*length(y))) # use 20% of the data as validation data
folds <- list(valid_tune_idx)
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, 
                                              data = data_train, gp_model = gp_model,
                                              num_try_random = 5, folds = folds,
                                              nrounds = 1000, early_stopping_rounds = 20,
                                              verbose_eval = 1, metric = metric, cv_seed = 4)

#--------------------Cross-validation for determining number of iterations----------------
gp_model <- GPModel(group_data = group, likelihood = likelihood)
dataset <- gpb.Dataset(data = X, label = y)
set.seed(1)
bst <- gpb.cv(data = dataset, gp_model = gp_model, params = params,
              nrounds = 1000, nfold = 5, early_stopping_rounds = 20, metric = metric)
print(paste0("Optimal number of iterations: ", bst$best_iter))

#--------------------Using a validation set for finding number of iterations----------------
# Partition data into training and validation data
set.seed(1)
train_ind <- sample.int(n,size=as.integer(n*0.8))
dtrain <- gpb.Dataset(data = X[train_ind,], label = y[train_ind])
dvalid <- gpb.Dataset.create.valid(dtrain, data = X[-train_ind,], label = y[-train_ind])
valids <- list(test = dvalid)
gp_model <- GPModel(group_data = group[train_ind], likelihood = likelihood)
gp_model$set_prediction_data(group_data_pred = group[-train_ind])
bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 1000,
                 params = params, verbose = 1, valids = valids,
                 early_stopping_rounds = 20, metric = metric)
print(paste0("Optimal number of iterations: ", bst$best_iter,
             ", best test error: ", bst$best_score))
# Plot validation error
val_error <- unlist(bst$record_evals$test[[1]]$eval)
plot(1:length(val_error), val_error, type="l", lwd=2, col="blue",
     xlab="iteration", ylab="Validation error", main="Validation error vs. boosting iteration")

#--------------------Model interpretation----------------
# Note: for the SHAPforxgboost package, the data matrix X needs to have column names
# We add them first:
X <- matrix(as.vector(X), ncol=ncol(X), dimnames=list(NULL, paste0("Covariate_",1:dim(X)[2])))
gp_model <- GPModel(group_data = group, likelihood = likelihood)
bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = nrounds, 
               params = params, verbose = 0)
# Split-based feature importances
feature_importances <- gpb.importance(bst, percentage = TRUE)
gpb.plot.importance(feature_importances, top_n = 5L, measure = "Gain")
# Partial dependence plot
gpb.plot.partial.dependence(bst, X, variable = 1, latent_scale = TRUE)
# Interaction plot
gpb.plot.part.dep.interact(bst, X, variables = c(1,2), latent_scale = TRUE)
# H-statistic for interactions
package_to_load <- "flashlight" # load required package (non-standard way of loading to avoid CRAN warnings)
do.call(require,list(package_to_load, character.only=TRUE))
cols <- paste0("Covariate_",1:p)
fl <- flashlight(model = bst, data = data.frame(y, X), y = "y", label = "gpb",
                 predict_fun = function(m, X) predict(m, data.matrix(X[, cols]), 
                                                      group_data_pred = rep(-1, dim(X)[1]),
                                                      pred_latent = TRUE)$fixed_effect)
plot(imp <- light_interaction(fl, v = cols, pairwise = TRUE))

# SHAP values and dependence plots
package_to_load <- "SHAPforxgboost" # load required package (non-standard way of loading to avoid CRAN warnings)
do.call(require,list(package_to_load, character.only=TRUE))
shap.plot.summary.wrap1(bst, X = X)
shap_long <- shap.prep(bst, X_train = X)
shap.plot.dependence(data_long = shap_long, x = "Covariate_1",
                     color_feature = "Covariate_2", smooth = FALSE)
# SHAP interaction values
source("https://raw.githubusercontent.com/fabsig/GPBoost/master/helpers/R_package_unify_gpboost_treeshap.R")# Load required function
packakes_to_load <- c("treeshap", "shapviz") # load required packages (non-standard way of loading to avoid CRAN warnings)
for (package in packakes_to_load) do.call(require,list(package, character.only=TRUE))
unified_bst <- gpboost_unify_treeshap(bst, X)
interactions_bst <- treeshap(unified_bst, X, interactions = T, verbose = 0)
shap_bst <- shapviz(interactions_bst)
top4 <- names(head(sv_importance(shap_bst, kind = "no"), 4))
sv_interaction(shap_bst[1:1000, top4])
sv_dependence(shap_bst, v = "Covariate_1", color_var = top4, interactions = TRUE)

#--------------------Saving a booster with a gp_model and loading it from a file----------------
# Train model and make predictions
gp_model <- GPModel(group_data = group, likelihood = likelihood)
bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = nrounds, 
               params = params, verbose = 0)
group_test <- c(1,2,-1)
Xtest <- matrix(runif(p*length(group_test)), ncol=p , nrow=length(group_test))
pred <- predict(bst, data = Xtest, group_data_pred = group_test, 
                predict_var = TRUE, pred_latent = TRUE)
# Save model to file
filename <- tempfile(fileext = ".json")
gpb.save(bst, filename = filename)
# Load from file and make predictions again (note: on older R versions, this can sometimes crash)
bst_loaded <- gpb.load(filename = filename)
pred_loaded <- predict(bst_loaded, data = Xtest, group_data_pred = group_test, 
                       predict_var = TRUE, pred_latent = TRUE)
# Check equality
pred$fixed_effect - pred_loaded$fixed_effect
pred$random_effect_mean - pred_loaded$random_effect_mean
pred$random_effect_cov - pred_loaded$random_effect_cov
# Accessing the saved gp_model
summary(bst_loaded$.__enclos_env__$private$gp_model)

# Note: can also convert to string and load from string
# model_str <- bst$save_model_to_string()
# bst_loaded <- gpb.load(model_str = model_str)

#--------------------Continue training----------------
gp_model_cont <- GPModel(group_data = group, likelihood = likelihood)
dataset <- gpb.Dataset(data = X, label = y)
# Train for 10 boosting iterations
bst <- gpb.train(data = dataset, gp_model = gp_model_cont, nrounds = 10,
                 params = params, verbose = 0)
# Continue training with more boosting iterations
bst_cont <- gpb.train(data = dataset, gp_model = gp_model_cont, nrounds = nrounds-10,
                      params = params, verbose = 0,
                      init_model = bst)
pred_cont <- predict(bst_cont, data = Xtest, group_data_pred = group_test, 
                     predict_var = TRUE, pred_latent = TRUE)
# Check equality
pred$fixed_effect - pred_cont$fixed_effect
pred$random_effect_mean - pred_cont$random_effect_mean
pred$random_effect_cov - pred_cont$random_effect_cov
summary(gp_model)
summary(gp_model_cont)

#--------------------Custom validation loss for choosing the number of iterations----------------
l4_loss <- function(preds, dtrain) {
  y <- getinfo(dtrain, "label")
  loss <- sum((preds - y)^4) / length(y)
  list(name = "l4_loss", value = loss, higher_better = FALSE)
}
gp_model <- GPModel(group_data = group, likelihood = likelihood)
dataset <- gpb.Dataset(data = X, label = y)
params_cust <- params
params_cust$first_metric_only <- FALSE  # early stop only on the first metric or not
set.seed(1)
bst <- gpb.cv(data = dataset, gp_model = gp_model, params = params_cust,
              nrounds = 1000, nfold = 5, early_stopping_rounds = 20,
              eval = list(metric, l4_loss),
              use_gp_model_for_validation = FALSE) # Currently, only use_gp_model_for_validation = False is supported
print(paste0("Optimal number of iterations: ", bst$best_iter))
l4_vals <- bst$record_evals$valid[["l4_loss"]]$eval
best_iter_l4 <- which.min(l4_vals)
cat("Best number of iterations for custom loss:", best_iter_l4, "\n")

#--------------------GPBoostOOS algorithm: Hyperparameters estimated out-of-sample----------------
# Create random effects model and dataset
gp_model <- GPModel(group_data = group, likelihood = likelihood)
dataset <- gpb.Dataset(X, label = y)
# Stage 1: run cross-validation to (i) determine to optimal number of iterations
#           and (ii) to estimate the GPModel on the out-of-sample data
cvbst <- gpb.cv(data = dataset, gp_model = gp_model, params = params,
                nrounds = 1000, nfold = 5, early_stopping_rounds = 20,
                fit_GP_cov_pars_OOS = TRUE, verbose = 0)
print(paste0("Optimal number of iterations: ", cvbst$best_iter))
# Fitted model (note: ideally, one would have to find the optimal combination of 
#               other tuning parameters such as the learning rate, tree depth, etc.)
summary(gp_model)
# Stage 2: Train tree-boosting model while holding the gp_model fix
bst <- gpb.train(data = dataset, gp_model = gp_model, nrounds = nrounds,
                 params = params, verbose = 0, train_gp_model_cov_pars = FALSE)
# The gp_model has not changed:
summary(gp_model)


#################################
# Combine tree-boosting and Gaussian process model
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
sigma2_1 <- 0.25 # marginal variance of GP
rho <- 0.1 # range parameter
D_scaled <- sqrt(3) * as.matrix(dist(coords)) / rho
Sigma <- sigma2_1 * (1 + D_scaled) * exp(-D_scaled) + diag(1E-20,n) # Matern 1.5 covariance
C <- t(chol(Sigma))
b_1 <- as.vector(C %*% rnorm(n=n))
b_1 <- b_1 - mean(b_1)
# Simulate fixed effects
X_train <- matrix(runif(2*ntrain),ncol=2)
x <- seq(from=0,to=1,length.out=nx^2)
X_test <- cbind(x,rep(0,nx^2))
X <- rbind(X_train,X_test)
f <- f1d(X[,1])
y <- simulate_response_variable(lp=f, rand_eff=b_1, likelihood=likelihood)
# Split into training and test data
y_train <- y[1:ntrain]
y_test <- y[1:ntest+ntrain]
b_1_train <- b_1[1:ntrain]
b_1_test <- b_1[1:ntest+ntrain]
hist(y_train, breaks=20)# visualize response variable

# Specify boosting parameters as list
params <- list(learning_rate = 0.1, max_depth = 3)
nrounds <- 10
if (likelihood %in% c("bernoulli_probit","bernoulli_logit")) {
  nrounds <- 50
} 

#--------------------Training----------------
# Define Gaussian process model
gp_model <- GPModel(gp_coords = coords_train, cov_function = "matern", cov_fct_shape = 1.5,
                    likelihood = likelihood)
# GPs become slow for large data sets -> use an approximation such as a Vecchia approximation:
# gp_model <- GPModel(gp_coords = coords_train, cov_function = "matern", cov_fct_shape = 1.5,
#                     likelihood = likelihood, gp_approx = "vecchia")
# Create dataset for gpb.train
dtrain <- gpb.Dataset(data = X_train, label = y_train)
bst <- gpb.train(data = dtrain, gp_model = gp_model,
                 nrounds = nrounds, params = params, verbose = 0)
# Takes a few seconds
summary(gp_model)# Trained GP model

#--------------------Prediction----------------
# 1. Predict response variable (pred_latent = FALSE)
pred_resp <- predict(bst, data = X_test, gp_coords_pred = coords_test, 
                     predict_var = TRUE, pred_latent = FALSE)
# pred_resp[["response_mean"]]: mean predictions of the response variable 
#   which combines predictions from the tree ensemble and the Gaussian process
# pred_resp[["response_var"]]: predictive (co-)variances (if predict_var=True)
# 2. Prediction of latent variables (pred_latent = TRUE)
pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                predict_var = TRUE, pred_latent = TRUE)
# pred[["fixed_effect"]]: predictions for the latent fixed effects / tree ensemble
# pred[["random_effect_mean"]]: mean predictions for the random effects
# pred[["random_effect_cov"]]: predictive (co-)variances (if predict_var=True) of the (latent) Gaussian process
# 3. Can also calculate predictive covariances
pred_cov = predict(bst, data=X_test[1:3,], gp_coords_pred=coords_test[1:3,],
                   predict_cov_mat=TRUE, pred_latent=TRUE)
# pred_cov[["random_effect_cov"]]: predictive covariances of the (latent) Gaussian process
if (likelihood == "gaussian") {
  # Predictive covariances for the response variable are currently only supported for Gaussian likelihoods
  pred_resp_cov = predict(bst, data=X_test[1:3,], gp_coords_pred=coords_test[1:3,],
                          predict_cov_mat=TRUE, pred_latent=FALSE)
  # pred_resp_cov[["response_var"]]: predictive covariances of the response variable
}

# Evaluate predictions
if (likelihood %in% c("bernoulli_probit","bernoulli_logit")) {
  print(paste0("Test error: ", 
               mean(as.numeric(pred_resp$response_mean>0.5) != y_test)))
} else {
  print(paste0("Test root mean square error: ",
               sqrt(mean((pred_resp$response_mean - y_test)^2))))
}
print(paste0("Test root mean square error for latent GP: ", 
             sqrt(mean((pred$random_effect_mean - b_1_test)^2))))

# Visualize predictions and compare to true values
packakes_to_load <- c("ggplot2", "viridis", "gridExtra") # load required packages (non-standard way of loading to avoid CRAN warnings)
for (package in packakes_to_load) do.call(require,list(package, character.only=TRUE))
plot1 <- ggplot(data = data.frame(s_1=coords_test[,1],s_2=coords_test[,2],b=b_1_test),aes(x=s_1,y=s_2,color=b)) +
  geom_point(size=4, shape=15) + scale_color_viridis(option = "B") + ggtitle("True latent GP and training locations") + 
  geom_point(data = data.frame(s_1=coords_train[,1],s_2=coords_train[,2],y=y_train),aes(x=s_1,y=s_2),size=3, col="white", alpha=1, shape=43)
plot2 <- ggplot(data = data.frame(s_1=coords_test[,1],s_2=coords_test[,2],b=pred$random_effect_mean),aes(x=s_1,y=s_2,color=b)) +
  geom_point(size=4, shape=15) + scale_color_viridis(option = "B") + ggtitle("Predicted latent GP mean")
plot3 <- ggplot(data = data.frame(s_1=coords_test[,1],s_2=coords_test[,2],b=sqrt(pred$random_effect_cov)),aes(x=s_1,y=s_2,color=b)) +
  geom_point(size=4, shape=15) + scale_color_viridis(option = "B") + labs(title="Predicted latent GP standard deviation", subtitle=" = prediction uncertainty")
plot4 <- ggplot(data=data.frame(x=X_test[,1],f=pred$fixed_effect), aes(x=x,y=f)) + geom_line(size=1) +
  geom_line(data=data.frame(x=x,f=f1d(x)), aes(x=x,y=f), size=1.5, color="darkred") +
  ggtitle("Predicted and true F(X)") + xlab("X") + ylab("y")
grid.arrange(plot1, plot2, plot3, plot4, ncol=2)

#--------------------Choosing tuning parameters----------------
# Choosing tuning parameters carefully is important.
# See the above demo code for grouped random effects on how this can be done.
# You just have to replace the gp_model. E.g.,    
# gp_model <- GPModel(gp_coords = coords_train, cov_function = "matern", cov_fct_shape = 1.5, likelihood = likelihood)

#--------------------Model interpretation----------------
# See the above demo code for grouped random effects on how this can be done.
