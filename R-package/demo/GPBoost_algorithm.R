#############################################################
# Examples on how to do use the GPBoost and LaGaBoost algorithms 
# for various likelihoods:
#   - "gaussian" (=regression)
#   - "bernoulli" (=classification)
#   - "poisson" and "gamma" (=Poisson and gamma regression)
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
  }
  return(y)
}

# Choose likelihood: either "gaussian" ("=" regression), 
#                     "bernoulli_probit", "bernoulli_logit", (= classification)
#                     "poisson", or "gamma"
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

#--------------------Choosing tuning parameters----------------
param_grid <- list("learning_rate" = c(1,0.1,0.01), 
                  "min_data_in_leaf" = c(10,100,1000),
                  "max_depth" = c(1,2,3,5,10),
                  "lambda_l2" = c(0,1,10))
other_params <- list(num_leaves = 2^10)
# Note: here we try different values for 'max_depth' and thus set 'num_leaves' to a large value.
#       An alternative strategy is to impose no limit on 'max_depth', 
#       and try different values for 'num_leaves' as follows:
# param_grid = list("learning_rate" = c(1,0.1,0.01),
#                   "min_data_in_leaf" = c(10,100,1000),
#                   "num_leaves" = 2^(1:10),
#                   "lambda_l2" = c(0,1,10))
# other_params <- list(max_depth = -1)
gp_model <- GPModel(group_data = group, likelihood = likelihood)
dataset <- gpb.Dataset(data = X, label = y)
set.seed(1)
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = other_params,
                                              num_try_random = NULL, nfold = 4,
                                              data = dataset, gp_model = gp_model,
                                              use_gp_model_for_validation = TRUE, verbose_eval = 1,
                                              nrounds = 1000, early_stopping_rounds = 10)
print(paste0("Best parameters: ",
             paste0(unlist(lapply(seq_along(opt_params$best_params), 
                                  function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, 
                                  y=opt_params$best_params, 
                                  n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", round(opt_params$best_score, digits=3)))
# Note: by default, 'test_neg_log_likelihood' is used as 'metric' 
#       Other evaluation metrics / scoring rules can be chosen using the 
#       'metric' argument, e.g., metric = "mse" or metric = "binary_logloss"
#       For more information on available metrics, see 
#       https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
metric = "mse"
if (likelihood %in% c("bernoulli_probit","bernoulli_logit")) {
  metric = "binary_logloss"
}
set.seed(1)
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = other_params,
                                              num_try_random = NULL, nfold = 4,
                                              data = dataset, gp_model = gp_model,
                                              use_gp_model_for_validation = TRUE, verbose_eval = 1,
                                              nrounds = 1000, early_stopping_rounds = 10,
                                              metric = metric)

# Faster computation for large data:
#   using manually defined validation data instead of cross-validation
valid_tune_idx <- sample.int(length(y), as.integer(0.2*length(y))) # use 20% of the data as validation data
folds <- list(valid_tune_idx)
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = other_params,
                                              num_try_random = NULL, folds = folds,
                                              data = dataset, gp_model = gp_model,
                                              use_gp_model_for_validation = TRUE, verbose_eval = 1,
                                              nrounds = 1000, early_stopping_rounds = 10)

#--------------------Cross-validation for determining number of iterations----------------
gp_model <- GPModel(group_data = group, likelihood = likelihood)
dataset <- gpb.Dataset(data = X, label = y)
bst <- gpb.cv(data = dataset, gp_model = gp_model,
              use_gp_model_for_validation = TRUE, params = params,
              nrounds = 1000, nfold = 4, early_stopping_rounds = 10)
print(paste0("Optimal number of iterations: ", bst$best_iter))

#--------------------Using a validation set for finding number of iterations----------------
# Partition data into training and validation data
set.seed(1)
train_ind <- sample.int(n,size=as.integer(n*0.7))
dtrain <- gpb.Dataset(data = X[train_ind,], label = y[train_ind])
dvalid <- gpb.Dataset.create.valid(dtrain, data = X[-train_ind,], label = y[-train_ind])
valids <- list(test = dvalid)
gp_model <- GPModel(group_data = group[train_ind], likelihood = likelihood)
gp_model$set_prediction_data(group_data_pred = group[-train_ind])
bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 1000,
                 params = params, verbose = 1, valids = valids,
                 early_stopping_rounds = 10, use_gp_model_for_validation = TRUE)
print(paste0("Optimal number of iterations: ", bst$best_iter,
             ", best test error: ", bst$best_score))
# Plot validation error
val_error <- unlist(bst$record_evals$test[[1]]$eval)
plot(1:length(val_error), val_error, type="l", lwd=2, col="blue",
     xlab="iteration", ylab="Validation error", main="Validation error vs. boosting iteration")

#--------------------Do Newton updates for tree leaves (only for Gaussian data)----------------
if (likelihood == "gaussian") {
  gp_model <- GPModel(group_data = group[train_ind], likelihood = likelihood)
  gp_model$set_prediction_data(group_data_pred = group[-train_ind])
  params_newton <- params
  params_newton$learning_rate <- 0.1
  bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 1000,
                   params = params_newton, verbose = 1, valids = valids,
                   early_stopping_rounds = 10, use_gp_model_for_validation = TRUE,
                   leaves_newton_update = TRUE)
  print(paste0("Optimal number of iterations: ", bst$best_iter,
               ", best test error: ", bst$best_score))
  # Plot validation error
  val_error <- unlist(bst$record_evals$test[[1]]$eval)
  plot(1:length(val_error), val_error, type="l", lwd=2, col="blue",
       xlab="iteration", ylab="Validation error", 
       main="Validation error vs. boosting iteration")
}

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
gpb.plot.partial.dependence(bst, X, variable = 1)
# Interaction plot
gpb.plot.part.dep.interact(bst, X, variables = c(1,2))
# H-statistic for interactions
library(flashlight)
cols <- paste0("Covariate_",1:p)
fl <- flashlight(model = bst, data = data.frame(y, X), y = "y", label = "gpb",
                 predict_fun = function(m, X) predict(m, data.matrix(X[, cols]), 
                                                      group_data_pred = rep(-1, dim(X)[1]),
                                                      pred_latent = TRUE)$fixed_effect)
plot(imp <- light_interaction(fl, v = cols, pairwise = TRUE))

# SHAP values and dependence plots
library(SHAPforxgboost)
shap.plot.summary.wrap1(bst, X = X)
shap_long <- shap.prep(bst, X_train = X)
shap.plot.dependence(data_long = shap_long, x = "Covariate_1",
                     color_feature = "Covariate_2", smooth = FALSE)

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

# Note: can also convert to string and load from string
model_str <- bst$save_model_to_string()
bst_loaded <- gpb.load(model_str = model_str)

#--------------------GPBoostOOS algorithm: Hyperparameters estimated out-of-sample----------------
# Create random effects model and dataset
gp_model <- GPModel(group_data = group, likelihood = likelihood)
dataset <- gpb.Dataset(X, label = y)
# Stage 1: run cross-validation to (i) determine to optimal number of iterations
#           and (ii) to estimate the GPModel on the out-of-sample data
cvbst <- gpb.cv(data = dataset, gp_model = gp_model,
                use_gp_model_for_validation = TRUE, params = params,
                nrounds = 1000, nfold = 4, early_stopping_rounds = 5,
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
D <- as.matrix(dist(coords))
Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
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
gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                    likelihood = likelihood)
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
library(ggplot2)
library(viridis)
library(gridExtra)
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
run_slow_demo <- FALSE # the following takes some time
if (run_slow_demo) {
  
  param_grid = list("learning_rate" = c(1,0.1,0.01), 
                    "min_data_in_leaf" = c(10,100,1000),
                    "max_depth" = c(1,2,3,5,10),
                    "lambda_l2" = c(0,1,10))
  other_params <- list(num_leaves = 2^10)
  gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                      likelihood = likelihood)
  dtrain <- gpb.Dataset(data = X_train, label = y_train)
  set.seed(1)
  opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = other_params,
                                                num_try_random = NULL, nfold = 4,
                                                data = dtrain, gp_model = gp_model,
                                                use_gp_model_for_validation = TRUE, verbose_eval = 1,
                                                nrounds = 1000, early_stopping_rounds = 10)
  
}

