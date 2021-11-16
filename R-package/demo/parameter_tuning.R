## Examples on how to do parameter tuning for the GPBoost algorithm
## Author: Fabio Sigrist

library(gpboost)

# Non-linear prior mean function for simulation in examples below
f1d <- function(x) 1/(1+exp(-(x-0.5)*10)) - 0.5
sim_non_lin_f <- function(n){
  X <- matrix(runif(2*n),ncol=2)
  f <- f1d(X[,1])
  return(list(X=X,f=f))
}

# --------------------Simulate data grouped random effects data----------------
n <- 1000 # number of samples
m <- 100 # number of groups
set.seed(100)
# Simulate random and fixed effects
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
b1 <- sqrt(0.5) * rnorm(m)
eps <- b1[group]
eps <- eps - mean(eps)
sim_data <- sim_non_lin_f(n=n)
f <- sim_data$f
X <- sim_data$X
# Simulate response variable
probs <- pnorm(f+eps)
y <- as.numeric(runif(n) < probs)

# --------------------Parameter tuning using cross-validation: deterministic and random grid search----------------
# Create random effects model and Dataset
gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
dtrain <- gpb.Dataset(data = X, label = y)
params <- list(objective = "binary", verbose = 0, "num_leaves" = 2^10, "max_bin" = 255)

# Small grid and deterministic search
param_grid_small = list("learning_rate" = c(1,0.1,0.01), "min_data_in_leaf" = c(20,100),
                        "max_depth" = c(5,10))
# Note: Usually smaller learning rates lead to more accurate models. However, it is
#         advisable to also try larger learning rates (e.g., 1 or larger) since when using 
#         gradient boosting, the scale of the gradient can depend on the loss function and the data,
#         and even a larger number of boosting iterations (say 1000) might not be enough for small learning rates.
#         This is in contrast to Newton boosting, where learning rates smaller than 0.1 are used
#         since the natural gradient is not scale-dependent.
set.seed(100)
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid_small,
                                              params = params,
                                              num_try_random = NULL,
                                              nfold = 4,
                                              data = dtrain,
                                              gp_model = gp_model,
                                              verbose_eval = 1,
                                              nrounds = 1000,
                                              early_stopping_rounds = 10,
                                              eval = "binary_logloss")
print(paste0("Best parameters: ",paste0(unlist(lapply(seq_along(opt_params$best_params), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=opt_params$best_params, n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", round(opt_params$best_score, digits=3)))
# I obtained the following best parameters:
# ***** New best score (0.593915259806541) found for the following parameter combination: learning_rate: 0.1, min_data_in_leaf: 100, max_depth: 10, nrounds: 35

# Larger grid and random search
param_grid_large = list("learning_rate" = c(5,1,0.5,0.1,0.05,0.01), "min_data_in_leaf" = c(5,10,20,50,100,200),
                        "max_depth" = c(1,3,5,10,20), "max_bin" = c(255,500,1000))
set.seed(100)
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid_large,
                                              params = params,
                                              num_try_random = 10,
                                              nfold = 4,
                                              data = dtrain,
                                              gp_model = gp_model,
                                              verbose_eval = 1,
                                              nrounds = 1000,
                                              early_stopping_rounds = 10,
                                              eval = "binary_logloss")
print(paste0("Best parameters: ", paste0(unlist(lapply(seq_along(opt_params$best_params), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=opt_params$best_params, n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", round(opt_params$best_score, digits=3)))

# Using another metric (AUC) instead of the log-loss
set.seed(100)
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid_large,
                                              params = params,
                                              num_try_random = 5,
                                              nfold = 4,
                                              data = dtrain,
                                              gp_model = gp_model,
                                              verbose_eval = 1,
                                              nrounds = 1000,
                                              early_stopping_rounds = 10,
                                              eval = "auc")
print(paste0("Best parameters: ", paste0(unlist(lapply(seq_along(opt_params$best_params), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=opt_params$best_params, n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", round(opt_params$best_score, digits=3)))
# Note: it is coincidence that the AUC and the log-loss have similar values on this data

# --------------------Parameter tuning using a validation set----------------
# Define training and validation data by setting indices of 'folds'
set.seed(100)
test_ind <- sample.int(n,size=as.integer(0.5*n)) # test indices / samples
folds <- list(test_ind)
# Parameter tuning using validation data
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid_small,
                                              params = params,
                                              folds = folds,
                                              data = dtrain,
                                              gp_model = gp_model,
                                              verbose_eval = 1,
                                              nrounds = 1000,
                                              early_stopping_rounds = 10,
                                              eval = "binary_logloss")
print(paste0("Best parameters: ", paste0(unlist(lapply(seq_along(opt_params$best_params), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=opt_params$best_params, n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", round(opt_params$best_score, digits=3)))
