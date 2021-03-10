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
set.seed(1)
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
# Create random effects model and datasets
gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
gp_model$set_optim_params(params=list("optimizer_cov" = "gradient_descent"))
dtrain <- gpb.Dataset(data = X, label = y)
params <- list(objective = "binary", verbose = 0, "num_leaves" = 2^10)

# Small grid and deterministic grid search
param_grid_small = list("learning_rate" = c(0.1,0.01), "min_data_in_leaf" = c(20,100),
                        "max_depth" = c(5,10), "max_bin" = c(255,1000))
set.seed(1)
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid_small,
                                              params = params,
                                              num_try_random = NULL,
                                              nfold = 4,
                                              data = dtrain,
                                              gp_model = gp_model,
                                              verbose_eval = 1,
                                              nrounds = 1000,
                                              early_stopping_rounds = 5,
                                              eval = "binary_logloss")
print(paste0("Best parmeters: ", paste0(unlist(lapply(seq_along(opt_params$best_params), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=opt_params$best_params, n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", opt_params$best_score))

# larger grid and random grid search
param_grid_large = list("learning_rate" = c(0.5,0.1,0.05,0.01), "min_data_in_leaf" = c(5,10,20,50,100,200),
                        "max_depth" = c(1,3,5,10,20), "max_bin" = c(255,500,1000,2000))
set.seed(1)
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid_large,
                                              params = params,
                                              num_try_random = 10,
                                              nfold = 4,
                                              data = dtrain,
                                              gp_model = gp_model,
                                              verbose_eval = 1,
                                              nrounds = 1000,
                                              early_stopping_rounds = 5,
                                              eval = "binary_logloss")
print(paste0("Best parmeters: ", paste0(unlist(lapply(seq_along(opt_params$best_params), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=opt_params$best_params, n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", opt_params$best_score))

# Other metric
set.seed(1)
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid_large,
                                              params = params,
                                              num_try_random = 5,
                                              nfold = 4,
                                              data = dtrain,
                                              gp_model = gp_model,
                                              verbose_eval = 1,
                                              nrounds = 1000,
                                              early_stopping_rounds = 5,
                                              eval = "auc")
print(paste0("Best parmeters: ", paste0(unlist(lapply(seq_along(opt_params$best_params), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=opt_params$best_params, n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", opt_params$best_score))


# --------------------Parameter tuning using a validation set----------------
# Define training and validation data
set.seed(1)
test_ind <- sample.int(n,size=as.integer(0.5*n)) # test indices / samples
folds <- list(test_ind)
# Create random effects model and datasets
gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
gp_model$set_optim_params(params=list("optimizer_cov" = "gradient_descent"))
dtrain <- gpb.Dataset(data = X, label = y)
params <- list(objective = "binary", verbose = 0)
# Parameter tuning using validation data
param_grid = list("learning_rate" = c(0.1,0.01), "min_data_in_leaf" = c(20,100),
                        "max_depth" = c(5,10), "max_bin" = c(255,1000))
set.seed(1)
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid,
                                              params = params,
                                              num_try_random = 5,
                                              folds = folds,
                                              data = dtrain,
                                              gp_model = gp_model,
                                              verbose_eval = 1,
                                              nrounds = 1000,
                                              early_stopping_rounds = 5,
                                              eval = "binary_logloss")
print(paste0("Best parmeters: ", paste0(unlist(lapply(seq_along(opt_params$best_params), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=opt_params$best_params, n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", opt_params$best_score))














#--------------------Cross-validation for finding number of iterations----------------
dtrain <- gpb.Dataset(data = X, label = y)
gp_model <- GPModel(group_data = group, likelihood = likelihood)
cvbst <- gpb.cv(params = params,
                data = dtrain,
                gp_model = gp_model,
                nrounds = 200,
                nfold = 4,
                verbose = 1,
                early_stopping_rounds = 5,
                use_gp_model_for_validation = TRUE)
print(paste0("Optimal number of iterations: ", cvbst$best_iter))
print(cvbst$best_score)
print(min(unlist(cvbst$record_evals[[2]])))




#--------------------Using a validation set for finding number of iterations----------------
set.seed(1)
train_ind <- sample.int(n,size=as.integer(0.8*n))
dtrain <- gpb.Dataset(data = X[train_ind,], label = y[train_ind])
dvalid <- gpb.Dataset.create.valid(dtrain, data = X[-train_ind,], label = y[-train_ind])
valids <- list(test = dvalid)
gp_model <- GPModel(group_data = group[train_ind], likelihood = "bernoulli_probit")
gp_model$set_prediction_data(group_data_pred = group[-train_ind])
bst <- gpb.train(data = dtrain,
                 gp_model = gp_model,
                 nrounds = 100,
                 params = params,
                 verbose = 1,
                 valids = valids,
                 early_stopping_rounds = 5,
                 use_gp_model_for_validation = TRUE)
print(paste0("Optimal number of iterations: ", bst$best_iter,
             ", best test error: ", bst$best_score))

# [1] "[15]:  test's binary_logloss:0.637775"
# [1] "[16]:  test's binary_logloss:0.637073"
# [1] "[17]:  test's binary_logloss:0.638177"
# [1] "[18]:  test's binary_logloss:0.637601"
# [1] "[19]:  test's binary_logloss:0.63834"
# > print(paste0("Optimal number of iterations: ", bst$best_iter,
#                +              ", best test error: ", bst$best_score))
# [1] "Optimal number of iterations: 14, best test error: 0.63704270607988"


test_ind <- (1:n)[-train_ind]
dtrain <- gpb.Dataset(data = X, label = y)
gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
cvbst <- gpb.cv(params = params,
                data = dtrain,
                gp_model = gp_model,
                nrounds = 100,
                folds = list(test_ind),
                verbose = 1,
                early_stopping_rounds = 5,
                use_gp_model_for_validation = TRUE)
print(paste0("Optimal number of iterations: ", cvbst$best_iter))

# [1] "[15]:  valid's binary_logloss:0.632928"
# [1] "[16]:  valid's binary_logloss:0.633622"
# [1] "[17]:  valid's binary_logloss:0.633649"
# [1] "[18]:  valid's binary_logloss:0.633968"
# [1] "[19]:  valid's binary_logloss:0.634278"
# > print(paste0("Optimal number of iterations: ", cvbst$best_iter))
# [1] "Optimal number of iterations: 14"
