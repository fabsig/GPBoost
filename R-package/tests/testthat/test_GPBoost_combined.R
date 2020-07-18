context("GPBoost_combined_boosting_GP_random_effects")

# Function that simulates uniform random variables
sim_rand_unif <- function(n, init_c=0.1){
  mod_lcg <- 134456 # modulus for linear congruential generator (random0 used)
  sim <- rep(NA, n)
  sim[1] <- floor(init_c * mod_lcg)
  for(i in 2:n) sim[i] <- (8121 * sim[i-1] + 28411) %% mod_lcg
  return(sim / mod_lcg)
}
# Function for non-linear mean
sim_friedman3=function(n, n_irrelevant=5){
  X <- matrix(sim_rand_unif(4*n,init_c=0.24234),ncol=4)
  X[,1] <- 100*X[,1]
  X[,2] <- X[,2]*pi*(560-40)+40*pi
  X[,4] <- X[,4]*10+1
  f <- sqrt(10)*atan((X[,2]*X[,3]-1/(X[,2]*X[,4]))/X[,1])
  X <- cbind(rep(1,n),X)
  if(n_irrelevant>0) X <- cbind(X,matrix(sim_rand_unif(5*n,init_c=0.6543),ncol=n_irrelevant))
  return(list(X=X,f=f))
}



test_that("Combine tree-boosting and grouped random effects model ", {
  ntrain <- ntest <- 1000
  n <- ntrain + ntest
  
  # Simulate fixed effects
  sim_data <- sim_friedman3(n=n, n_irrelevant=5)
  f <- sim_data$f
  X <- sim_data$X
  # Simulate grouped random effects
  sigma2_1 <- 0.6 # variance of first random effect 
  sigma2_2 <- 0.4 # variance of second random effect
  sigma2 <- 0.1^2 # error variance
  m <- 40 # number of categories / levels for grouping variable
  # first random effect
  group <- rep(1,ntrain) # grouping variable
  for(i in 1:m) group[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
  group <- c(group, group)
  Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
  b1 <- sqrt(sigma2_1) * qnorm(sim_rand_unif(n=m, init_c=0.542))
  # Second random effect
  n_obs_gr <- ntrain/m# number of sampels per group
  group2 <- rep(1,ntrain) # grouping variable
  for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
  group2 <- c(group2,group2)
  Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
  b2 <- sqrt(sigma2_2) * qnorm(sim_rand_unif(n=n_obs_gr, init_c=0.2354))
  eps <- Z1 %*% b1 + Z2 %*% b2
  group_data <- cbind(group,group2)
  # Error term
  xi <- sqrt(sigma2) * qnorm(sim_rand_unif(n=n, init_c=0.756))
  # Observed data
  y <- f + eps + xi
  # Signal-to-noise ratio of approx. 1
  # var(f) / var(eps)
  # Split in training and test data
  y_train <- y[1:ntrain]
  X_train <- X[1:ntrain,]
  group_data_train <- group_data[1:ntrain,]
  y_test <- y[1:ntest+ntrain]
  X_test <- X[1:ntest+ntrain,]
  f_test <- f[1:ntest+ntrain]
  group_data_test <- group_data[1:ntest+ntrain,]
  
  # Define random effects model
  gp_model <- GPModel(group_data = group_data_train)
  # CV for finding number of boosting iterations
  dtrain <- gpb.Dataset(X_train, label = y_train)
  params <- list(learning_rate = 0.01,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 has_gp_model = TRUE)
  folds <- list()
  for(i in 1:4) folds[[i]] <- as.integer(1:(ntrain/4) + (ntrain/4) * (i-1))
  cvbst <- gpb.cv(params = params,
                  data = dtrain,
                  gp_model = gp_model,
                  nrounds = 100,
                  nfold = 4,
                  eval = "l2",
                  early_stopping_rounds = 5,
                  use_gp_model_for_validation = FALSE,
                  fit_GP_cov_pars_OOS = FALSE,
                  folds = folds,
                  verbose=0)
  expect_equal(cvbst$best_iter, 62)
  expect_lt(abs(cvbst$best_score-1.020787), 1E-6)

  # Create random effects model and train GPBoost model
  gp_model <- GPModel(group_data = group_data_train)
  bst <- gpboost(data = X_train,
                 label = y_train,
                 gp_model = gp_model,
                 nrounds = 62,
                 learning_rate = 0.01,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 verbose = 0,
                 leaves_newton_update = FALSE)
  cov_pars <- c(0.005083987, 0.590482497, 0.390662914)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  
  # Prediction
  pred <- predict(bst, data = X_test, group_data_pred = group_data_test)
  expect_equal(ntrain, 1000)
  expect_equal(ntest, 1000)
  expect_lt(sqrt(mean((pred$fixed_effect - f_test)^2)),0.262)
  expect_lt(sqrt(mean((pred$fixed_effect - y_test)^2)),1.023)
  expect_lt(sqrt(mean((pred$fixed_effect + pred$random_effect_mean - y_test)^2)),0.226)
  
  # Fit parameters on out-of-sample data
  gp_model <- GPModel(group_data = group_data_train)
  cvbst <- gpb.cv(params = params,
                  data = dtrain,
                  gp_model = gp_model,
                  nrounds = 100,
                  nfold = 4,
                  eval = "l2",
                  early_stopping_rounds = 5,
                  use_gp_model_for_validation = FALSE,
                  fit_GP_cov_pars_OOS = TRUE,
                  folds = folds)
  expect_equal(cvbst$best_iter, 62)
  cov_pars_OOS <- c(0.05471771, 0.59169629, 0.38950232)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_OOS)),1E-6)
  
  # Train tree-boosting model while holding the GPModel fix
  bst <- gpb.train(data = dtrain,
                   gp_model = gp_model,
                   nrounds = 62,
                   learning_rate = 0.05,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "regression_l2",
                   verbose = 0,
                   train_gp_model_cov_pars = FALSE)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_OOS)),1E-6)# no change in covariance parameters
  
  # Newton updates for tree leaves
  params <- list(learning_rate = 0.1,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 has_gp_model = TRUE,
                 leaves_newton_update = TRUE)
  gp_model <- GPModel(group_data = group_data_train)
  cvbst <- gpb.cv(params = params,
                  data = dtrain,
                  gp_model = gp_model,
                  nrounds = 100,
                  nfold = 4,
                  eval = "l2",
                  early_stopping_rounds = 5,
                  use_gp_model_for_validation = FALSE,
                  fit_GP_cov_pars_OOS = TRUE,
                  folds = folds)
  expect_equal(cvbst$best_iter, 59)
  cov_pars_OOS <- c(0.04534303, 0.60689622, 0.38936079)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_OOS)),1E-6)
  
  # Using validation set
  dtrain <- gpb.Dataset(data = X_train, label = y_train)
  dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
  valids <- list(test = dtest)
  # Do not include random effect predictions for validation
  gp_model <- GPModel(group_data = group_data_train)
  bst <- gpb.train(data = dtrain,
                   gp_model = gp_model,
                   nrounds = 100,
                   learning_rate = 0.01,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "regression_l2",
                   verbose = 1,
                   valids = valids,
                   early_stopping_rounds = 5,
                   use_gp_model_for_validation = FALSE)
  expect_equal(bst$best_iter, 57)
  expect_lt(abs(bst$best_score - 1.02941984003819),1E-6)
  
  # Include random effect predictions for validation 
  gp_model <- GPModel(group_data = group_data_train)
  gp_model$set_prediction_data(group_data_pred = group_data_test)
  bst <- gpb.train(data = dtrain,
                   gp_model = gp_model,
                   nrounds = 100,
                   learning_rate = 0.01,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "regression_l2",
                   verbose = 1,
                   valids = valids,
                   early_stopping_rounds = 5,
                   use_gp_model_for_validation = TRUE)
  expect_equal(bst$best_iter, 59)
  expect_lt(abs(bst$best_score - 0.04388658),1E-6)
})


test_that("Combine tree-boosting and Gaussian process model ", {
  ntrain <- ntest <- 500
  n <- ntrain + ntest
  
  # Simulate fixed effects
  sim_data <- sim_friedman3(n=n, n_irrelevant=5)
  f <- sim_data$f
  X <- sim_data$X
  # Simulate grouped random effects
  sigma2_1 <- 1 # marginal variance of GP
  rho <- 0.1 # range parameter
  sigma2 <- 0.1 # error variance
  d <- 2 # dimension of GP locations
  coords <- matrix(sim_rand_unif(n=n*d, init_c=0.63), ncol=d)
  D <- as.matrix(dist(coords))
  Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
  C <- t(chol(Sigma))
  b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.864))
  eps <- as.vector(C %*% b_1)
  # Error term
  xi <- sqrt(sigma2) * qnorm(sim_rand_unif(n=n, init_c=0.36))
  # Observed data
  y <- f + eps + xi
  # Split in training and test data
  y_train <- y[1:ntrain]
  X_train <- X[1:ntrain,]
  coords_train <- coords[1:ntrain,]
  y_test <- y[1:ntest+ntrain]
  X_test <- X[1:ntest+ntrain,]
  f_test <- f[1:ntest+ntrain]
  coords_test <- coords[1:ntest+ntrain,]
  
  # Train model
  gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
  gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring"))
  bst <- gpb.train(data = dtrain,
                   gp_model = gp_model,
                   nrounds = 20,
                   learning_rate = 0.05,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "regression_l2")
  cov_pars_est <- c(0.1358229, 0.9099908, 0.1115316)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),1E-6)
  # Prediction
  pred <- predict(bst, data = X_test, gp_coords_pred = coords_test)
  expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-0.5229658),1E-6)
  expect_lt(abs(sqrt(mean((pred$fixed_effect - y_test)^2))-1.170505),1E-6)
  expect_lt(abs(sqrt(mean((pred$fixed_effect + pred$random_effect_mean - y_test)^2))-0.8304062),1E-6)
  
  # Use validation set to determine number of boosting iteration
  dtrain <- gpb.Dataset(data = X_train, label = y_train)
  dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
  valids <- list(test = dtest)
  gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
  gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring"))
  bst <- gpb.train(data = dtrain,
                   gp_model = gp_model,
                   nrounds = 100,
                   learning_rate = 0.05,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "regression_l2",
                   verbose = 0,
                   valids = valids,
                   early_stopping_rounds = 5,
                   use_gp_model_for_validation = FALSE)
  expect_equal(bst$best_iter, 27)
  expect_lt(abs(bst$best_score - 1.293498),1E-6)
  
  # Also use GPModel for calculating validation error
  gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential")
  gp_model$set_optim_params(params=list(maxit=20, optimizer_cov="fisher_scoring"))
  gp_model$set_prediction_data(gp_coords_pred = coords_test)
  bst <- gpb.train(data = dtrain,
                   gp_model = gp_model,
                   nrounds = 100,
                   learning_rate = 0.05,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "regression_l2",
                   verbose = 0,
                   valids = valids,
                   early_stopping_rounds = 5,
                   use_gp_model_for_validation = TRUE)
  expect_equal(bst$best_iter, 27)
  expect_lt(abs(bst$best_score - 0.550003),1E-6)
})

