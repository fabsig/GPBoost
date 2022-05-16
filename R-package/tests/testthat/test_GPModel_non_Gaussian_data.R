context("GPModel_non_Gaussian_data")

TOLERANCE <- 1e-3
TOLERANCE2 <- 1E-6

# Function that simulates uniform random variables
sim_rand_unif <- function(n, init_c=0.1){
  mod_lcg <- 2^32 # modulus for linear congruential generator (random0 used)
  sim <- rep(NA, n)
  sim[1] <- floor(init_c * mod_lcg)
  for(i in 2:n) sim[i] <- (22695477 * sim[i-1] + 1) %% mod_lcg
  return(sim / mod_lcg)
}

# Simulate data
n <- 100 # number of samples
# Simulate locations / features of GP
d <- 2 # dimension of GP locations
coords <- matrix(sim_rand_unif(n=n*d, init_c=0.1), ncol=d)
D <- as.matrix(dist(coords))
# Simulate GP
sigma2_1 <- 1^2 # marginal variance of GP
rho <- 0.1 # range parameter
Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
L <- t(chol(Sigma))
b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.8))
# GP random coefficients
Z_SVC <- matrix(sim_rand_unif(n=n*2, init_c=0.6), ncol=2) # covariate data for random coefficients
colnames(Z_SVC) <- c("var1","var2")
b_2 <- qnorm(sim_rand_unif(n=n, init_c=0.17))
b_3 <- qnorm(sim_rand_unif(n=n, init_c=0.42))
# First grouped random effects model
m <- 10 # number of categories / levels for grouping variable
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
b_gr_1 <- qnorm(sim_rand_unif(n=m, init_c=0.565))
# Second grouped random effect
n_obs_gr <- n/m # number of samples per group
group2 <- rep(1,n) # grouping variable
for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
b_gr_2 <- qnorm(sim_rand_unif(n=n_obs_gr, init_c=0.36))
# Grouped random slope / coefficient
x <- cos((1:n-n/2)^2*5.5*pi/n) # covariate data for random slope
Z3 <- diag(x) %*% Z1
b_gr_3 <- qnorm(sim_rand_unif(n=m, init_c=0.5678))
# Data for linear mixed effects model
X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # design matrix / covariate data for fixed effect
beta <- c(0.1,2) # regression coefficients
# cluster_ids 
cluster_ids <- c(rep(1,0.4*n),rep(2,0.6*n))
# GP with multiple observations at the same locations
coords_multiple <- matrix(sim_rand_unif(n=n*d/4, init_c=0.1), ncol=d)
coords_multiple <- rbind(coords_multiple,coords_multiple,coords_multiple,coords_multiple)
D_multiple <- as.matrix(dist(coords_multiple))
Sigma_multiple <- sigma2_1*exp(-D_multiple/rho)+diag(1E-10,n)
L_multiple <- t(chol(Sigma_multiple))
b_multiple <- qnorm(sim_rand_unif(n=n, init_c=0.8))

test_that("Binary classification with Gaussian process model ", {
  
  probs <- pnorm(L %*% b_1)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
  # Label needs to have correct format
  expect_error(fitGPModel(gp_coords = coords, cov_function = "exponential",
                          likelihood = "bernoulli_probit",
                          y = b_1, params = list(optimizer_cov = "gradient_descent")))
  yw <- y
  yw[3] <- yw[3] + 1E-6
  expect_error(fitGPModel(gp_coords = coords, cov_function = "exponential",
                          likelihood = "bernoulli_probit",
                          y = yw, params = list(optimizer_cov = "gradient_descent")))
  # Only gradient descent can be used
  expect_error(fitGPModel(gp_coords = coords, cov_function = "exponential",
                          likelihood = "bernoulli_probit",
                          y = y, params = list(optimizer_cov = "fisher_scoring")))
  # Estimation using gradient descent
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
  fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                     lr_cov = 0.1, use_nesterov_acc = FALSE,
                                     convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.9419234, 0.1866877)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
  expect_equal(gp_model$get_num_optim_iter(), 40)
  # Estimation using gradient descent and Nesterov acceleration
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
  fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                     lr_cov = 0.01, use_nesterov_acc = TRUE, acc_rate_cov = 0.5))
  cov_pars2 <- c(0.9646422, 0.1844797)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars2)),TOLERANCE2)
  expect_equal(gp_model$get_num_optim_iter(), 26)
  # Estimation using Nelder-Mead
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
  fit(gp_model, y = y, params = list(optimizer_cov = "nelder_mead"))
  cov_pars3 <- c(0.9998047, 0.1855072)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars3)),TOLERANCE2)
  expect_equal(gp_model$get_num_optim_iter(), 6)
  # Estimation using BFGS
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
  fit(gp_model, y = y, params = list(optimizer_cov = "bfgs"))
  cov_pars3 <- c(0.9419084, 0.1866882)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars3)),TOLERANCE2)
  expect_equal(gp_model$get_num_optim_iter(), 4)
  # Estimation using Adam
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
  fit(gp_model, y = y, params = list(optimizer_cov = "adam"))
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars3)),TOLERANCE2)
  expect_equal(gp_model$get_num_optim_iter(), 200)
  
  # Prediction
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                         y = y, params = list(optimizer_cov = "gradient_descent", lr_cov=0.01, use_nesterov_acc=FALSE))
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(-0.6595663, -0.6638940, 0.4997690)
  expected_cov <- c(0.6482224576, 0.5765285950, -0.0001030520, 0.5765285950,
                    0.6478191338, -0.0001163496, -0.0001030520, -0.0001163496, 0.4435551436)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
  # Predict variances
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, predict_response = FALSE)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE2)
  # Predict response
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var=TRUE, predict_response = TRUE)
  expected_mu <- c(0.3037139, 0.3025143, 0.6612807)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(pred$var-expected_mu*(1-expected_mu))),TOLERANCE2)
  
  # Predict training data random effects
  training_data_random_effects <- predict_training_data_random_effects(gp_model)
  pred_random_effects <- predict(gp_model, gp_coords_pred = coords)
  expect_lt(sum(abs(training_data_random_effects - pred_random_effects$mu)),1E-6)
  
  # Evaluate approximate negative marginal log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
  expect_lt(abs(nll-63.6205917),TOLERANCE2)
  
  # Do optimization using optim and e.g. Nelder-Mead
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
  opt <- optim(par=c(1,0.1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
  cov_pars <- c(0.9419234, 0.1866877)
  expect_lt(sum(abs(opt$par-cov_pars)),TOLERANCE)
  expect_lt(abs(opt$value-(63.6126363)),TOLERANCE)
  expect_equal(as.integer(opt$counts[1]), 47)
  
  ###################
  ## Random coefficient GPs
  ###################
  probs <- pnorm(as.vector(L %*% b_1 + Z_SVC[,1] * L %*% b_2 + Z_SVC[,2] * L %*% b_3))
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.543) < probs)
  # Fit model
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                         y = y, likelihood = "bernoulli_probit",
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 1, use_nesterov_acc = TRUE, acc_rate_cov=0.5, maxit=1000))
  expected_values <- c(0.3701097, 0.2846740, 2.1160325, 0.3305266, 0.1241462, 0.1846456)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE2)
  expect_equal(gp_model$get_num_optim_iter(), 39)
  # Prediction
  gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential", likelihood = "bernoulli_probit")
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           gp_rand_coef_data_pred=Z_SVC_test,
                           cov_pars = c(1,0.1,0.8,0.15,1.1,0.08),
                           predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(0.18346008, 0.03479258, -0.17247579)
  expected_cov <- c(1.039879e+00, 7.521981e-01, -3.256500e-04, 7.521981e-01,
                    8.907289e-01, -6.719282e-05, -3.256500e-04, -6.719282e-05, 9.147899e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(1,0.1,0.8,0.15,1.1,0.08),y=y)
  expect_lt(abs(nll-65.1768199),TOLERANCE)
  
  ###################
  ##  Multiple cluster IDs
  ###################
  probs <- pnorm(L %*% b_1)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.2978341) < probs)
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         y = y, cluster_ids = cluster_ids,likelihood = "bernoulli_probit",
                         params = list(optimizer_cov = "gradient_descent", lr_cov=0.2, use_nesterov_acc=FALSE))
  cov_pars <- c(0.5085134, 0.2011667)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
  expect_equal(gp_model$get_num_optim_iter(), 20)
  # Prediction
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  cluster_ids_pred = c(1,3,1)
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                      cluster_ids = cluster_ids,likelihood = "bernoulli_probit")
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           cluster_ids_pred = cluster_ids_pred,
                           cov_pars = c(1.5,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(0.1509569, 0.0000000, 0.9574946)
  expected_cov <- c(1.2225959453, 0.0000000000, 0.0003074858, 0.0000000000,
                    1.5000000000, 0.0000000000, 0.0003074858, 0.0000000000, 1.0761874845)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
})

test_that("Binary classification with Gaussian process model with multiple observations at the same location", {
  eps_multiple <- as.vector(L_multiple %*% b_multiple)
  probs <- pnorm(eps_multiple)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.9341) < probs)
  
  gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                         y = y,likelihood = "bernoulli_probit",
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov=0.1, use_nesterov_acc=TRUE))
  cov_pars <- c(0.6857065, 0.2363754)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
  expect_equal(gp_model$get_num_optim_iter(), 8)
  # Prediction
  coord_test <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           cov_pars = c(1.5,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(-0.2633282, -0.2637633, -0.2637633)
  expected_cov <- c(0.9561355, 0.8535206, 0.8535206, 0.8535206, 1.0180227,
                    1.0180227, 0.8535206, 1.0180227, 1.0180227)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
  pred_resp <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                                cov_pars = c(1.5,0.15), predict_var = TRUE, predict_response = TRUE)
  expect_lt(sum(abs(pred_resp$mu-c(0.4253296, 0.4263502, 0.4263502))),TOLERANCE2)
  expect_lt(sum(abs(pred_resp$var-c(0.2444243, 0.2445757, 0.2445757))),TOLERANCE2)
  
  # Predict training data random effects
  training_data_random_effects <- predict_training_data_random_effects(gp_model)
  pred_random_effects <- predict(gp_model, gp_coords_pred = coords_multiple)
  expect_lt(sum(abs(training_data_random_effects - pred_random_effects$mu)),1E-6)
  
  # Multiple cluster IDs and multiple observations
  coord_test <- cbind(c(0.1,0.11,0.11),c(0.9,0.91,0.91))
  cluster_ids_pred = c(0L,3L,3L)
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test, cluster_ids_pred = cluster_ids_pred,
                           cov_pars = c(1.5,0.15), predict_cov_mat = TRUE, predict_response = FALSE)
  expected_cov <- c(0.9561355, 0.0000000, 0.0000000, 0.0000000, 1.5000000,
                    1.5000000, 0.0000000, 1.5000000, 1.5000000)
  expect_lt(sum(abs(pred$mu-c(-0.2633282, rep(0,2)))),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
  pred_resp <- gp_model$predict(y = y, gp_coords_pred = coord_test, cluster_ids_pred = cluster_ids_pred,
                                cov_pars = c(1.5,0.15), predict_var = TRUE, predict_response = TRUE)
  expect_lt(sum(abs(pred_resp$mu-c(0.4253296, 0.5000000, 0.5000000))),TOLERANCE2)
  expect_lt(sum(abs(pred_resp$var-c(0.2444243, 0.2500000, 0.2500000))),TOLERANCE2)
})

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  test_that("Binary classification with one grouped random effects ", {
    
    probs <- pnorm(Z1 %*% b_gr_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.823431) < probs)
    
    # Estimation using gradient descent
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                       convergence_criterion = "relative_change_in_parameters"))
    cov_pars <- c(0.40255)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 62)
    
    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                       lr_cov = 0.1, use_nesterov_acc = TRUE, acc_rate_cov = 0.5))
    cov_pars2 <- c(0.4012595)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars2)),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    
    # Estimation using gradient descent and too large learning rate
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                       lr_cov = 10, use_nesterov_acc = FALSE))
    cov_pars <- c(0.4026051)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 4)
    
    # Prediction
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, params = list(optimizer_cov = "gradient_descent", 
                                                use_nesterov_acc = FALSE, lr_cov = 0.1))
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.000000, -0.796538, -0.796538, 0.000000)
    expected_cov <- c(0.1133436, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
                      0.1407783, 0.1407783, 0.0000000, 0.0000000, 0.1407783,
                      0.1407783, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.4070775)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
    # Predict variances
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,6,11,16)])),TOLERANCE2)
    # Predict response
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_response = TRUE)
    expected_mu <- c(0.5000000, 0.2279027, 0.2279027, 0.5000000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    # Prediction for only new groups
    group_test <- c(-1,-1,-2,-2)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-rep(0,4))),TOLERANCE2)
    expect_lt(sum(abs(pred$var-rep(0,0.4070775))),TOLERANCE2)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_response = TRUE)
    expect_lt(sum(abs(pred$mu-rep(0.5,4))),TOLERANCE2)
    # Prediction for only new cluster_ids
    cluster_ids_pred <- c(-1L,-1L,-2L,-2L)
    group_test <- c(1,99999,3,3)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, cluster_ids_pred = cluster_ids_pred,
                    predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-rep(0,4))),TOLERANCE2)
    expect_lt(sum(abs(pred$var-rep(0.4070771,4))),TOLERANCE2)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, cluster_ids_pred = cluster_ids_pred,
                    predict_response = TRUE)
    expect_lt(sum(abs(pred$mu-rep(0.5,4))),TOLERANCE2)
    
    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
    first_occurences <- match(unique(group), group)
    training_data_random_effects <- all_training_data_random_effects[first_occurences] 
    group_unique <- unique(group)
    pred_random_effects <- predict(gp_model, group_data_pred = group_unique)
    expect_lt(sum(abs(training_data_random_effects - pred_random_effects$mu)),1E-6)
    
    # Estimation using Nelder-Mead
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "nelder_mead"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.4027452)),TOLERANCE2)
    # Prediction
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-c(0.0000000, -0.7935873, -0.7935873, 0.0000000))),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$var)-c(0.1130051, 0.1401125, 0.1401125, 0.4027452))),TOLERANCE2)
    
    # Estimation using BFGS
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "bfgs"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.4025483)),TOLERANCE2)
    # Prediction
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-c(0.0000000, -0.7934523, -0.7934523, 0.0000000))),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$var)-c(0.1129896, 0.1400821, 0.1400821, 0.4025483))),TOLERANCE2)
    
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
    expect_lt(abs(nll-65.8590638),TOLERANCE2)
    
    # Do optimization using optim and e.g. Nelder-Mead
    gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
    opt <- optim(par=c(2), fn=gp_model$neg_log_likelihood, y=y, method="Brent", lower=0, upper=1E9)
    cov_pars <- c(0.40255)
    expect_lt(sum(abs(opt$par-cov_pars)),TOLERANCE)
    expect_lt(abs(opt$value-(65.2599674)),TOLERANCE)
  })
  
  test_that("Binary classification with multiple grouped random effects ", {
    
    probs <- pnorm(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.57341) < probs)
    gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                           y = y, likelihood = "bernoulli_probit",
                           params = list(optimizer_cov = "gradient_descent",
                                         lr_cov = 0.2, use_nesterov_acc = FALSE, maxit=100))
    expected_values <- c(0.3060671, 0.9328884, 0.3146682)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 37)
    
    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
    first_occurences_1 <- match(unique(group), group)
    first_occurences_2 <- match(unique(group2), group2)
    pred_random_effects <- all_training_data_random_effects[first_occurences_1,1]
    pred_random_slopes <- all_training_data_random_effects[first_occurences_1,3]
    pred_random_effects_crossed <- all_training_data_random_effects[first_occurences_2,2] 
    group_unique <- unique(group)
    group_data_pred = cbind(group_unique,rep(-1,length(group_unique)))
    x_pr = rep(0,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr)
    expect_lt(sum(abs(pred_random_effects - preds$mu)),1E-6)
    # Check whether random slopes are correct
    x_pr = rep(1,length(group_unique))
    preds2 <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr)
    expect_lt(sum(abs(pred_random_slopes - (preds2$mu-preds$mu))),1E-6)
    # Check whether crossed random effects are correct
    group_unique <- unique(group2)
    group_data_pred = cbind(rep(-1,length(group_unique)),group_unique)
    x_pr = rep(0,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr)
    expect_lt(sum(abs(pred_random_effects_crossed - preds$mu)),1E-6)
    
    # Prediction
    gp_model <- GPModel(likelihood = "bernoulli_probit", group_data = cbind(group,group2),
                        group_rand_coef_data = x, ind_effect_group_rand_coef = 1)
    group_data_pred = cbind(c(1,1,77),c(2,1,98))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2), predict_cov_mat = TRUE)
    expected_mu <- c(0.5195889, -0.6411954, 0.0000000)
    expected_cov <- c(0.3422367, 0.1554011, 0.0000000, 0.1554011,
                      0.3457334, 0.0000000, 0.0000000, 0.0000000, 1.8080000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
    # Predict variances
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE2)
    # Multiple random effects: training with Nelder-Mead
    gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                           y = y, likelihood = "bernoulli_probit",
                           params = list(optimizer_cov = "nelder_mead"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.3055487, 0.9300562, 0.3048811))),TOLERANCE2)
    # Multiple random effects: training with BFGS
    gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                           y = y, likelihood = "bernoulli_probit",
                           params = list(optimizer_cov = "bfgs"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.3030693, 0.9293106, 0.3037503))),TOLERANCE2)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.8,1.2),y=y)
    expect_lt(abs(nll-60.6422359),TOLERANCE)
    # Multiple cluster_ids
    gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                           y = y, cluster_ids = cluster_ids, likelihood = "bernoulli_probit",
                           params = list(optimizer_cov = "gradient_descent",
                                         lr_cov = 0.2, use_nesterov_acc = FALSE, maxit=100))
    expected_values <- c(0.1634433, 0.8952201, 0.3219087)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 42)
    # Prediction
    cluster_ids_pred = c(1,3,1)
    gp_model <- GPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                        cluster_ids = cluster_ids, likelihood = "bernoulli_probit")
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2), cluster_ids_pred = cluster_ids_pred, predict_cov_mat = TRUE)
    expected_mu <- c(-0.2159939, 0.0000000, 0.0000000)
    expected_cov <- c(0.4547941, 0.0000000, 0.0000000, 0.0000000,
                      1.7120000, 0.0000000, 0.0000000, 0.0000000, 1.8080000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
    # Only one RE and random coefficient
    probs <- pnorm(Z1 %*% b_gr_1 + Z3 %*% b_gr_3)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.957341) < probs)
    gp_model <- fitGPModel(group_data = group, group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                           y = y, likelihood = "bernoulli_probit",
                           params = list(optimizer_cov = "gradient_descent",
                                         lr_cov = 0.1, use_nesterov_acc = TRUE, maxit=100))
    expected_values <- c(1.00742383, 0.02612587)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 100)
  })
  
  
  test_that("Binary classification for combined Gaussian process and grouped random effects ", {
    
    probs <- pnorm(L %*% b_1 + Z1 %*% b_gr_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.67341) < probs)
    
    # Estimation using gradient descent
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                       lr_cov = 0.2, use_nesterov_acc = FALSE,
                                       convergence_criterion = "relative_change_in_parameters"))
    cov_pars <- c(0.3181509, 1.2788456, 0.1218680)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 55)
    
    # Prediction
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                           group_data = group, y = y, params = list(optimizer_cov = "gradient_descent", 
                                                                    use_nesterov_acc = FALSE, lr_cov = 0.2))
    coord_test <- cbind(c(0.1,0.21,0.7),c(0.9,0.91,0.55))
    group_test <- c(1,3,9999)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(0.1217634, -0.9592585, -0.2694489)
    expected_cov <- c(1.0745455607, 0.2190063794, 0.0040797451, 0.2190063794,
                      1.0089298170, 0.0000629706, 0.0040797451, 0.0000629706, 1.0449941968)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
    # Predict variances
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE2)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred = group_test, predict_response = TRUE)
    expected_mu <- c(0.5336859, 0.2492699, 0.4252731)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    
    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model)
    pred_GP <- predict(gp_model, gp_coords_pred = coords, group_data_pred=rep(-1,dim(coords)[1]))
    expect_lt(sum(abs(training_data_random_effects[,2] - pred_GP$mu)),1E-6)
    # Grouped REs
    preds <- predict(gp_model, group_data_pred = group, gp_coords_pred = coords)
    pred_RE <- preds$mu - pred_GP$mu
    expect_lt(sum(abs(training_data_random_effects[,1] - pred_RE)),1E-6)
    
    # Estimation using Nelder-Mead
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        group_data = group, likelihood = "bernoulli_probit")
    fit(gp_model, y = y, params = list(optimizer_cov = "nelder_mead", delta_rel_conv=1E-8))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.3181320, 1.2795124, 0.1218866))),TOLERANCE2)
    
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(1.1,0.9,0.2),y=y)
    expect_lt(abs(nll-65.7219266),TOLERANCE2)
    
    # Do optimization using optim and e.g. Nelder-Mead
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                        group_data = group, likelihood = "bernoulli_probit")
    capture.output( opt <- optim(par=c(1.5,1,0.1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead"), file='NUL')
    cov_pars <- c(0.3181509, 1.2788456, 0.1218680)
    expect_lt(sum(abs(opt$par-cov_pars)),TOLERANCE)
    expect_lt(abs(opt$value-(63.7432077)),TOLERANCE)
    expect_equal(as.integer(opt$counts[1]), 164)
  })
  
  
  test_that("Combined GP and grouped random effects model with random coefficients ", {
    
    probs <- pnorm(as.vector(L %*% b_1 + Z_SVC[,1] * L %*% b_2 + Z_SVC[,2] * L %*% b_3) + 
                     Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.9867234) < probs)
    
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                           group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                           y = y, likelihood = "bernoulli_probit",
                           params = list(optimizer_cov = "gradient_descent",
                                         lr_cov = 0.2, use_nesterov_acc = FALSE, maxit=10))
    expected_values <- c(0.09859312, 0.35813763, 0.50164573, 0.67372019,
                         0.08825524, 0.77807532, 0.10896128, 1.03921290, 0.09538707)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    
    # Prediction
    gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential", likelihood = "bernoulli_probit",
                        group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1)
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    group_data_pred = cbind(c(1,1,7),c(2,1,3))
    group_rand_coef_data_pred = c(0,0.1,0.3)
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             gp_rand_coef_data_pred=Z_SVC_test,
                             group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.9,0.8,1.2,1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE)
    expected_mu <- c(1.612451, 1.147407, -1.227187)
    expected_cov <- c(1.63468526, 1.02982815, -0.01916993, 1.02982815,
                      1.43601348, -0.03404720, -0.01916993, -0.03404720, 1.55017397)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.8,1.2,1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-71.4286594),TOLERANCE)
  })
  
  
  test_that("Combined GP and grouped random effects model with cluster_id's not constant ", {
    
    probs <- pnorm(L %*% b_1 + Z1 %*% b_gr_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
    
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                           y = y, cluster_ids = cluster_ids,likelihood = "bernoulli_probit",
                           params = list(optimizer_cov = "gradient_descent", lr_cov=0.2, use_nesterov_acc = FALSE))
    cov_pars <- c(0.276476226, 0.007278016, 0.132195703)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
    expect_equal(gp_model$get_num_optim_iter(), 261)
    
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    group_data_pred = c(1,1,9999)
    cluster_ids_pred = c(1,3,1)
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                        cluster_ids = cluster_ids,likelihood = "bernoulli_probit")
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test, group_data_pred = group_data_pred,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(1.5,1,0.15), predict_cov_mat = TRUE)
    expected_mu <- c(0.1074035, 0.0000000, 0.2945508)
    expected_cov <- c(0.98609786, 0.00000000, -0.02013244, 0.00000000,
                      2.50000000, 0.00000000, -0.02013244, 0.00000000, 2.28927616)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
  })
  
  
  test_that("Binary classification Gaussian process model with Vecchia approximation", {
    
    probs <- pnorm(L %*% b_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
    # Estimation using gradient descent
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        likelihood = "bernoulli_probit", vecchia_approx=TRUE, num_neighbors=n-1), file='NUL')
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                       convergence_criterion = "relative_change_in_parameters"))
    cov_pars <- c(0.9419234, 0.1866877)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 40)
    # Prediction
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", vecchia_approx=TRUE, num_neighbors=n-1,
                                           y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = FALSE, lr_cov=0.01)), file='NUL')
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.6595663, -0.6638940, 0.4997690)
    expected_cov <- c(0.6482224576, 0.5765285950, -0.0001030520, 0.5765285950,
                      0.6478191338, -0.0001163496, -0.0001030520, -0.0001163496, 0.4435551436)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
    # Predict variances
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE2)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE)
    expected_mu <- c(0.3037139, 0.3025143, 0.6612807)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
    expect_lt(abs(nll-63.6205917),TOLERANCE2)
    
    
    #######################
    ## Less neighbours than observations
    #######################
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        likelihood = "bernoulli_probit", vecchia_approx=TRUE, num_neighbors=30), file='NUL')
    fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                       convergence_criterion = "relative_change_in_parameters"))
    cov_pars <- c(1.101290, 0.207112)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 41)
    # Prediction
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           likelihood = "bernoulli_probit", vecchia_approx=TRUE, num_neighbors=30,
                                           y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = FALSE, lr_cov=0.01)), file='NUL')
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.6849454, -0.6911604, 0.5437782)
    expected_cov <- c(6.700096e-01, 5.989224e-01, -8.020211e-06, 5.989224e-01,
                      6.694618e-01, -2.692538e-06, -8.020211e-06, -2.692538e-06, 4.190919e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
    # Predict variances
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE2)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE)
    expected_mu <- c(0.2980473, 0.2963518, 0.6759756)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    # Evaluate approximate negative marginal log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
    expect_lt(abs(nll-63.4059092),TOLERANCE2)
    
    ###################
    ## Random coefficient GPs
    ###################
    probs <- pnorm(as.vector(L %*% b_1 + Z_SVC[,1] * L %*% b_2 + Z_SVC[,2] * L %*% b_3))
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.543) < probs)
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                                           y = y, likelihood = "bernoulli_probit", vecchia_approx=TRUE, num_neighbors=n-1,
                                           params = list(optimizer_cov = "gradient_descent",
                                                         lr_cov = 1, use_nesterov_acc = TRUE, acc_rate_cov=0.5, maxit=1000)), file='NUL')
    expected_values <- c(0.3701097, 0.2846740, 2.1160323, 0.3305266, 0.1241462, 0.1846456)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 39)
    # Prediction
    capture.output( gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC,
                                        cov_function = "exponential", likelihood = "bernoulli_probit",
                                        vecchia_approx=TRUE, num_neighbors=n-1), file='NUL')
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test, gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE)
    expected_mu <- c(0.18346008, 0.03479258, -0.17247579)
    expected_cov <- c(1.039879e+00, 7.521981e-01, -3.256500e-04, 7.521981e-01,
                      8.907289e-01, -6.719282e-05, -3.256500e-04, -6.719282e-05, 9.147899e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-65.1768199),TOLERANCE)
    
    ###################
    ##  Multiple cluster IDs
    ###################
    probs <- pnorm(L %*% b_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2978341) < probs)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, cluster_ids = cluster_ids,likelihood = "bernoulli_probit",
                                           vecchia_approx=TRUE, num_neighbors=n-1,
                                           params = list(optimizer_cov = "gradient_descent", lr_cov=0.2, use_nesterov_acc = FALSE)), file='NUL')
    cov_pars <- c(0.5085134, 0.2011667)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
    expect_equal(gp_model$get_num_optim_iter(), 20)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    cluster_ids_pred = c(1,3,1)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        cluster_ids = cluster_ids,likelihood = "bernoulli_probit"), file='NUL')
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(1.5,0.15), predict_cov_mat = TRUE)
    expected_mu <- c(0.1509569, 0.0000000, 0.9574946)
    expected_cov <- c(1.2225959453, 0.0000000000, 0.0003074858, 0.0000000000,
                      1.5000000000, 0.0000000000, 0.0003074858, 0.0000000000, 1.0761874845)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
  })
  
  test_that("Binary classification Gaussian process model with Wendland covariance function", {
    
    probs <- pnorm(L %*% b_1)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
    # Estimation using gradient descent
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland", 
                                           cov_fct_shape=0, cov_fct_taper_range=0.1,
                                           y = y, likelihood = "bernoulli_probit",
                                           params = list(optimizer_cov = "gradient_descent", std_dev = FALSE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5)), file='NUL')
    cov_pars <- c(0.5748213)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 17)
    # Prediction
    coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    expected_mu <- c(-0.05584777, -0.05921225, 0.05156614)
    expected_cov <- c(0.5733359, 0.4223618, 0.0000000, 0.4223618, 0.5727027, 0.0000000, 0.0000000, 0.0000000, 0.5318419)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
    # Predict variances
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE2)
    # Predict response
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE)
    expected_mu <- c(0.4822433, 0.4811706, 0.5166166)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
    
  })
  
}

test_that("Binary classification with linear predictor and grouped random effects model ", {
  
  probs <- pnorm(Z1 %*% b_gr_1 + X%*%beta)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.542) < probs)
  
  # Estimation using gradient descent
  gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
  fit(gp_model, y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                          optimizer_coef = "gradient_descent", 
                                          use_nesterov_acc = FALSE, lr_cov = 0.05, lr_coef = 1))
  cov_pars <- c(0.408371)
  coef <- c(-0.1113766, 1.5182602)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
  expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE)
  expect_equal(gp_model$get_num_optim_iter(), 52)
  # Estimation using gradient descent and Nesterov acceleration
  gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                         y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                                   optimizer_coef = "gradient_descent", lr_cov = 0.05, lr_coef = 1,
                                                   use_nesterov_acc = TRUE, acc_rate_cov = 0.2, acc_rate_coef = 0.1))
  cov_pars <- c(0.4072038)
  coef <- c(-0.1113238, 1.5178344)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
  expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE)
  expect_equal(gp_model$get_num_optim_iter(), 43)
  
  # Estimation using Nelder-Mead
  gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                         y = y, X=X, params = list(optimizer_cov = "nelder_mead",
                                                   optimizer_coef = "nelder_mead", delta_rel_conv=1e-12))
  cov_pars <- c(0.3999745)
  coef <- c(-0.1109537, 1.5149593)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
  expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE)
  expect_equal(gp_model$get_num_optim_iter(), 178)
  
  # # Estimation using BFGS
  # Does not converge to correct solution (version 0.7.2, 21.02.2022)
  # gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
  #                        y = y, X=X, params = list(optimizer_cov = "bfgs",
  #                                                  optimizer_coef = "bfgs"))
  # cov_pars <- c(0.3999729)
  # coef <- c(-0.1109532, 1.5149592)
  # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
  # expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE)
  # expect_equal(gp_model$get_num_optim_iter(), 17)
  
  # Prediction
  gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                         y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                                   optimizer_coef = "gradient_descent",
                                                   use_nesterov_acc=FALSE, lr_coef=1))
  X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
  group_test <- c(1,3,3,9999)
  pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                  predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(-0.81132177, -0.08574621, 0.21768660, 1.40591471)
  expected_cov <- c(0.1380239, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.1688251, 0.1688251, 0.0000000, 0.0000000, 0.1688251, 0.1688251, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.4051196)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
  # Predict response
  pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test, predict_response = TRUE)
  expected_mu <- c(0.2234684, 0.4683921, 0.5797885, 0.8821984)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  
  # Predict training data random effects
  all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
  first_occurences <- match(unique(group), group)
  training_data_random_effects <- all_training_data_random_effects[first_occurences] 
  group_unique <- unique(group)
  X_zero <- cbind(rep(0,length(group_unique)),rep(0,length(group_unique)))
  pred_random_effects <- predict(gp_model, group_data_pred = group_unique, X_pred = X_zero)
  expect_lt(sum(abs(training_data_random_effects - pred_random_effects$mu)),1E-6)
  
  # Standard deviations
  capture.output( gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit", 
                                         y = y, X=X, params = list(std_dev = TRUE, optimizer_cov = "gradient_descent",
                                                                   optimizer_coef = "gradient_descent", 
                                                                   use_nesterov_acc = TRUE, lr_cov = 0.1, lr_coef = 1)),
                  file='NUL')
  cov_pars <- c(0.4004722 )
  coef <- c(-0.1112265, 0.2565839, 1.5155559, 0.2636460)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE2)
  
  # Providing initial covariance parameters and coefficients
  cov_pars <- c(1)
  coef <- c(2,5)
  gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                         y = y, X=X, params = list(maxit=0, init_cov_pars=cov_pars, init_coef=coef))
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
  expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE)
  
  if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
    # Large data
    n_L <- 1e6 # number of samples
    m_L <- n_L/10 # number of categories / levels for grouping variable
    group_L <- rep(1,n_L) # grouping variable
    for(i in 1:m_L) group_L[((i-1)*n_L/m_L+1):(i*n_L/m_L)] <- i
    keps <- 1E-10
    b1_L <- qnorm(sim_rand_unif(n=m_L, init_c=0.671)*(1-keps) + keps/2)
    X_L <- cbind(rep(1,n_L),sim_rand_unif(n=n_L, init_c=0.8671)-0.5) # design matrix / covariate data for fixed effect
    probs_L <- pnorm(b1_L[group_L] + X_L%*%beta)
    y_L <- as.numeric(sim_rand_unif(n=n_L, init_c=0.12378)*(1-keps) + keps/2 < probs_L)
    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- fitGPModel(group_data = group_L, likelihood = "bernoulli_probit",
                           y = y_L, X=X_L, params = list(optimizer_cov = "gradient_descent",
                                                         optimizer_coef = "gradient_descent", lr_cov = 0.05, lr_coef = 0.1,
                                                         use_nesterov_acc = TRUE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.9771949)),TOLERANCE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-c(0.09784701, 1.99437261))),TOLERANCE)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    
    # Estimation using Nelder-Mead
    gp_model <- fitGPModel(group_data = group_L, likelihood = "bernoulli_probit",
                           y = y_L, X=X_L, params = list(optimizer_cov = "nelder_mead",
                                                         optimizer_coef = "nelder_mead"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.9812891)),TOLERANCE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-c(0.09880008, 1.99563966))),TOLERANCE)
    expect_equal(gp_model$get_num_optim_iter(), 117)
  }
  
})

test_that("Binary classification with linear predictor and Gaussian process model ", {
  
  probs <- pnorm(L %*% b_1 + X%*%beta)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.199) < probs)
  # Estimation
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                         y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                                   optimizer_coef = "gradient_descent",
                                                   use_nesterov_acc = TRUE, lr_cov=0.1, lr_coef = 0.1, maxit=100))
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(1.2665345, 0.2855689))),TOLERANCE)
  expect_lt(sum(abs(as.vector(gp_model$get_coef())-c(0.203884, 1.466401))),TOLERANCE)
  expect_equal(gp_model$get_num_optim_iter(), 17)
  
  # Prediction
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  X_test <- cbind(rep(1,3),c(-0.5,0.2,1))
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, X_pred = X_test,
                  predict_var = TRUE, predict_response = FALSE)
  expected_mu <- c(-0.7241663, 0.2901968, 2.5453020)
  expected_var <- c(0.7541041, 0.7511734, 0.4414069)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE)
  expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE)
  
  # Estimation using Nelder-Mead
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                         y = y, X=X, params = list(optimizer_cov = "nelder_mead",
                                                   optimizer_coef = "nelder_mead", maxit=1000, delta_rel_conv=1e-12))
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(1.271740, 0.287552 ))),TOLERANCE)
  expect_lt(sum(abs(as.vector(gp_model$get_coef())-c(0.199947, 1.466620 ))),TOLERANCE)
  expect_equal(gp_model$get_num_optim_iter(), 343)
  
  # Standard deviations
  capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit", 
                                         y = y, X=X, params = list(std_dev = TRUE, optimizer_cov = "gradient_descent",
                                                                   optimizer_coef = "gradient_descent", 
                                                                   use_nesterov_acc = TRUE, lr_cov = 0.1, lr_coef = 0.1, maxit=100)),
                  file='NUL')
  coef <- c(0.2038840, 0.5404603, 1.4664012, 0.3028299)
  expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE2)
  
})


test_that("Binary classification with Gaussian process model and logit link function", {
  
  probs <- 1/(1+exp(- L %*% b_1))
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
  # Estimation 
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_logit",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE, lr_cov=0.01))
  cov_pars <- c(1.4300136, 0.1891952)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
  expect_equal(gp_model$get_num_optim_iter(), 85)
  # Prediction
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(-0.7792960, -0.7876208, 0.5476390)
  expected_cov <- c(1.024267e+00, 9.215206e-01, 5.561435e-05, 9.215206e-01, 1.022897e+00,
                    2.028618e-05, 5.561435e-05, 2.028618e-05, 7.395747e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
  # Predict response
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var=TRUE, predict_response = TRUE)
  expected_mu <- c(0.3442815, 0.3426873, 0.6159933)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(pred$var-expected_mu*(1-expected_mu))),TOLERANCE2)
  # Evaluate approximate negative marginal log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
  expect_lt(abs(nll-66.299571),TOLERANCE2)
})


test_that("Poisson regression ", {
  
  # Single level grouped random effects
  mu <- exp(Z1 %*% b_gr_1)
  y <- qpois(sim_rand_unif(n=n, init_c=0.04532), lambda = mu)
  # Estimation 
  gp_model <- fitGPModel(group_data = group, likelihood = "poisson",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE, lr_cov=0.1))
  cov_pars <- c(0.4033406)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
  expect_equal(gp_model$get_num_optim_iter(), 8)
  # Prediction
  group_test <- c(1,3,3,9999)
  pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(0.07765297, -0.87488533, -0.87488533, 0.00000000)
  expected_cov <- c(0.07526284, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                    0.15041230, 0.15041230, 0.00000000, 0.00000000, 0.15041230,
                    0.15041230, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.40334058)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
  # Predict response
  pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var=TRUE, predict_response = TRUE)
  expected_mu <- c(1.1221925, 0.4494731, 0.4494731, 1.2234446)
  expected_var <- c(1.2206301, 0.4822647, 0.4822647, 1.9670879)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE2)
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
  expect_lt(abs(nll-140.4554806),TOLERANCE)
  
  # Multiple random effects
  mu <- exp(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
  y <- qpois(sim_rand_unif(n=n, init_c=0.74532), lambda = mu)
  # Estimation 
  gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                         ind_effect_group_rand_coef = 1, likelihood = "poisson",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE, lr_cov=0.1))
  cov_pars <- c(0.4069344, 1.6988978, 1.3415016)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
  expect_equal(gp_model$get_num_optim_iter(), 7)
  # Prediction
  group_data_pred = cbind(c(1,1,77),c(2,1,98))
  group_rand_coef_data_pred = c(0,0.1,0.3)
  pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                           cov_pars = c(0.9,0.8,1.2), predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(0.92620057, -0.08200469, 0.00000000)
  expected_cov <- c(0.07730896, 0.04403442, 0.00000000, 0.04403442, 0.11600469,
                    0.00000000, 0.00000000, 0.00000000, 1.80800000)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
  
  # Gaussian process model
  mu <- exp(L %*% b_1)
  y <- qpois(sim_rand_unif(n=n, init_c=0.435), lambda = mu)
  # Estimation 
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "poisson",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE))
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(1.1853922, 0.1500197))),TOLERANCE2)
  expect_equal(gp_model$get_num_optim_iter(), 6)
  # Prediction
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(0.4329068, 0.4042531, 0.6833738)
  expected_cov <- c(6.550626e-01, 5.553938e-01, -8.406290e-06, 5.553938e-01, 6.631295e-01, -7.658261e-06, -8.406290e-06, -7.658261e-06, 4.170417e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
  # Predict response
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var=TRUE, predict_response = TRUE)
  expected_mu <- c(2.139213, 2.087188, 2.439748)
  expected_var <- c(6.373433, 6.185895, 5.519896)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE)
  # Evaluate approximate negative marginal log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
  expect_lt(abs(nll-195.03708036),TOLERANCE2)
})


test_that("Gamma regression ", {
  # Single level grouped random effects
  mu <- exp(Z1 %*% b_gr_1)
  shape <- 1
  y <- qgamma(sim_rand_unif(n=n, init_c=0.04532), scale = mu/shape, shape = shape)
  # Estimation 
  gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE, lr_cov=0.1))
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.5174554)),TOLERANCE2)
  expect_equal(gp_model$get_num_optim_iter(), 6)
  # Prediction
  group_test <- c(1,3,3,9999)
  pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(0.2095341, -0.9170767, -0.9170767, 0.0000000)
  expected_cov <- c(0.08105393, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                    0.09842279, 0.09842279, 0.00000000, 0.00000000, 0.09842279,
                    0.09842279, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.51745540)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
  # Predict response
  pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var=TRUE, predict_response = TRUE)
  expected_mu <- c(1.2841038, 0.4198468, 0.4198468, 1.2952811)
  expected_var <- c(1.9273575, 0.2127346, 0.2127346, 3.9519573)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(pred$var-expected_var)),TOLERANCE2)
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
  expect_lt(abs(nll-105.676137),TOLERANCE)
  
  # Multiple random effects
  mu <- exp(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
  y <- qgamma(sim_rand_unif(n=n, init_c=0.04532), scale = mu/shape, shape = shape)
  # Estimation 
  gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                         ind_effect_group_rand_coef = 1, likelihood = "gamma",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE, lr_cov=0.1))
  cov_pars <- c(0.5050690, 1.2043329, 0.5280103)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
  expect_equal(gp_model$get_num_optim_iter(), 10)
  # Prediction
  group_data_pred = cbind(c(1,1,77),c(2,1,98))
  group_rand_coef_data_pred = c(0,0.1,0.3)
  pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                           cov_pars = c(0.9,0.8,1.2), predict_var = TRUE, predict_response = FALSE)
  expected_mu <- c(0.1121777, 0.1972216, 0.0000000)
  expected_var <- c(0.2405621, 0.2259258, 1.8080000)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOLERANCE2)
  
  # Gaussian process model
  mu <- exp(L %*% b_1)
  y <- qgamma(sim_rand_unif(n=n, init_c=0.435), scale = mu/shape, shape = shape)
  # Estimation 
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "gamma",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE, lr_cov=0.1))
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(1.0649094, 0.2738999))),TOLERANCE2)
  expect_equal(gp_model$get_num_optim_iter(), 8)
  # Prediction
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(0.3376250, 0.3023855, 0.7810425)
  expected_cov <- c(0.4567916157, 0.4033257822, -0.0002256179, 0.4033257822, 0.4540419202, 
                    -0.0002258048, -0.0002256179, -0.0002258048, 0.3368598330)
  expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE2)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE2)
  # Evaluate approximate negative marginal log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
  expect_lt(abs(nll-154.4561783),TOLERANCE2)
})


# Avoid being tested on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  test_that("Saving a GPModel and loading from file works for non-Gaussian data", {
    
    probs <- pnorm(Z1 %*% b_gr_1 + X%*%beta)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.542) < probs)
    # Train model
    gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                           y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                                     optimizer_coef = "gradient_descent",
                                                     use_nesterov_acc=TRUE))
    # Make predictions
    X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
    group_test <- c(1,3,3,9999)
    pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                    predict_cov_mat = TRUE, predict_response = FALSE)
    # Predict response
    pred_resp <- predict(gp_model, y=y, group_data_pred = group_test,
                         X_pred = X_test, predict_var = TRUE, predict_response = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model,filename = filename)
    # Delete model
    rm(gp_model)
    # Load from file and make predictions again
    gp_model_loaded <- loadGPModel(filename = filename)
    pred_loaded <- predict(gp_model_loaded, group_data_pred = group_test,
                           X_pred = X_test, predict_cov_mat = TRUE, predict_response = FALSE)
    pred_resp_loaded <- predict(gp_model_loaded, y=y, group_data_pred = group_test,
                                X_pred = X_test, predict_var = TRUE, predict_response = TRUE)
    
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    expect_equal(pred_resp$mu, pred_resp_loaded$mu)
    expect_equal(pred_resp$var, pred_resp_loaded$var)
  })
}
