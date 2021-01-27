context("generalized_GPModel")

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
Z_SVC <- matrix(sim_rand_unif(n=n*2, init_c=0.6), ncol=2) # covariate data for random coeffients
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
n_obs_gr <- n/m # number of sampels per group
group2 <- rep(1,n) # grouping variable
for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
b_gr_2 <- qnorm(sim_rand_unif(n=n_obs_gr, init_c=0.36))
# Grouped random slope / coefficient
x <- cos((1:n-n/2)^2*5.5*pi/n) # covariate data for random slope
Z3 <- diag(x) %*% Z1
b_gr_3 <- qnorm(sim_rand_unif(n=m, init_c=0.5678))
# Data for linear mixed effects model
X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # desing matrix / covariate data for fixed effect
beta <- c(0.1,2) # regression coefficents
# cluster_ids 
cluster_ids <- c(rep(1,0.4*n),rep(2,0.6*n))


print("Ignore [GPBoost] [Fatal]")
test_that("Binary classification with Gaussian process model ", {

  probs <- pnorm(L %*% b_1)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
  # Label needs to have correct format
  expect_error(fitGPModel(gp_coords = coords, cov_function = "exponential",
                          likelihood = "bernoulli_probit",
                          y = b_1, params = list(optimizer_cov = "gradient_descent")))
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
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 40)
  # Estimation using gradient descent and Nesterov acceleration
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
  fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                     lr_cov = 0.01, use_nesterov_acc = TRUE, acc_rate_cov = 0.5))
  cov_pars2 <- c(0.9646422, 0.1844797)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars2)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 26)
  
  # Prediction
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc=FALSE))
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(-0.6595663, -0.6638940, 0.4997690)
  expected_cov <- c(0.6482224576, 0.5765285950, -0.0001030520, 0.5765285950,
                    0.6478191338, -0.0001163496, -0.0001030520, -0.0001163496, 0.4435551436)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict variances
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, predict_response = FALSE)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
  # Predict response
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var=TRUE, predict_response = TRUE)
  expected_mu <- c(0.3037139, 0.3025143, 0.6612807)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(pred$var-expected_mu*(1-expected_mu))),1E-6)

  # Evaluate approximate negative marginal log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
  expect_lt(abs(nll-63.6205917),1E-6)
  
  # Do optimization using optim and e.g. Nelder-Mead
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
  opt <- optim(par=c(1,0.1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
  cov_pars <- c(0.9419234, 0.1866877)
  expect_lt(sum(abs(opt$par-cov_pars)),1E-3)
  expect_lt(abs(opt$value-(63.6126363)),1E-5)
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
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 39)
  # Prediction
  gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential", likelihood = "bernoulli_probit")
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           gp_rand_coef_data_pred=Z_SVC_test,
                           cov_pars = c(1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE)
  expected_mu <- c(0.18346008, 0.03479258, -0.17247579)
  expected_cov <- c(1.039879e+00, 7.521981e-01, -3.256500e-04, 7.521981e-01,
                    8.907289e-01, -6.719282e-05, -3.256500e-04, -6.719282e-05, 9.147899e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(1,0.1,0.8,0.15,1.1,0.08),y=y)
  expect_lt(abs(nll-65.1768199),1E-5)
  
  ###################
  ##  Multiple cluster IDs
  ###################
  probs <- pnorm(L %*% b_1)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.2978341) < probs)
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         y = y, cluster_ids = cluster_ids,likelihood = "bernoulli_probit",
                         params = list(optimizer_cov = "gradient_descent", lr_cov=0.2, use_nesterov_acc=FALSE))
  cov_pars <- c(0.5085134, 0.2011667)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
  expect_equal(gp_model$get_num_optim_iter(), 20)
  # Prediction
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  cluster_ids_pred = c(1,3,1)
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                      cluster_ids = cluster_ids,likelihood = "bernoulli_probit")
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           cluster_ids_pred = cluster_ids_pred,
                           cov_pars = c(1.5,0.15), predict_cov_mat = TRUE)
  expected_mu <- c(0.1509569, 0.0000000, 0.9574946)
  expected_cov <- c(1.2225959453, 0.0000000000, 0.0003074858, 0.0000000000,
                    1.5000000000, 0.0000000000, 0.0003074858, 0.0000000000, 1.0761874845)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})


test_that("Binary classification with grouped random effects model ", {
  
  probs <- pnorm(Z1 %*% b_gr_1)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.823431) < probs)
  
  # Estimation using gradient descent
  gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
  fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                     lr_cov = 0.1, use_nesterov_acc = FALSE,
                                     convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.40255)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 62)
  
  # Estimation using gradient descent and Nesterov acceleration
  gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
  fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                     lr_cov = 0.1, use_nesterov_acc = TRUE, acc_rate_cov = 0.5))
  cov_pars2 <- c(0.4012595)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars2)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 10)
  
  # Estimation using gradient descent and too large learning rate
  gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
  fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                     lr_cov = 10, use_nesterov_acc = FALSE))
  cov_pars <- c(0.4026051)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
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
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict variances
  pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,6,11,16)])),1E-6)
  # Predict response
  pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_response = TRUE)
  expected_mu <- c(0.5000000, 0.2279027, 0.2279027, 0.5000000)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  # Prediction for only new groups
  group_test <- c(-1,-1,-2,-2)
  pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
  expect_lt(sum(abs(pred$mu-rep(0,4))),1E-6)
  expect_lt(sum(abs(pred$var-rep(0,0.4070775))),1E-6)
  pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_response = TRUE)
  expect_lt(sum(abs(pred$mu-rep(0.5,4))),1E-6)
  # Prediction for only new cluster_ids
  cluster_ids_pred <- c(-1,-1,-2,-2)
  group_test <- c(1,3,3,9999)
  pred <- predict(gp_model, y=y, group_data_pred = group_test, cluster_ids_pred = cluster_ids_pred,
                  predict_var = TRUE, predict_response = FALSE)
  expect_lt(sum(abs(pred$mu-rep(0,4))),1E-6)
  expect_lt(sum(abs(pred$var-rep(0,0.4070775))),1E-6)
  pred <- predict(gp_model, y=y, group_data_pred = group_test, cluster_ids_pred = cluster_ids_pred,
                  predict_response = TRUE)
  expect_lt(sum(abs(pred$mu-rep(0.5,4))),1E-6)
  
  # Evaluate approximate negative marginal log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
  expect_lt(abs(nll-65.8590638),1E-6)
  
  # Do optimization using optim and e.g. Nelder-Mead
  gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
  opt <- optim(par=c(2), fn=gp_model$neg_log_likelihood, y=y, method="Brent", lower=0, upper=1E9)
  cov_pars <- c(0.40255)
  expect_lt(sum(abs(opt$par-cov_pars)),1E-3)
  expect_lt(abs(opt$value-(65.2599674)),1E-5)
  
  # Multiple random effects
  probs <- pnorm(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.57341) < probs)
  gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                         y = y, likelihood = "bernoulli_probit",
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.2, use_nesterov_acc = FALSE, maxit=100))
  expected_values <- c(0.3060671, 0.9328884, 0.3146682)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 37)
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
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict variances
  pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                           cov_pars = c(0.9,0.8,1.2), predict_var = TRUE)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
  
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.8,1.2),y=y)
  expect_lt(abs(nll-60.6422359),1E-5)
  
  # Multiple cluster_ids
  gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                         y = y, cluster_ids = cluster_ids, likelihood = "bernoulli_probit",
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.2, use_nesterov_acc = FALSE, maxit=100))
  expected_values <- c(0.1634433, 0.8952201, 0.3219087)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
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
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})


test_that("Binary classification for combined Gaussian process and grouped random effects model ", {
  
  probs <- pnorm(L %*% b_1 + Z1 %*% b_gr_1)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.67341) < probs)
  
  # Estimation using gradient descent
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                      group_data = group, likelihood = "bernoulli_probit")
  fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                     lr_cov = 0.2, use_nesterov_acc = FALSE,
                                     convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.3181509, 1.2788456, 0.1218680)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
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
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict variances
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred = group_test, predict_var = TRUE, predict_response = FALSE)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
  # Predict response
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred = group_test, predict_response = TRUE)
  expected_mu <- c(0.5336859, 0.2492699, 0.4252731)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  
  # Evaluate approximate negative marginal log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(1.1,0.9,0.2),y=y)
  expect_lt(abs(nll-65.7219266),1E-6)
  
  # Do optimization using optim and e.g. Nelder-Mead
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                      group_data = group, likelihood = "bernoulli_probit")
  opt <- optim(par=c(1.5,1,0.1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
  cov_pars <- c(0.3181509, 1.2788456, 0.1218680)
  expect_lt(sum(abs(opt$par-cov_pars)),1E-3)
  expect_lt(abs(opt$value-(63.7432077)),1E-5)
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
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
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
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.8,1.2,1,0.1,0.8,0.15,1.1,0.08),y=y)
  expect_lt(abs(nll-71.4286594),1E-5)
})


test_that("Combined GP and grouped random effects model with cluster_id's not constant ", {
  
  probs <- pnorm(L %*% b_1 + Z1 %*% b_gr_1)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
  
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                         y = y, cluster_ids = cluster_ids,likelihood = "bernoulli_probit",
                         params = list(optimizer_cov = "gradient_descent", lr_cov=0.2, use_nesterov_acc = FALSE))
  cov_pars <- c(0.276476226, 0.007278016, 0.132195703)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
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
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})


test_that("Binary classification Gaussian process model with Vecchia approximation", {
  
  probs <- pnorm(L %*% b_1)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
  
  # Estimation using gradient descent
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                      likelihood = "bernoulli_probit", vecchia_approx=TRUE, num_neighbors=n-1)
  fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                     lr_cov = 0.1, use_nesterov_acc = FALSE,
                                     convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.9419234, 0.1866877)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 40)
  
  # Prediction
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         likelihood = "bernoulli_probit", vecchia_approx=TRUE, num_neighbors=n-1,
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = FALSE))
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(-0.6595663, -0.6638940, 0.4997690)
  expected_cov <- c(0.6482224576, 0.5765285950, -0.0001030520, 0.5765285950,
                    0.6478191338, -0.0001163496, -0.0001030520, -0.0001163496, 0.4435551436)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict variances
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, predict_response = FALSE)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
  # Predict response
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE)
  expected_mu <- c(0.3037139, 0.3025143, 0.6612807)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  
  # Evaluate approximate negative marginal log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
  expect_lt(abs(nll-63.6205917),1E-6)
  
  # Do optimization using optim and e.g. Nelder-Mead
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_probit")
  opt <- optim(par=c(1,0.1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
  cov_pars <- c(0.9419234, 0.1866877)
  expect_lt(sum(abs(opt$par-cov_pars)),1E-3)
  expect_lt(abs(opt$value-(63.6126363)),1E-5)
  expect_equal(as.integer(opt$counts[1]), 47)
  
  #######################
  ## Less neighbours than observations
  #######################
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                      likelihood = "bernoulli_probit", vecchia_approx=TRUE, num_neighbors=30)
  fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", 
                                     lr_cov = 0.1, use_nesterov_acc = FALSE,
                                     convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(1.101290, 0.207112)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 41)
  # Prediction
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         likelihood = "bernoulli_probit", vecchia_approx=TRUE, num_neighbors=30,
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = FALSE))
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(-0.6849454, -0.6911604, 0.5437782)
  expected_cov <- c(6.700096e-01, 5.989224e-01, -8.020211e-06, 5.989224e-01,
                    6.694618e-01, -2.692538e-06, -8.020211e-06, -2.692538e-06, 4.190919e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict variances
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE, predict_response = FALSE)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
  # Predict response
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_response = TRUE)
  expected_mu <- c(0.2980473, 0.2963518, 0.6759756)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  # Evaluate approximate negative marginal log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
  expect_lt(abs(nll-63.4059092),1E-6)
  
  ###################
  ## Random coefficient GPs
  ###################
  probs <- pnorm(as.vector(L %*% b_1 + Z_SVC[,1] * L %*% b_2 + Z_SVC[,2] * L %*% b_3))
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.543) < probs)
  # Fit model
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                         y = y, likelihood = "bernoulli_probit", vecchia_approx=TRUE, num_neighbors=n-1,
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 1, use_nesterov_acc = TRUE, acc_rate_cov=0.5, maxit=1000))
  expected_values <- c(0.3701097, 0.2846740, 2.1160323, 0.3305266, 0.1241462, 0.1846456)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 39)
  # Prediction
  gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC,
                      cov_function = "exponential", likelihood = "bernoulli_probit",
                      vecchia_approx=TRUE, num_neighbors=n-1)
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test, gp_rand_coef_data_pred=Z_SVC_test,
                           cov_pars = c(1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE)
  expected_mu <- c(0.18346008, 0.03479258, -0.17247579)
  expected_cov <- c(1.039879e+00, 7.521981e-01, -3.256500e-04, 7.521981e-01,
                    8.907289e-01, -6.719282e-05, -3.256500e-04, -6.719282e-05, 9.147899e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(1,0.1,0.8,0.15,1.1,0.08),y=y)
  expect_lt(abs(nll-65.1768199),1E-5)
  
  ###################
  ##  Multiple cluster IDs
  ###################
  probs <- pnorm(L %*% b_1)
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.2978341) < probs)
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         y = y, cluster_ids = cluster_ids,likelihood = "bernoulli_probit",
                         vecchia_approx=TRUE, num_neighbors=n-1,
                         params = list(optimizer_cov = "gradient_descent", lr_cov=0.2, use_nesterov_acc = FALSE))
  cov_pars <- c(0.5085134, 0.2011667)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
  expect_equal(gp_model$get_num_optim_iter(), 20)
  # Prediction
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  cluster_ids_pred = c(1,3,1)
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                      cluster_ids = cluster_ids,likelihood = "bernoulli_probit")
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           cluster_ids_pred = cluster_ids_pred,
                           cov_pars = c(1.5,0.15), predict_cov_mat = TRUE)
  expected_mu <- c(0.1509569, 0.0000000, 0.9574946)
  expected_cov <- c(1.2225959453, 0.0000000000, 0.0003074858, 0.0000000000,
                    1.5000000000, 0.0000000000, 0.0003074858, 0.0000000000, 1.0761874845)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})


test_that("Binary classification with linear predictor and grouped random effects model ", {

  probs <- pnorm(Z1 %*% b_gr_1 + X%*%beta )
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.542) < probs)
  
  # Estimation using gradient descent
  gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
  fit(gp_model, y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                          optimizer_coef = "gradient_descent", 
                                          use_nesterov_acc = FALSE, lr_cov = 0.05, lr_coef = 0.01))
  cov_pars <- c(0.3944304)
  coef <- c(-0.1084191, 1.5093854)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 37)
  # Estimation using gradient descent and Nesterov acceleration
  gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                         y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                          optimizer_coef = "gradient_descent", lr_cov = 0.05, lr_coef = 0.005,
                                          use_nesterov_acc = TRUE, acc_rate_cov = 0.2, acc_rate_coef = 0.1))
  cov_pars <- c(0.3929977)
  coef <- c(-0.1087517, 1.5075622)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 80)
  # Defaul choices
  gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",y = y, X=X)
  # summary(gp_model)
  cov_pars <- c(0.4142176)
  coef <- c(-0.111881, 1.5211917)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 104)
  # # Compare to lme4
  # library(lme4)
  # mod <- glmer(y~X.2 + (1|group),data=data.frame(y=y,X=X,group=group),family=binomial(link = "probit"))
  # summary(mod)
  
  # Prediction
  gp_model <- fitGPModel(group_data = group, likelihood = "bernoulli_probit",
                         y = y, X=X, params = list(optimizer_cov = "gradient_descent",
                                                   optimizer_coef = "gradient_descent",
                                                   use_nesterov_acc=FALSE))
  X_test <- cbind(rep(1,4),c(-0.5,0.2,0.4,1))
  group_test <- c(1,3,3,9999)
  pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test,
                  predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(-0.81474865, -0.09028342, 0.21433362,1.41101818)
  expected_cov <- c(0.1398763, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.1717424,
                    0.1717424, 0.0000000, 0.0000000, 0.1717424, 0.1717424, 0.0000000,
                    0.0000000, 0.0000000, 0.0000000, 0.4203508)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict response
  pred <- predict(gp_model, y=y, group_data_pred = group_test, X_pred = X_test, predict_response = TRUE)
  expected_mu <- c(0.2226949, 0.4667648, 0.5784791, 0.8817843)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
})

test_that("Binary classification with Gaussian process model and logit link function", {
  
  probs <- 1/(1+exp(- L %*% b_1))
  y <- as.numeric(sim_rand_unif(n=n, init_c=0.2341) < probs)
  # Estimation 
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "bernoulli_logit",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE))
  cov_pars <- c(1.4300136, 0.1891952)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 85)
  # Prediction
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(-0.7792960, -0.7876208, 0.5476390)
  expected_cov <- c(1.024267e+00, 9.215206e-01, 5.561435e-05, 9.215206e-01, 1.022897e+00,
                    2.028618e-05, 5.561435e-05, 2.028618e-05, 7.395747e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict response
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var=TRUE, predict_response = TRUE)
  expected_mu <- c(0.3442815, 0.3426873, 0.6159933)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(pred$var-expected_mu*(1-expected_mu))),1E-6)
  # Evaluate approximate negative marginal log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
  expect_lt(abs(nll-66.299571),1E-6)
})


test_that("Poisson regression ", {
  
  # Single level grouped random effects
  mu <- exp(Z1 %*% b_gr_1)
  y <- qpois(sim_rand_unif(n=n, init_c=0.04532), lambda = mu)
  # Estimation 
  gp_model <- fitGPModel(group_data = group, likelihood = "poisson",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE))
  cov_pars <- c(0.4190871)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 67)
  # Prediction
  group_test <- c(1,3,3,9999)
  pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(0.07820113, -0.88736752, -0.88736752, 0.00000000)
  expected_cov <- c(0.07576021, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                    0.15376286, 0.15376286, 0.00000000, 0.00000000, 0.15376286,
                    0.15376286, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.41908708)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict response
  pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var=TRUE, predict_response = TRUE)
  expected_mu <- c(1.1230871, 0.4446419, 0.4446419, 1.2331151)
  expected_var <- c(1.2223583, 0.4775035, 0.4775035, 2.0246838)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(pred$var-expected_var)),1E-6)
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
  expect_lt(abs(nll-140.4554806),1E-5)
  
  # Multiple random effects
  mu <- exp(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
  y <- qpois(sim_rand_unif(n=n, init_c=0.74532), lambda = mu)
  # Estimation 
  gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                         ind_effect_group_rand_coef = 1, likelihood = "poisson",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE))
  cov_pars <- c(0.422674, 1.673567, 1.325769)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 58)
  # Prediction
  group_data_pred = cbind(c(1,1,77),c(2,1,98))
  group_rand_coef_data_pred = c(0,0.1,0.3)
  pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                           cov_pars = c(0.9,0.8,1.2), predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(0.92620057, -0.08200469, 0.00000000)
  expected_cov <- c(0.07730896, 0.04403442, 0.00000000, 0.04403442, 0.11600469,
                    0.00000000, 0.00000000, 0.00000000, 1.80800000)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict response
  pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                           cov_pars = c(0.9,0.8,1.2), predict_var = TRUE, predict_response = TRUE)
  expected_mu <- c(2.6244072, 0.9762834, 2.4694612)
  expected_var <- c(3.177997, 1.093519, 33.559738)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(pred$var-expected_var)),1E-6)
  
  # Gaussian process model
  mu <- exp(L %*% b_1)
  y <- qpois(sim_rand_unif(n=n, init_c=0.435), lambda = mu)
  # Estimation 
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "poisson",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE))
  cov_pars <- c(1.193947, 0.152378)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 17)
  # Prediction
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(0.4356454, 0.4067834, 0.6852349)
  expected_cov <- c(6.525332e-01, 5.537427e-01, -8.722219e-06, 5.537427e-01, 6.607166e-01,
                    -7.951295e-06, -8.722219e-06, -7.951295e-06, 4.145557e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict response
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var=TRUE, predict_response = TRUE)
  expected_mu <- c(2.142368, 2.089953, 2.441256)
  expected_var <- c(6.366765, 6.179095, 5.502758)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(pred$var-expected_var)),1E-6)
  # Evaluate approximate negative marginal log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
  expect_lt(abs(nll-195.03708036),1E-6)
})


test_that("Gamma regression ", {
  # Single level grouped random effects
  mu <- exp(Z1 %*% b_gr_1)
  shape <- 1
  y <- qgamma(sim_rand_unif(n=n, init_c=0.04532), scale = mu/shape, shape = shape)
  # Estimation 
  gp_model <- fitGPModel(group_data = group, likelihood = "gamma",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE))
  cov_pars <- c(0.5275497)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 51)
  # Prediction
  group_test <- c(1,3,3,9999)
  pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(0.2101641, -0.9204240, -0.9204240, 0.0000000)
  expected_cov <- c(0.08134093, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                    0.09851401, 0.09851401, 0.00000000, 0.00000000, 0.09851401,
                    0.09851401, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.52754971)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict response
  pred <- predict(gp_model, y=y, group_data_pred = group_test, predict_var=TRUE, predict_response = TRUE)
  expected_mu <- c(1.2850975, 0.4184629, 0.4184629, 1.3018351)
  expected_var <- c(1.9313699, 0.2113697, 0.2113697, 4.0497469)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(pred$var-expected_var)),1E-6)
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9),y=y)
  expect_lt(abs(nll-105.676137),1E-5)
  
  # Multiple random effects
  mu <- exp(Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3)
  y <- qgamma(sim_rand_unif(n=n, init_c=0.04532), scale = mu/shape, shape = shape)
  # Estimation 
  gp_model <- fitGPModel(group_data = cbind(group,group2), group_rand_coef_data = x,
                         ind_effect_group_rand_coef = 1, likelihood = "gamma",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE))
  cov_pars <- c(0.5122239, 1.2053430, 0.5502887)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 72)
  # Prediction
  group_data_pred = cbind(c(1,1,77),c(2,1,98))
  group_rand_coef_data_pred = c(0,0.1,0.3)
  pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                           cov_pars = c(0.9,0.8,1.2), predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(0.1121777, 0.1972216, 0.0000000)
  expected_cov <- c(0.2405621, 0.1157781, 0.0000000, 0.1157781,
                    0.2259258, 0.0000000, 0.0000000, 0.0000000, 1.8080000)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Gaussian process model
  mu <- exp(L %*% b_1)
  y <- qgamma(sim_rand_unif(n=n, init_c=0.435), scale = mu/shape, shape = shape)
  # Estimation 
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", likelihood = "gamma",
                         y = y, params = list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE))
  cov_pars <- c(1.0453253, 0.2638284)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 53)
  # Prediction
  coord_test <- cbind(c(0.1,0.11,0.7),c(0.9,0.91,0.55))
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE, predict_response = FALSE)
  expected_mu <- c(0.3346496, 0.2999703, 0.7757082)
  expected_cov <- c(0.4606703917, 0.4062301909, -0.0002110654, 0.4062301909, 0.4578850576, -0.0002109648, -0.0002110654, -0.0002109648, 0.3404264471)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Evaluate approximate negative marginal log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.9,0.2),y=y)
  expect_lt(abs(nll-154.4561783),1E-6)
})
