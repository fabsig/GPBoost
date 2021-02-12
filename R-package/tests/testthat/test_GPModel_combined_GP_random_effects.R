context("GPModel_combined_GP_grouped_random_effects")

# Function that simulates uniform random variables
sim_rand_unif <- function(n, init_c=0.1){
  mod_lcg <- 2^32 # modulus for linear congruential generator (random0 used)
  sim <- rep(NA, n)
  sim[1] <- floor(init_c * mod_lcg)
  for(i in 2:n) sim[i] <- (22695477 * sim[i-1] + 1) %% mod_lcg
  return(sim / mod_lcg)
}

# Create data
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
b_gr_1 <- qnorm(sim_rand_unif(n=m, init_c=0.56))
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

# Error term
xi <- qnorm(sim_rand_unif(n=n, init_c=0.1)) / 5
# Data for linear mixed effects model
X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # desing matrix / covariate data for fixed effect
beta <- c(2,2) # regression coefficents
# cluster_ids 
cluster_ids <- c(rep(1,0.4*n),rep(2,0.6*n))

# Sum up random effects
eps <- as.vector(L %*% b_1) + as.vector(Z1 %*% b_gr_1)
eps_svc <- as.vector(L %*% b_1 + Z_SVC[,1] * L %*% b_2 + Z_SVC[,2] * L %*% b_3) + 
  Z1 %*% b_gr_1 + Z2 %*% b_gr_2 + Z3 %*% b_gr_3


test_that("Combined Gaussian process and grouped random effects model ", {
  
  y <- eps + xi
  # Estimation using gradient descent and Nesterov acceleration
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group)
  fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                     lr_cov = 0.15, use_nesterov_acc = TRUE,
                                     acc_rate_cov = 0.8, delta_rel_conv=1E-6))
  cov_pars <- c(0.02924971, 0.09509924, 0.61463579, 0.30619763, 1.02189002, 0.25932007, 0.11327419, 0.04276286)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(dim(gp_model$get_cov_pars())[2], 4)
  expect_equal(dim(gp_model$get_cov_pars())[1], 2)
  expect_equal(gp_model$get_num_optim_iter(), 33)
  
  # Fisher scoring
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                         params = list(optimizer_cov = "fisher_scoring", std_dev = FALSE))
  cov_pars <- c(0.02262645, 0.61471473, 1.02446559, 0.11177327)
  cov_pars_est <- as.vector(gp_model$get_cov_pars())
  expect_lt(sum(abs(cov_pars_est-cov_pars)),1E-5)
  expect_equal(class(cov_pars_est), "numeric")
  expect_equal(length(cov_pars_est), 4)
  expect_equal(gp_model$get_num_optim_iter(), 8)
  
  # Prediction from fitted model
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group, y = y,
                         params = list(optimizer_cov = "fisher_scoring", std_dev = FALSE))
  coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
  group_test <- c(1,2,9999)
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                  group_data_pred = group_test, predict_cov_mat = TRUE)
  expected_mu <- c(0.3769074, 0.6779193, 0.1803276)
  expected_cov <- c(0.619329940, 0.007893047, 0.001356784, 0.007893047, 0.402082274,
                    -0.014950019, 0.001356784, -0.014950019, 1.046082243)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict variances
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                  group_data_pred = group_test, predict_var = TRUE)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
  
  # Prediction using given paraneters
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group)
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, group_data_pred = group_test,
                  cov_pars = c(0.02,1,1.2,0.9), predict_cov_mat = TRUE)
  expected_mu <- c(0.3995192, 0.6775987, 0.3710522)
  expected_cov <- c(0.1257410304, 0.0017195802, 0.0007660953, 0.0017195802,
                    0.0905110441, -0.0028869470, 0.0007660953, -0.0028869470, 1.1680614026)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,0.9,1.6,0.2),y=y)
  expect_lt(abs(nll-134.3491913),1E-6)
  
  # Do optimization using optim and e.g. Nelder-Mead
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group)
  opt <- optim(par=c(0.1,1.5,2,0.2), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
  expect_lt(sum(abs(opt$par-cov_pars)),1E-3)
  expect_lt(abs(opt$value-(132.4136164)),1E-5)
  expect_equal(as.integer(opt$counts[1]), 335)
})


test_that("Combined GP and grouped random effects model with linear regression term ", {
  
  y <- eps + X%*%beta + xi
  # Fit model
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                         y = y, X=X,
                         params = list(optimizer_cov = "fisher_scoring", optimizer_coef = "wls", std_dev = TRUE))
  cov_pars <- c(0.02258493, 0.09172947, 0.61704845, 0.30681934, 1.01910740, 0.25561489, 0.11202133, 0.04174140)
  coef <- c(2.06686646, 0.34643130, 1.92847425, 0.09983966)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 8)
  
  # Prediction 
  coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
  group_test <- c(1,2,9999)
  X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
  pred <- predict(gp_model, gp_coords_pred = coord_test, group_data_pred = group_test,
                  X_pred = X_test, predict_cov_mat = TRUE)
  expected_mu <- c(1.442617, 3.129006, 2.946252)
  expected_cov <- c(0.615200495, 0.007850776, 0.001344528, 0.007850776, 0.399458031,
                    -0.014866034, 0.001344528, -0.014866034, 1.045700453)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-5)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})


test_that("Combined GP and grouped random effects model with random coefficients ", {
  
  y <- eps_svc + xi
  # Fit model
  gp_model <- fitGPModel(y = y, gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                         group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                         params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                       acc_rate_cov = 0.5, maxit=10))
  expected_values <- c(0.4005820, 0.3111155, 0.4564903, 0.2693683, 1.3819153, 0.7034572,
                       1.0378165, 0.5916405, 1.3684672, 0.6861339, 0.1854759, 0.1430030,
                       0.5790945, 0.9748316, 0.2103132, 0.4453663, 0.2639379, 0.8772996, 0.2210313, 0.9282390)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 10)
  
  # Prediction
  gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential",
                      group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1)
  coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
  Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
  group_data_pred = cbind(c(1,1,7),c(2,1,3))
  group_rand_coef_data_pred = c(0,0.1,0.3)
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           gp_rand_coef_data_pred=Z_SVC_test,
                           group_data_pred=group_data_pred, group_rand_coef_data_pred=group_rand_coef_data_pred,
                           cov_pars = c(0.1,0.9,0.8,1.2,1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE)
  expected_mu <- c(0.8657964, 1.5419953, -2.5645509)
  expected_cov <- c(1.177484599, 0.073515374, 0.030303784, 0.073515374,
                    0.841043737, 0.004484463, 0.030303784, 0.004484463, 1.011570695)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Fisher scoring
  gp_model <- fitGPModel(y = y, gp_coords = coords, cov_function = "exponential", gp_rand_coef_data = Z_SVC,
                         group_data = cbind(group,group2), group_rand_coef_data = x, ind_effect_group_rand_coef = 1,
                         params = list(optimizer_cov = "fisher_scoring", std_dev = FALSE,
                                       use_nesterov_acc= FALSE, maxit=2))
  expected_values <- c(0.6093408, 0.8157278, 1.6016549, 1.2415390,1.7255119,
                       0.1400087, 1.1872654, 0.1469588, 0.4605333, 0.2635957)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 2)
  
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,0.9,0.8,1.2,1,0.1,0.8,0.15,1.1,0.08),y=y)
  expect_lt(abs(nll-182.3674191),1E-5)
})


test_that("Combined GP and grouped random effects model with cluster_id's not constant ", {
  
  y <- eps + xi
  # Fisher scoring
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                         y = y, cluster_ids = cluster_ids,
                         params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE))
  cov_pars <- c(0.005306836, 0.087915468, 0.615012714, 0.315022228,
                1.043024690, 0.228236254, 0.113716679, 0.039839629)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
  expect_equal(gp_model$get_num_optim_iter(), 8)
  
  # Prediction
  coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
  group_data_pred = c(1,1,9999)
  cluster_ids_pred = c(1,3,1)
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                      cluster_ids = cluster_ids)
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test, group_data_pred = group_data_pred,
                           cluster_ids_pred = cluster_ids_pred,
                           cov_pars = c(0.1,1.5,1,0.15), predict_cov_mat = TRUE)
  expected_mu <- c(0.1275193, 0.0000000, 0.5948827)
  expected_cov <- c(0.76147286, 0.00000000, -0.01260688, 0.00000000, 2.60000000,
                    0.00000000, -0.01260688, 0.00000000, 2.15607110)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})

test_that("Saving GPModel works ", {
  
  y <- eps + X%*%beta + xi
  # Fit model
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential", group_data = group,
                         y = y, X=X,
                         params = list(optimizer_cov = "fisher_scoring", optimizer_coef = "wls"))
  
  # Prediction 
  coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
  group_test <- c(1,2,9999)
  X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
  pred <- predict(gp_model, gp_coords_pred = coord_test, group_data_pred = group_test,
                  X_pred = X_test, predict_cov_mat = TRUE)
  # Save model to file
  filename <- tempfile(fileext = ".RData")
  saveGPModel(gp_model,filename = filename)
  # Delete model
  gp_model$finalize()
  expect_null(gp_model$.__enclos_env__$private$handle)
  rm(gp_model)
  # Load from file and make predictions again
  gp_model_loaded <- loadGPModel(filename = filename)
  pred_loaded <- predict(gp_model_loaded, gp_coords_pred = coord_test, group_data_pred = group_test,
                         X_pred = X_test, predict_cov_mat = TRUE)
  expect_equal(pred$mu, pred_loaded$mu)
  expect_equal(pred$cov, pred_loaded$cov)

})

