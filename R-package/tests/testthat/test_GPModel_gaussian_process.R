context("GPModel_grouped_random_effects")

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
d <- 2 # dimension
coords <- matrix(sim_rand_unif(n=n*d, init_c=0.1), ncol=d)
D <- as.matrix(dist(coords))
# Simulate GP
sigma2_1 <- 1^2 # marginal variance of GP
rho <- 0.1 # range parameter
Sigma <- sigma2_1*exp(-D/rho)+diag(1E-20,n)
C <- t(chol(Sigma))
b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.8))
eps <- as.vector(C %*% b_1)
# Random coefficients
Z_SVC <- matrix(sim_rand_unif(n=n*2, init_c=0.6), ncol=2) # covariate data for random coeffients
colnames(Z_SVC) <- c("var1","var2")
b_2 <- qnorm(sim_rand_unif(n=n, init_c=0.17))
b_3 <- qnorm(sim_rand_unif(n=n, init_c=0.42))
eps_svc <- as.vector(C %*% b_1 + Z_SVC[,1] * C %*% b_2 + Z_SVC[,2] * C %*% b_3)
# Error term
xi <- qnorm(sim_rand_unif(n=n, init_c=0.1)) / 5
# Data for linear mixed effects model
X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # desing matrix / covariate data for fixed effect
beta <- c(2,2) # regression coefficents
# cluster_ids 
cluster_ids <- c(rep(1,0.4*n),rep(2,0.6*n))
# GP with multiple observations at the same locations
coords_multiple <- matrix(sim_rand_unif(n=n*d/4, init_c=0.1), ncol=d)
coords_multiple <- rbind(coords_multiple,coords_multiple,coords_multiple,coords_multiple)
D_multiple <- as.matrix(dist(coords_multiple))
Sigma_multiple <- sigma2_1*exp(-D_multiple/rho)+diag(1E-10,n)
C_multiple <- t(chol(Sigma_multiple))
b_multiple <- qnorm(sim_rand_unif(n=n, init_c=0.8))
eps_multiple <- as.vector(C_multiple %*% b_multiple)


test_that("Gaussian process model ", {
  
  y <- eps + xi
  # Estimation using gradient descent and Nesterov acceleration
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
  fit(gp_model, y = y, std_dev = TRUE, params = list(optimizer_cov = "gradient_descent",
                                                     lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                     acc_rate_cov = 0.5, delta_rel_conv=1E-6,
                                                     convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.03276544, 0.07715339, 1.07617623, 0.25177590, 0.11352557, 0.03770062)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(dim(gp_model$get_cov_pars())[2], 3)
  expect_equal(dim(gp_model$get_cov_pars())[1], 2)
  expect_equal(gp_model$get_num_optim_iter(), 374)
  
  # Gradient descent without Nesterov acceleration
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         y = y, std_dev = TRUE, params = list(optimizer_cov = "gradient_descent",
                                                     lr_cov = 0.1, use_nesterov_acc = FALSE,
                                                     delta_rel_conv=1E-6, convergence_criterion = "relative_change_in_parameters"))
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),5E-6)
  expect_equal(gp_model$get_num_optim_iter(), 723)
  
  # Using a too large learning rate
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         y = y, std_dev = TRUE, params = list(optimizer_cov = "gradient_descent",maxit=1500,
                                                              lr_cov = 1, use_nesterov_acc = FALSE,
                                                              delta_rel_conv=1E-6, convergence_criterion = "relative_change_in_parameters"))
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),2E-5)
  expect_equal(gp_model$get_num_optim_iter(), 1136)
  
  # Different terminations criterion and a too large learning rate
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         y = y, std_dev = TRUE, params = list(optimizer_cov = "gradient_descent",
                                                              lr_cov = 1, use_nesterov_acc = FALSE,
                                                              delta_rel_conv=1E-9))
  cov_pars_other_crit <- c(0.03240165, 0.07700151, 1.07657315, 0.25170095, 0.11345181, 0.03764893)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other_crit)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 550)
  
  # Fisher scoring
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         y = y, std_dev = FALSE,
                         params = list(optimizer_cov = "fisher_scoring",
                                       delta_rel_conv=1E-6, use_nesterov_acc = FALSE,
                                       convergence_criterion = "relative_change_in_parameters"))
  cov_pars_est <- as.vector(gp_model$get_cov_pars())
  expect_lt(sum(abs(cov_pars_est-cov_pars[c(1,3,5)])),1E-5)
  expect_equal(class(cov_pars_est), "numeric")
  expect_equal(length(cov_pars_est), 3)
  expect_equal(gp_model$get_num_optim_iter(), 16)
  
  # Different termination criterion
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring",
                                       delta_rel_conv=1E-6, use_nesterov_acc = FALSE,
                                       convergence_criterion = "relative_change_in_log_likelihood"))
  cov_pars_other_crit <- c(0.03300593, 0.07725225, 1.07584118, 0.25180563, 0.11357012, 0.03773325)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other_crit)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 8)
  
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         y = y, std_dev = TRUE, params = list(optimizer_cov = "gradient_descent",
                                                              lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                              acc_rate_cov = 0.2, delta_rel_conv=1E-6))
  cov_pars_other_crit <- c(0.03956853, 0.08006581, 1.07081043, 0.25358121, 0.11485798, 0.03865007)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other_crit)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 82)
  
  # Prediction from fitted model
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         y = y, std_dev = FALSE,
                         params = list(optimizer_cov = "fisher_scoring",
                                       delta_rel_conv=1E-6, use_nesterov_acc = FALSE,
                                       convergence_criterion = "relative_change_in_parameters"))
  coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
  expected_mu <- c(0.06960478, 1.61299381, 0.44053480)
  expected_cov <- c(6.218737e-01, 2.024102e-05, 2.278875e-07, 2.024102e-05,
                    3.535390e-01, 8.479210e-07, 2.278875e-07, 8.479210e-07, 4.202154e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Prediction using given paraneters
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                  cov_pars = c(0.02,1.2,0.9), predict_cov_mat = TRUE)
  expected_mu <- c(0.08704577, 1.63875604, 0.48513581)
  expected_cov <- c(1.189093e-01, 1.171632e-05, -4.172444e-07, 1.171632e-05,
                    7.427727e-02, 1.492859e-06, -4.172444e-07, 1.492859e-06, 8.107455e-02)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1.6,0.2),y=y)
  expect_lt(abs(nll-124.2549533),1E-6)
  
  # Do optimization using optim and e.g. Nelder-Mead
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
  opt <- optim(par=c(0.1,2,0.2), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
  expect_lt(sum(abs(opt$par-cov_pars[c(1,3,5)])),1E-3)
  expect_lt(abs(opt$value-(122.7752694)),1E-5)
  expect_equal(as.integer(opt$counts[1]), 198)
})


test_that("Gaussian process model with linear regression term ", {
  
  y <- eps + X%*%beta + xi
  # Fit model
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         y = y, X=X, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring", optimizer_coef = "wls",
                                       delta_rel_conv=1E-6, use_nesterov_acc = FALSE,
                                       convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.008461342, 0.069973492, 1.001562822, 0.214358560, 0.094656409, 0.029400407)
  coef <- c(2.30780026, 0.21365770, 1.89951426, 0.09484768)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_lt(sum(abs( as.vector(gp_model$get_coef())-coef)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 11)
  
  # Prediction 
  coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
  X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
  pred <- predict(gp_model, gp_coords_pred = coord_test,
                  X_pred = X_test, predict_cov_mat = TRUE)
  expected_mu <- c(1.196952, 4.063324, 3.156427)
  expected_cov <- c(6.305383e-01, 1.358861e-05, 8.317903e-08, 1.358861e-05,
                    3.469270e-01, 2.686334e-07, 8.317903e-08, 2.686334e-07, 4.255400e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})


# Ignore [GPBoost] [Warning]
test_that("Gaussian process and two random coefficients ", {
  
  y <- eps_svc + xi
  # Fit model
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         gp_rand_coef_data = Z_SVC, y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                       acc_rate_cov = 0.5, maxit=10))
  expected_values <- c(0.24968994, 0.22559907, 0.83542391, 0.41810207, 0.15034219,
                       0.10037844, 1.65329625, 0.84506015, 0.08796681, 0.06828663,
                       0.23702546, 0.61869306, 0.08649348, 0.33490111)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 10)
  
  # Prediction
  gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential")
  coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
  Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           gp_rand_coef_data_pred=Z_SVC_test,
                           cov_pars = c(0.1,1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE)
  expected_mu <- c(-0.1669209, 1.6166381, 0.2861320)
  expected_cov <- c(9.643323e-01, 3.536846e-04, -1.783557e-04, 3.536846e-04,
                    5.155009e-01, 4.554321e-07, -1.783557e-04, 4.554321e-07, 7.701614e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Fisher scoring
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         gp_rand_coef_data = Z_SVC, y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring",
                                       use_nesterov_acc= FALSE, maxit=5))
  expected_values <- c(0.000242813, 0.197623969, 1.120356660, 0.442501100,
                       0.141084495, 0.070778399, 1.670556231, 0.798785653, 0.055598038,
                       0.047379252, 0.430573036, 0.605871724, 0.038976112, 0.116595965)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 5)
  
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,0.1,0.8,0.15,1.1,0.08),y=y)
  expect_lt(abs(nll-149.4422184),1E-5)
})


test_that("Gaussian process model with cluster_id's not constant ", {
  
  y <- eps + xi
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         y = y, std_dev = TRUE, cluster_ids = cluster_ids,
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                       acc_rate_cov = 0.5, delta_rel_conv=1E-6,
                                       convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.05414149, 0.08722111, 1.05789166, 0.22886740, 0.12702368, 0.04076914)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 239)
  
  # Fisher scoring
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         y = y, std_dev = TRUE, cluster_ids = cluster_ids,
                         params = list(optimizer_cov = "fisher_scoring",
                                       use_nesterov_acc = FALSE, delta_rel_conv=1E-6,
                                       convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.05414149, 0.08722111, 1.05789166, 0.22886740, 0.12702368, 0.04076914)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
  expect_equal(gp_model$get_num_optim_iter(), 20)
  
  # Prediction
  coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
  cluster_ids_pred = c(1,3,1)
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                      cluster_ids = cluster_ids)
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           cluster_ids_pred = cluster_ids_pred,
                           cov_pars = c(0.1,1,0.15), predict_cov_mat = TRUE)
  expected_mu <- c(-0.01437506, 0.00000000, 0.93112902)
  expected_cov <- c(0.743055189, 0.000000000, -0.000140644, 0.000000000,
                    1.100000000, 0.000000000, -0.000140644, 0.000000000, 0.565243468)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})


test_that("Gaussian process model with multiple observations at the same location ", {
  
  y <- eps_multiple + xi
  gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                       delta_rel_conv=1E-6, maxit = 500))
  cov_pars <- c(0.037145465, 0.006065652, 1.151982610, 0.434770575, 0.191648634, 0.102375515)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 12)
  
  # Fisher scoring
  gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring",
                                       use_nesterov_acc = FALSE, delta_rel_conv=1E-6,
                                       convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.037136462, 0.006064181, 1.153630335, 0.435788570, 0.192080613, 0.102631006)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
  expect_equal(gp_model$get_num_optim_iter(), 14)
  
  # Prediction
  coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
  cluster_ids_pred = c(1,3,1)
  gp_model <- GPModel(gp_coords = coords_multiple, cov_function = "exponential")
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           cov_pars = c(0.1,1,0.15), predict_cov_mat = TRUE)
  expected_mu <- c(-0.1460550, 1.0042814, 0.7840301)
  expected_cov <- c(0.6739502109, 0.0008824337, -0.0003815281, 0.0008824337,
                    0.6060039551, -0.0004157361, -0.0003815281, -0.0004157361, 0.7851787946)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})


# Ignore [GPBoost] [Warning]
test_that("Vecchia approximation for Gaussian process model ", {
  
  y <- eps + xi
  # Full model 
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                      vecchia_approx=TRUE, num_neighbors=n-1)
  fit(gp_model, y = y, std_dev = TRUE, params = list(optimizer_cov = "gradient_descent",
                                                     lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                     acc_rate_cov = 0.5, delta_rel_conv=1E-6,
                                                     convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.03276544, 0.07715339, 1.07617623, 0.25177590, 0.11352557, 0.03770062)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(dim(gp_model$get_cov_pars())[2], 3)
  expect_equal(dim(gp_model$get_cov_pars())[1], 2)
  expect_equal(gp_model$get_num_optim_iter(), 374)
  
  # Vechia approximation with 30 neighbors
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                      vecchia_approx=TRUE, num_neighbors=30)
  fit(gp_model, y = y, std_dev = TRUE, params = list(optimizer_cov = "gradient_descent",
                                                     lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                     acc_rate_cov = 0.5, delta_rel_conv=1E-6,
                                                     maxit=100, convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.02464980, 0.07379861, 1.06141118, 0.23237579, 0.11459636, 0.03567217)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 100)
  
  # Fisher scoring & random ordering
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         vecchia_approx=TRUE, num_neighbors=30, vecchia_ordering="random",
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring",
                                       use_nesterov_acc = FALSE,
                                       delta_rel_conv=1E-6, maxit=100,
                                       convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.01692036, 0.06802553, 1.15762655, 0.26474644, 0.12429916, 0.03962540)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 17)
  
  # Prediction from fitted model
  coord_test <- cbind(c(0.1,0.10001,0.7),c(0.9,0.90001,0.55))
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                  vecchia_pred_type = "order_obs_first_cond_obs_only")
  expected_mu <- c(0.07848340, 0.07848138, 0.95991066)
  expected_cov <- c(0.6017198, 0.0000000, 0.0000000, 0.0000000,
                    0.6017320, 0.0000000, 0.0000000, 0.0000000, 0.5100577)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Fisher scoring & default ordering
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         vecchia_approx=TRUE, num_neighbors=30, vecchia_ordering="none",
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring",
                                       use_nesterov_acc = FALSE,
                                       delta_rel_conv=1E-6, maxit=100,
                                       convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.01829104, 0.07043221, 1.06577048, 0.23037437, 0.11335918, 0.03484927)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 20)
  
  # Prediction using given paraneters
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                  cov_pars = c(0.02,1.2,0.9), predict_cov_mat = TRUE,
                  vecchia_pred_type = "order_obs_first_cond_obs_only")
  expected_mu <- c(0.08665472, 0.08664854, 0.98675487)
  expected_cov <- c(0.1189100, 0.00000000, 0.00000000, 0.00000000,
                    0.1189129, 0.00000000, 0.00000000, 0.00000000, 0.1159135)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Prediction with different ordering
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                  cov_pars = c(0.02,1.2,0.9), predict_cov_mat = TRUE,
                  vecchia_pred_type = "order_obs_first_cond_all")
  expected_mu <- c(0.08665472, 0.08661259, 0.98675487)
  expected_cov <- c(0.11891004, 0.09889262, 0.00000000, 0.09889262,
                    0.11891291, 0.00000000, 0.00000000, 0.00000000, 0.11591347)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Prediction with different ordering
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                  cov_pars = c(0.02,1.2,0.9), predict_cov_mat = TRUE,
                  vecchia_pred_type = "order_pred_first")
  expected_mu <- c(0.1120768, 0.1119616, 0.5522175)
  expected_cov <- c(1.187731e-01, 9.875550e-02, 1.935232e-07, 9.875550e-02,
                    1.187756e-01, -2.161678e-07, 1.935232e-07, -2.161678e-07, 6.709577e-02)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Prediction with different ordering
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                  cov_pars = c(0.02,1.2,0.9), predict_cov_mat = TRUE,
                  vecchia_pred_type = "latent_order_obs_first_cond_obs_only")
  expected_mu <- c(0.08513257, 0.08512608, 0.90542680)
  expected_cov <- c(1.189086e-01, 7.322771e-03, -7.263024e-07, 7.322771e-03,
                    1.189114e-01, -7.261547e-07, -7.263024e-07, -7.261547e-07, 1.149206e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Prediction with different ordering
  pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                  cov_pars = c(0.02,1.2,0.9), predict_cov_mat = TRUE,
                  vecchia_pred_type = "latent_order_obs_first_cond_all")
  expected_mu <- c(0.08513257, 0.08512602, 0.90542680)
  expected_cov <- c(1.189086e-01, 9.889112e-02, -7.263386e-07, 9.889112e-02,
                    1.189114e-01, -7.261806e-07, -7.263385e-07, -7.261805e-07, 1.149206e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1.6,0.2),y=y)
  expect_lt(abs(nll-126.6644435),1E-6)
})

test_that("Vecchia approximation for Gaussian process model with linear regression term ", {
  
  y <- eps + X%*%beta + xi
  # Fit model
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         vecchia_approx=TRUE, num_neighbors=30,
                         y = y, X=X, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring", optimizer_coef = "wls",
                                       delta_rel_conv=1E-6, use_nesterov_acc = FALSE,
                                       convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(2.022073e-61, 6.645280e-02, 9.611996e-01, 1.903751e-01, 9.059417e-02, 2.660703e-02)
  coef <- c(2.35074728, 0.18885871, 1.91556300, 0.09439146)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 20)
  
  # Prediction 
  coord_test <- cbind(c(0.1,0.10001,0.7),c(0.9,0.90001,0.55))
  X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
  pred <- predict(gp_model, gp_coords_pred = coord_test,
                  X_pred = X_test, predict_cov_mat = TRUE,
                  vecchia_pred_type = "order_obs_first_cond_all")
  expected_mu <- c(1.191993, 2.532897, 3.619877)
  expected_cov <- c(0.6139641, 0.6138193, 0.0000000, 0.6138193,
                    0.6139746, 0.0000000, 0.0000000, 0.0000000, 0.5724550)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-5)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})


test_that("Gaussian process model with cluster_id's not constant ", {
  
  y <- eps + xi
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         vecchia_approx=TRUE, num_neighbors=30,
                         y = y, std_dev = TRUE, cluster_ids = cluster_ids,
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.05, use_nesterov_acc = TRUE,
                                       acc_rate_cov = 0.5, delta_rel_conv=1E-6,
                                       convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.03359375, 0.07751222, 1.07019293,
                0.22080542, 0.12201912, 0.03761778)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 656)
  
  # Fisher scoring
  gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                         vecchia_approx=TRUE, num_neighbors=30,
                         y = y, std_dev = TRUE, cluster_ids = cluster_ids,
                         params = list(optimizer_cov = "fisher_scoring",
                                       use_nesterov_acc = FALSE, delta_rel_conv=1E-6,
                                       convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.03359131, 0.07751098, 1.07019567, 0.22080490, 0.12201865, 0.03761748)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
  expect_equal(gp_model$get_num_optim_iter(), 15)
  
  # Prediction
  coord_test <- cbind(c(0.1,0.2,0.1001),c(0.9,0.4,0.9001))
  cluster_ids_pred = c(1,3,1)
  gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                      vecchia_approx=TRUE, num_neighbors=30, cluster_ids = cluster_ids)
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           cluster_ids_pred = cluster_ids_pred,
                           vecchia_pred_type = "order_obs_first_cond_all",
                           cov_pars = c(0.1,1,0.15), predict_cov_mat = TRUE)
  expected_mu <- c(-0.01438585, 0.00000000, -0.01500132)
  expected_cov <- c(0.7430552, 0.0000000, 0.6423148, 0.0000000,
                    1.1000000, 0.0000000, 0.6423148, 0.0000000, 0.7434589)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})


test_that("Vecchia approximation for Gaussian process model with multiple observations at the same location ", {
  
  y <- eps_multiple + xi
  gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                         vecchia_approx=TRUE, num_neighbors=30,
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                       delta_rel_conv=1E-6, maxit = 500))
  cov_pars <- c(0.038572101, 0.006430877, 1.082908053, 0.376533991, 0.169594641, 0.084952224)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 10)
  
  # Fisher scoring
  gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                         vecchia_approx=TRUE, num_neighbors=30,
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring",
                                       use_nesterov_acc = FALSE, delta_rel_conv=1E-6,
                                       convergence_criterion = "relative_change_in_parameters"))
  cov_pars <- c(0.038558364, 0.006428588, 1.084369959, 0.377374632, 0.169970239, 0.085155496)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
  expect_equal(gp_model$get_num_optim_iter(), 17)
  
  # Prediction
  coord_test <- cbind(c(0.1,0.2,0.1001),c(0.9,0.4,0.9001))
  cluster_ids_pred = c(1,3,1)
  gp_model <- GPModel(gp_coords = coords_multiple, cov_function = "exponential",
                      vecchia_approx=TRUE, num_neighbors=30)
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           vecchia_pred_type = "order_obs_first_cond_all",
                           cov_pars = c(0.1,1,0.15), predict_cov_mat = TRUE)
  expected_mu <- c(-0.1517240, 1.1449797, -0.1518975)
  expected_cov <- c(0.6739791631, 0.0007737063, 0.5732697669, 0.0007737063,
                    0.6237424619, 0.0006580952, 0.5732697669, 0.0006580952, 0.6744448690)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})


# Crashes currently. TODO: find bug
# # Ignore [GPBoost] [Warning]
# test_that("Vecchia approximation for Gaussian process and two random coefficients ", {
#   
#   y <- eps_svc + xi
#   # Fit model
#   gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#                          vecchia_approx=TRUE, num_neighbors=30,
#                          gp_rand_coef_data = Z_SVC, y = y, std_dev = TRUE,
#                          params = list(optimizer_cov = "gradient_descent",
#                                        lr_cov = 0.1, use_nesterov_acc = TRUE,
#                                        acc_rate_cov = 0.5, maxit=10))
#   expected_values <- c(0.24579587, 0.21600711, 0.80230753, 0.40094567,
#                        0.15003184, 0.10035712, 1.52691187, 0.80317611, 0.09023968,
#                        0.07149687, 0.23869704, 0.59534450, 0.08751736, 0.32229992)
#   expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
#   
#   # Prediction
#   gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential")
#   coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
#   Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
#   pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
#                            gp_rand_coef_data_pred=Z_SVC_test,
#                            cov_pars = c(0.1,1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE)
#   expected_mu <- c(-0.1669209, 1.6166381, 0.2861320)
#   expected_cov <- c(9.643323e-01, 3.536846e-04, -1.783557e-04, 3.536846e-04,
#                     5.155009e-01, 4.554321e-07, -1.783557e-04, 4.554321e-07, 7.701614e-01)
#   expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
#   expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
#   
#   # Fisher scoring
#   gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
#                          vecchia_approx=TRUE, num_neighbors=30,
#                          gp_rand_coef_data = Z_SVC, y = y, std_dev = TRUE,
#                          params = list(optimizer_cov = "fisher_scoring",
#                                        use_nesterov_acc= FALSE, maxit=5))
#   expected_values <- c(0.000242813, 0.197623969, 1.120356660, 0.442501100,
#                        0.141084495, 0.070778399, 1.670556231, 0.798785653, 0.055598038,
#                        0.047379252, 0.430573036, 0.605871724, 0.038976112, 0.116595965)
#   expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
#   expect_equal(gp_model$get_num_optim_iter(), 5)
#   
#   # Evaluate negative log-likelihood
#   nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,0.1,0.8,0.15,1.1,0.08),y=y)
#   expect_lt(abs(nll-149.4422184),1E-5)
# })