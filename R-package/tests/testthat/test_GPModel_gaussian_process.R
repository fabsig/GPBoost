context("GPModel_gaussian_process")

DEFAULT_OPTIM_PARAMS <- list(optimizer_cov = "gradient_descent",
                             lr_cov = 0.1, use_nesterov_acc = TRUE,
                             acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                             optimizer_coef = "gradient_descent", lr_coef = 0.1,
                             convergence_criterion = "relative_change_in_log_likelihood")
DEFAULT_OPTIM_PARAMS_STD <- c(DEFAULT_OPTIM_PARAMS, list(std_dev = TRUE))
TOLERANCE_LOOSE <- 1E-2
TOLERANCE_STRICT <- 1E-6

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
X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # design matrix / covariate data for fixed effect
beta <- c(2,2) # regression coefficients
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

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  test_that("Gaussian process model ", {
    
    y <- eps + xi
    # Estimation using gradient descent and Nesterov acceleration
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    capture.output( fit(gp_model, y = y, params = DEFAULT_OPTIM_PARAMS_STD), 
                    file='NUL')
    cov_pars <- c(0.03784221, 0.07943467, 1.07390943, 0.25351519, 0.11451432, 0.03840236)
    num_it <- 59
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(dim(gp_model$get_cov_pars())[2], 3)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Gradient descent without Nesterov acceleration
    params <- DEFAULT_OPTIM_PARAMS_STD
    params$use_nesterov_acc <- FALSE
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params) , file='NUL')
    cov_pars_other <- c(0.04040441, 0.08036674, 1.06926607, 0.25360131, 0.11502362, 0.03877014)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)),5E-6)
    expect_equal(gp_model$get_num_optim_iter(), 97)
    # Using a too large learning rate
    params <- DEFAULT_OPTIM_PARAMS_STD
    params$lr_cov <- 1
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params) , file='NUL')
    cov_pars_other <- c(0.04487369, 0.08285696, 1.07537253, 0.25676579, 0.11566763, 0.03928107)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 27)
    # Different terminations criterion
    params <- DEFAULT_OPTIM_PARAMS_STD
    params$convergence_criterion = "relative_change_in_parameters"
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params), file='NUL')
    cov_pars_other_crit <- c(0.03276547, 0.07715343, 1.07617676, 0.25177603, 0.11352557, 0.03770062)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other_crit)), TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    ll <- gp_model$neg_log_likelihood(y=y,cov_pars=gp_model$get_cov_pars()[1,])
    expect_lt(abs(ll-122.7752664),TOLERANCE_STRICT)
    # Fisher scoring
    params <- DEFAULT_OPTIM_PARAMS_STD
    params$optimizer_cov = "fisher_scoring"
    params$lr_cov <- 1
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = params), file='NUL')
    cov_pars_fisher <- c(0.03300593, 0.07725225, 1.07584118, 0.25180563, 0.11357012, 0.03773325)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_fisher)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    # Nelder-mead
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = list(optimizer_cov = "nelder_mead")), file='NUL')
    cov_pars_est <- as.vector(gp_model$get_cov_pars())
    expect_lt(sum(abs(cov_pars_est-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 40)
    # BFGS
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = list(optimizer_cov = "bfgs")), file='NUL')
    cov_pars_est <- as.vector(gp_model$get_cov_pars())
    expect_lt(sum(abs(cov_pars_est-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 16)
    # Adam
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = list(optimizer_cov = "adam")), file='NUL')
    cov_pars_est <- as.vector(gp_model$get_cov_pars())
    expect_lt(sum(abs(cov_pars_est-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    expect_equal(gp_model$get_num_optim_iter(), 498)
    
    # Prediction from fitted model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y,
                                           params = list(optimizer_cov = "fisher_scoring",
                                                         delta_rel_conv = 1E-6, use_nesterov_acc = FALSE,
                                                         convergence_criterion = "relative_change_in_parameters")), file='NUL')
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    expect_error(predict(gp_model))# coord data not provided
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expected_mu <- c(0.06960478, 1.61299381, 0.44053480)
    expected_cov <- c(6.218737e-01, 2.024102e-05, 2.278875e-07, 2.024102e-05,
                      3.535390e-01, 8.479210e-07, 2.278875e-07, 8.479210e-07, 4.202154e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    # Prediction of variances only
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
    
    # Predict training data random effects
    training_data_random_effects <- predict_training_data_random_effects(gp_model)
    pred_random_effects <- predict(gp_model, gp_coords_pred = coords)
    expect_lt(sum(abs(training_data_random_effects - pred_random_effects$mu)),1E-6)
    
    # Prediction using given parameters
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    cov_pars_pred = c(0.02,1.2,0.9)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE)
    expected_mu <- c(0.08704577, 1.63875604, 0.48513581)
    expected_cov <- c(1.189093e-01, 1.171632e-05, -4.172444e-07, 1.171632e-05,
                      7.427727e-02, 1.492859e-06, -4.172444e-07, 1.492859e-06, 8.107455e-02)
    cov_no_nugget <- expected_cov
    cov_no_nugget[c(1,5,9)] <- expected_cov[c(1,5,9)] - cov_pars_pred[1]
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    # Prediction of variances only
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, 
                    cov_pars = cov_pars_pred, predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
    # Predict latent process
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-cov_no_nugget)),1E-6)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1.6,0.2),y=y)
    expect_lt(abs(nll-124.2549533),1E-6)
    
    # Do optimization using optim and e.g. Nelder-Mead
    gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
    opt <- optim(par=c(0.1,2,0.2), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
    expect_lt(sum(abs(opt$par-cov_pars[c(1,3,5)])),TOLERANCE_LOOSE)
    expect_lt(abs(opt$value-(122.7752694)),1E-5)
    expect_equal(as.integer(opt$counts[1]), 198)
    
    # Other covariance functions
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 0.5,
                                           y = y, params = DEFAULT_OPTIM_PARAMS_STD) , file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 1.5,
                                           y = y, params = DEFAULT_OPTIM_PARAMS_STD) , file='NUL')
    cov_pars_other <- c(0.22912036, 0.08482545, 0.87876886, 0.24053588, 0.10725076, 0.01542466)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 16)
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern",
                                           cov_fct_shape = 2.5,
                                           y = y, params = DEFAULT_OPTIM_PARAMS_STD) , file='NUL')
    cov_pars_other <- c(.27467781, 0.08345405, 0.82842693, 0.23533957, 0.10567129, 0.01069569)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_other)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 17)
  })
  
  test_that("Gaussian process model with linear regression term ", {
    
    y <- eps + X%*%beta + xi
    # Fit model
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X=X,
                           params = list(optimizer_cov = "fisher_scoring", optimizer_coef = "wls",
                                         delta_rel_conv = 1E-6, use_nesterov_acc = FALSE, std_dev = TRUE,
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
    
    # Gradient descent
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y=y, X=X,
                           params = list(optimizer_cov = "gradient_descent", optimizer_coef = "gradient_descent",
                                         delta_rel_conv = 1E-6, use_nesterov_acc = TRUE, std_dev = FALSE, lr_coef=1))
    cov_pars <- c(0.01624576, 0.99717015, 0.09616822)
    coef <- c(2.305484, 1.899207)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 99)
    
    # Nelder-Mead
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X=X, params = list(optimizer_cov = "nelder_mead",
                                                     optimizer_coef = "nelder_mead",
                                                     maxit=1000, delta_rel_conv = 1e-12))
    cov_pars <- c(0.008459373, 1.001564796, 0.094655964)
    coef <- c(2.307798, 1.899516)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 429)
    # BFGS
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X=X, params = list(optimizer_cov = "bfgs",
                                                     maxit=1000))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-2)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-2)
    expect_equal(gp_model$get_num_optim_iter(), 25)
    # Adam
    gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                           y = y, X=X, params = list(optimizer_cov = "adam",
                                                     maxit=5000))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-2)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-2)
    expect_equal(gp_model$get_num_optim_iter(), 1986)
    
  })
  
}

test_that("Gaussian process and two random coefficients ", {
  
  y <- eps_svc + xi
  # Fit model
  capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                         gp_rand_coef_data = Z_SVC, y = y,
                                         params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                       acc_rate_cov = 0.5, maxit=10)), file='NUL')
  expected_values <- c(0.25740068, 0.22608704, 0.83503539, 0.41896403, 0.15039055,
                       0.10090869, 1.61010233, 0.84207763, 0.09015444, 0.07106099, 
                       0.25064640, 0.62279880, 0.08720822, 0.32047865)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 10)
  
  # Prediction
  gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential")
  coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
  Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
  expect_error(gp_model$predict(y = y, gp_coords_pred = coord_test,
                                cov_pars = c(0.1,1,0.1,0.8,0.15,1.1,0.08)))# random slope data not provided
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           gp_rand_coef_data_pred=Z_SVC_test,
                           cov_pars = c(0.1,1,0.1,0.8,0.15,1.1,0.08), predict_cov_mat = TRUE)
  expected_mu <- c(-0.1669209, 1.6166381, 0.2861320)
  expected_cov <- c(9.643323e-01, 3.536846e-04, -1.783557e-04, 3.536846e-04,
                    5.155009e-01, 4.554321e-07, -1.783557e-04, 4.554321e-07, 7.701614e-01)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict variances
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           gp_rand_coef_data_pred=Z_SVC_test,
                           cov_pars = c(0.1,1,0.1,0.8,0.15,1.1,0.08), predict_var = TRUE)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
  
  # Fisher scoring
  capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                         gp_rand_coef_data = Z_SVC, y = y,
                                         params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                       use_nesterov_acc= FALSE, maxit=5)), file='NUL')
  expected_values <- c(0.000242813, 0.176573955, 1.008181385, 0.397341267, 0.141084495, 
                       0.070671768, 1.432715033, 0.708039197, 0.055598038, 0.048988825, 
                       0.430573036, 0.550644708, 0.038976112, 0.106110593)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 5)
  
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,0.1,0.8,0.15,1.1,0.08),y=y)
  expect_lt(abs(nll-149.4422184),1E-5)
})

test_that("Gaussian process model with cluster_id's not constant ", {
  
  y <- eps + xi
  capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                         y = y, cluster_ids = cluster_ids,
                                         params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                       acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                                                       convergence_criterion = "relative_change_in_parameters")), file='NUL')
  cov_pars <- c(0.05414149, 0.08722111, 1.05789166, 0.22886740, 0.12702368, 0.04076914)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 247)
  
  # Fisher scoring
  capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                         y = y, cluster_ids = cluster_ids,
                                         params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                       use_nesterov_acc = FALSE, delta_rel_conv = 1E-6,
                                                       convergence_criterion = "relative_change_in_parameters")), file='NUL')
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
  # Predict variances
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           cluster_ids_pred = cluster_ids_pred,
                           cov_pars = c(0.1,1,0.15), predict_var = TRUE)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
})

test_that("Gaussian process model with multiple observations at the same location ", {
  
  y <- eps_multiple + xi
  capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", y = y,
                                         params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                                       delta_rel_conv = 1E-6, maxit = 500)), file='NUL')
  cov_pars <- c(0.037145465, 0.006065652, 1.151982610, 0.434770575, 0.191648634, 0.102375515)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 12)
  
  # Fisher scoring
  capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential", y = y,
                                         params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                       use_nesterov_acc = FALSE, delta_rel_conv = 1E-6,
                                                       convergence_criterion = "relative_change_in_parameters")), file='NUL')
  cov_pars <- c(0.037136462, 0.006064181, 1.153630335, 0.435788570, 0.192080613, 0.102631006)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
  expect_equal(gp_model$get_num_optim_iter(), 14)
  
  # Predict training data random effects
  training_data_random_effects <- predict_training_data_random_effects(gp_model)
  pred_random_effects <- predict(gp_model, gp_coords_pred = coords_multiple)
  expect_lt(sum(abs(training_data_random_effects - pred_random_effects$mu)),1E-6)
  
  # Prediction
  coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
  gp_model <- GPModel(gp_coords = coords_multiple, cov_function = "exponential")
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           cov_pars = c(0.1,1,0.15), predict_cov_mat = TRUE)
  expected_mu <- c(-0.1460550, 1.0042814, 0.7840301)
  expected_cov <- c(0.6739502109, 0.0008824337, -0.0003815281, 0.0008824337,
                    0.6060039551, -0.0004157361, -0.0003815281, -0.0004157361, 0.7851787946)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  # Predict variances
  pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                           cov_pars = c(0.1,1,0.15), predict_var = TRUE)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
})

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  test_that("Vecchia approximation for Gaussian process model ", {
    
    y <- eps + xi
    params_vecchia <- list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                           lr_cov = 0.1, use_nesterov_acc = TRUE,
                           acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                           convergence_criterion = "relative_change_in_parameters")
    # Maximal number of neighbors
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = n-1,
                                        vecchia_ordering = "none"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_vecchia), file='NUL')
    cov_pars <- c(0.03276544, 0.07715339, 1.07617623, 0.25177590, 0.11352557, 0.03770062)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(dim(gp_model$get_cov_pars())[2], 3)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    
    # Same thing without Vecchia approximation
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential"), file='NUL')
    capture.output( fit(gp_model, y = y, params = params_vecchia), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(dim(gp_model$get_cov_pars())[2], 3)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    
    # Random ordering
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n-1, vecchia_ordering="random", y = y,
                                           params = params_vecchia), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(dim(gp_model$get_cov_pars())[2], 3)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), 382)
    
    # Prediction using given parameters
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = n-1,
                                        vecchia_ordering = "none"), file='NUL')
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    cov_pars = c(0.02,1.2,0.9)
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars, predict_cov_mat = TRUE, num_neighbors_pred=n+2,
                    vecchia_pred_type = "order_obs_first_cond_all", predict_response = TRUE)
    expected_mu <- c(0.08704577, 1.63875604, 0.48513581)
    expected_cov <- c(1.189093e-01, 1.171632e-05, -4.172444e-07, 1.171632e-05,
                      7.427727e-02, 1.492859e-06, -4.172444e-07, 1.492859e-06, 8.107455e-02)
    exp_cov_no_nugget <- expected_cov
    exp_cov_no_nugget[c(1,5,9)] <- expected_cov[c(1,5,9)] - cov_pars[1]
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars, predict_cov_mat = TRUE,
                    vecchia_pred_type = "order_obs_first_cond_all", predict_response = FALSE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-exp_cov_no_nugget)),1E-6)
    # Prediction of variances only
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test, num_neighbors_pred=n+2, 
                    cov_pars = cov_pars, predict_var = TRUE, predict_response = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
    # Predict latent process
    pred <- predict(gp_model, y = y, gp_coords_pred = coord_test, num_neighbors_pred=n+2, 
                    cov_pars = cov_pars, predict_var = TRUE, predict_response = FALSE)
    expect_lt(sum(abs(as.vector(pred$var)-exp_cov_no_nugget[c(1,5,9)])),1E-6)
    
    # Vecchia approximation with 30 neighbors
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30,
                                        vecchia_ordering = "none"), file='NUL')
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                       acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                                                       maxit=1000, convergence_criterion = "relative_change_in_parameters"))
                    , file='NUL')
    cov_pars <- c(0.01836578, 0.07047080, 1.06569263, 0.23039271, 0.11337414, 0.03485905)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 1000)
    
    # Prediction from fitted model
    coord_test <- cbind(c(0.1,0.10001,0.7),c(0.9,0.90001,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                    vecchia_pred_type = "order_obs_first_cond_obs_only")
    expected_mu <- c(0.07272558, 0.07272311, 0.89224890)
    expected_cov <- c(0.5985135, 0.0000000, 0.0000000, 0.0000000, 0.5985250,
                      0.0000000, 0.0000000, 0.0000000, 0.5644097)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    
    # Vecchia approximation with 30 neighbors and random ordering
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30, vecchia_ordering="random"), file='NUL')
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                       acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                                                       maxit=1000, convergence_criterion = "relative_change_in_parameters"))
                    , file='NUL')
    cov_pars <- c(0.01692218, 0.06802646, 1.15762525, 0.26474702, 0.12429957, 0.03962567)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 920)
    
    # Prediction from fitted model
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE,
                    vecchia_pred_type = "order_obs_first_cond_obs_only")
    expected_mu <- c(0.07848313, 0.07848112, 0.95991103)
    expected_cov <- c(0.6017199, 0.0000000, 0.0000000, 0.0000000, 0.6017322,
                      0.0000000, 0.0000000, 0.0000000, 0.5100582)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    
    # Fisher scoring & default ordering
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30, vecchia_ordering="none", y = y,
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc = FALSE,
                                                         delta_rel_conv = 1E-6, maxit=100,
                                                         convergence_criterion = "relative_change_in_parameters")), file='NUL')
    cov_pars <- c(0.01829104, 0.07043221, 1.06577048, 0.23037437, 0.11335918, 0.03484927)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 20)
    
    # Prediction using given parameters
    cov_pars_pred <- c(0.02,1.2,0.9)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE,
                    vecchia_pred_type = "order_obs_first_cond_obs_only")
    expected_mu <- c(0.08665472, 0.08664854, 0.98675487)
    expected_cov <- c(0.1189100, 0.00000000, 0.00000000, 0.00000000,
                      0.1189129, 0.00000000, 0.00000000, 0.00000000, 0.1159135)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE,
                        vecchia_pred_type = "order_obs_first_cond_obs_only")
    expect_lt(sum(abs(pred_var$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])),1E-6)
    
    # Prediction with vecchia_pred_type = "order_obs_first_cond_all"
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE,
                    vecchia_pred_type = "order_obs_first_cond_all")
    expected_mu <- c(0.08665472, 0.08661259, 0.98675487)
    expected_cov <- c(0.11891004, 0.09889262, 0.00000000, 0.09889262,
                      0.11891291, 0.00000000, 0.00000000, 0.00000000, 0.11591347)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE,
                        vecchia_pred_type = "order_obs_first_cond_all")
    expect_lt(sum(abs(pred_var$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])),1E-6)
    
    # Prediction with vecchia_pred_type = "order_pred_first"
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE,
                    vecchia_pred_type = "order_pred_first")
    expected_mu <- c(0.1120768, 0.1119616, 0.5522175)
    expected_cov <- c(1.187731e-01, 9.875550e-02, 1.935232e-07, 9.875550e-02,
                      1.187756e-01, -2.161678e-07, 1.935232e-07, -2.161678e-07, 6.709577e-02)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE,
                        vecchia_pred_type = "order_pred_first")
    expect_lt(sum(abs(pred_var$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])),1E-6)
    
    # Prediction with vecchia_pred_type = "latent_order_obs_first_cond_obs_only"
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE,
                    vecchia_pred_type = "latent_order_obs_first_cond_obs_only", predict_response = FALSE)
    expected_mu <- c(0.08513257, 0.08512608, 0.90542680)
    expected_cov <- c(1.189086e-01, 7.322771e-03, -7.263024e-07, 7.322771e-03,
                      1.189114e-01, -7.261547e-07, -7.263024e-07, -7.261547e-07, 1.149206e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE,
                        vecchia_pred_type = "latent_order_obs_first_cond_obs_only", predict_response = FALSE)
    expect_lt(sum(abs(pred_var$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])),1E-6)
    
    # Prediction with vecchia_pred_type = "latent_order_obs_first_cond_all"
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = cov_pars_pred, predict_cov_mat = TRUE,
                    vecchia_pred_type = "latent_order_obs_first_cond_all", predict_response = FALSE)
    expected_mu <- c(0.08513257, 0.08512602, 0.90542680)
    expected_cov <- c(1.189086e-01, 9.889112e-02, -7.263386e-07, 9.889112e-02,
                      1.189114e-01, -7.261806e-07, -7.263385e-07, -7.261805e-07, 1.149206e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    pred_var <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                        cov_pars = cov_pars_pred, predict_var = TRUE,
                        vecchia_pred_type = "latent_order_obs_first_cond_all", predict_response = FALSE)
    expect_lt(sum(abs(pred_var$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred_var$var)-expected_cov[c(1,5,9)])),1E-6)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1.6,0.2),y=y)
    expect_lt(abs(nll-126.6644435),1E-6)
  })
  
  test_that("Vecchia approximation for Gaussian process model with linear regression term ", {
    
    y <- eps + X%*%beta + xi
    # Fit model
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n+2,
                                           vecchia_ordering = "none", y = y, X=X,
                                           params = list(optimizer_cov = "fisher_scoring", optimizer_coef = "wls", std_dev = TRUE,
                                                         delta_rel_conv = 1E-6, use_nesterov_acc = FALSE,
                                                         convergence_criterion = "relative_change_in_parameters")), file='NUL')
    cov_pars <- c(0.008461342, 0.069973492, 1.001562822, 0.214358560, 0.094656409, 0.029400407)
    coef <- c(2.30780026, 0.21365770, 1.89951426, 0.09484768)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    
    # Prediction 
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE,
                    vecchia_pred_type = "latent_order_obs_first_cond_all", predict_response = FALSE)
    expected_mu <- c(1.196952, 4.063324, 3.156427)
    expected_cov <- c(6.305383e-01, 1.358861e-05, 8.317903e-08, 1.358861e-05,
                      3.469270e-01, 2.686334e-07, 8.317903e-08, 2.686334e-07, 4.255400e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  })
  
  test_that("Vecchia approximation for Gaussian process model with cluster_id's not constant ", {
    
    y <- eps + xi
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30,
                                           vecchia_ordering = "none", y = y, cluster_ids = cluster_ids,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.05, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, delta_rel_conv = 1E-6,
                                                         convergence_criterion = "relative_change_in_parameters")), file='NUL')
    cov_pars <- c(0.03359375, 0.07751222, 1.07019293,
                  0.22080542, 0.12201912, 0.03761778)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 662)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30,
                                           vecchia_ordering = "none", y = y, cluster_ids = cluster_ids,
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc = FALSE, delta_rel_conv = 1E-6,
                                                         convergence_criterion = "relative_change_in_parameters")), file='NUL')
    cov_pars <- c(0.03359131, 0.07751098, 1.07019567, 0.22080490, 0.12201865, 0.03761748)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
    expect_equal(gp_model$get_num_optim_iter(), 15)
    
    # Prediction
    coord_test <- cbind(c(0.1,0.2,0.1001),c(0.9,0.4,0.9001))
    cluster_ids_pred = c(1,3,1)
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = 30, 
                                        vecchia_ordering = "none", cluster_ids = cluster_ids), file='NUL')
    capture.output( pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                                             cluster_ids_pred = cluster_ids_pred,
                                             vecchia_pred_type = "order_obs_first_cond_all",
                                             cov_pars = c(0.1,1,0.15), predict_cov_mat = TRUE), file='NUL')
    expected_mu <- c(-0.01438585, 0.00000000, -0.01500132)
    expected_cov <- c(0.7430552, 0.0000000, 0.6423148, 0.0000000,
                      1.1000000, 0.0000000, 0.6423148, 0.0000000, 0.7434589)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  })
  
  test_that("Vecchia approximation for Gaussian process model 
            with multiple observations at the same location ", {
              
              y <- eps_multiple + xi
              capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                                     gp_approx = "vecchia", num_neighbors = n-1, y = y,
                                                     vecchia_ordering = "none", 
                                                     params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                                   lr_cov = 0.1, use_nesterov_acc = FALSE,
                                                                   delta_rel_conv = 1E-6, maxit = 500)), file='NUL')
              cov_pars <- c(0.037145465, 0.006065652, 1.151982610, 0.434770575, 0.191648634, 0.102375515)
              expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-4)
              expect_equal(gp_model$get_num_optim_iter(), 12)
              
              # Fisher scoring
              capture.output( gp_model <- fitGPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                                     gp_approx = "vecchia", num_neighbors = n-1, y = y, 
                                                     vecchia_ordering = "none", 
                                                     params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                                   use_nesterov_acc = FALSE, delta_rel_conv = 1E-6,
                                                                   convergence_criterion = "relative_change_in_parameters")), file='NUL')
              cov_pars <- c(0.037136462, 0.006064181, 1.153630335, 0.435788570, 0.192080613, 0.102631006)
              expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-5)
              expect_equal(gp_model$get_num_optim_iter(), 14)
              
              # Prediction
              coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
              cluster_ids_pred = c(1,3,1)
              capture.output( gp_model <- GPModel(gp_coords = coords_multiple, cov_function = "exponential",
                                                  gp_approx = "vecchia", num_neighbors = n+2,
                                                  vecchia_ordering = "none"), file='NUL')
              pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                                       vecchia_pred_type = "order_obs_first_cond_all",
                                       cov_pars = c(0.1,1,0.15), predict_cov_mat = TRUE)
              expected_mu <- c(-0.1460550, 1.0042814, 0.7840301)
              expected_cov <- c(0.6739502109, 0.0008824337, -0.0003815281, 0.0008824337,
                                0.6060039551, -0.0004157361, -0.0003815281, -0.0004157361, 0.7851787946)
              expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
              expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
              
            })
  
  test_that("Vecchia approximation for Gaussian process and two random coefficients ", {
    
    y <- eps_svc + xi
    # Fit model using gradient descent with Nesterov acceleration
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n-1,
                                           gp_rand_coef_data = Z_SVC, vecchia_ordering = "none", y = y,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5, maxit=10)), file='NUL')
    expected_values <- c(0.25740068, 0.22608704, 0.83503539, 0.41896403, 0.15039055,
                         0.10090869, 1.61010233, 0.84207763, 0.09015444, 0.07106099,
                         0.25064640, 0.62279880, 0.08720822, 0.32047865)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    
    # Prediction
    capture.output( gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, cov_function = "exponential",
                                        gp_approx = "vecchia", num_neighbors = n+2,
                                        vecchia_ordering = "none"), file='NUL')
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(0.1,1,0.1,0.8,0.15,1.1,0.08),
                             predict_cov_mat = TRUE, vecchia_pred_type = "order_obs_first_cond_all")
    expected_mu <- c(-0.1669209, 1.6166381, 0.2861320)
    expected_cov <- c(9.643323e-01, 3.536846e-04, -1.783557e-04, 3.536846e-04,
                      5.155009e-01, 4.554321e-07, -1.783557e-04, 4.554321e-07, 7.701614e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = n-1,
                                           gp_rand_coef_data = Z_SVC, vecchia_ordering = "none", y = y,
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc= FALSE, maxit=5)), file='NUL')
    expected_values <- c(0.000242813, 0.176573955, 1.008181385, 0.397341267, 0.141084495, 
                         0.070671768, 1.432715033, 0.708039197, 0.055598038, 0.048988825, 
                         0.430573036, 0.550644708, 0.038976112, 0.106110593)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-149.4422184),1E-5)
    
    # Fit model using gradient descent with Nesterov acceleration
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30,
                                           gp_rand_coef_data = Z_SVC, vecchia_ordering = "none", y = y,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = FALSE, maxit=10)), file='NUL')
    expected_values <- c(0.40353068, 0.25914955, 0.70029206, 0.42641715, 0.16737937,
                         0.12728492, 1.14005726, 0.80008664, 0.11213991, 0.11988861,
                         0.49234325, 0.70169773, 0.08919812, 0.20748266)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    
    # Prediction
    capture.output( gp_model <- GPModel(gp_coords = coords, gp_rand_coef_data = Z_SVC, 
                                        cov_function = "exponential", gp_approx = "vecchia", 
                                        num_neighbors = 30, vecchia_ordering = "none"), file='NUL')
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    Z_SVC_test <- cbind(c(0.1,0.3,0.7),c(0.5,0.2,0.4))
    pred <- gp_model$predict(y = y, gp_coords_pred = coord_test,
                             gp_rand_coef_data_pred=Z_SVC_test,
                             cov_pars = c(0.1,1,0.1,0.8,0.15,1.1,0.08),
                             predict_cov_mat = TRUE, vecchia_pred_type = "order_obs_first_cond_all")
    expected_mu <- c(-0.1688452, 1.6191401, 0.9079859)
    expected_cov <- c(0.9643376, 0.0000000, 0.0000000, 0.0000000, 0.5155902,
                      0.0000000, 0.0000000, 0.0000000, 1.0239525)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 30,
                                           gp_rand_coef_data = Z_SVC, vecchia_ordering = "none", y = y,
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc= FALSE, maxit=5)), file='NUL')
    expected_values <- c(0.04427393, 0.20241802, 0.99631971, 0.43532853, 0.17012209,
                         0.09359131, 1.40235416, 0.78811422, 0.07835884, 0.07553935,
                         0.77593979, 0.64290472, 0.03673858, 0.07640891)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,0.1,0.8,0.15,1.1,0.08),y=y)
    expect_lt(abs(nll-152.6033062),1E-6)
  })
  
  test_that("Wendland covariance function for Gaussian process model ", {
    
    y <- eps + xi
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "wendland", 
                                        cov_fct_taper_shape = 0, cov_fct_taper_range = 0.1), file='NUL')
    capture.output( fit(gp_model, y = y, params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                       acc_rate_cov = 0.5)) , file='NUL')
    cov_pars <- c(0.002911765, 0.116338096, 0.993996193, 0.211276385)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 280)
    # Prediction using given parameters
    capture.output( gp_model <- GPModel(gp_coords = coords, cov_function = "wendland", 
                                        cov_fct_taper_shape = 1, cov_fct_taper_range = 2), 
                    file='NUL')
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = c(0.02,1.2), predict_cov_mat = TRUE)
    expected_mu <- c(-0.008405567, 1.493836307, 0.720565199)
    expected_cov <- c(2.933992e-02, 2.223241e-06, 1.352544e-05, 2.223241e-06, 2.496193e-02,
                      1.130906e-05, 1.352544e-05, 1.130906e-05, 2.405649e-02)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    # Prediction of variances only
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = c(0.02,1.2), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),1E-6)
    # Fisher scoring
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland", 
                                           cov_fct_taper_shape = 0, cov_fct_taper_range = 0.1, y = y,
                                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                         use_nesterov_acc = FALSE,
                                                         delta_rel_conv = 1E-6)), file='NUL')
    cov_pars <- c(2.946448e-08, 1.599928e-01, 1.391589e+00, 2.933997e-01)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.02,1.2),y=y)
    expect_lt(abs(nll-136.9508962),1E-6)
    
    # Other taper shapes
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland", 
                                           cov_fct_taper_shape = 1, cov_fct_taper_range = 0.15, y = y,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5)), file='NUL')
    cov_pars <- c(0.0564441, 0.0497191, 0.9921285, 0.1752661)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 19)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = c(0.02,1.2), predict_cov_mat = TRUE)
    expected_mu <- c(-0.007404038, 1.487424320, 0.200022114)
    expected_cov <- c(1.113020e+00, -6.424533e-30, -4.186440e-22, -6.424533e-30, 3.522739e-01,
                      9.018454e-10, -4.186440e-22, 9.018454e-10, 6.092985e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
    # Other taper shapes
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "wendland", 
                                           cov_fct_taper_shape = 2, cov_fct_taper_range = 0.08, y = y,
                                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                                         lr_cov = 0.1, use_nesterov_acc = TRUE,
                                                         acc_rate_cov = 0.5)), file='NUL')
    cov_pars <- c(0.00327103, 0.06579668, 1.08812978, 0.18151366)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)
    expect_equal(gp_model$get_num_optim_iter(), 187)
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test,
                    cov_pars = c(0.02,1.2), predict_cov_mat = TRUE)
    expected_mu <- c(-2.314198e-05, 8.967992e-01, 2.430054e-02)
    expected_cov <- c(1.2200000, 0.0000000, 0.0000000, 0.0000000, 0.9024792, 0.0000000, 0.0000000, 0.0000000, 1.1887157)
    expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  })
  
  test_that("Tapering ", {
    y <- eps + X%*%beta + xi
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    # No tapering
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, X = X, params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    cov_pars <- c(0.01621846, 0.07384498, 0.99717680, 0.21704099, 0.09616230, 0.03034715)
    coef <- c(2.30554610, 0.21565230, 1.89920767, 0.09567547)
    num_it <- 99
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expected_mu <- c(1.195910242, 4.060125034, 3.15963272)
    expected_cov <- c(6.304732e-01, 1.313601e-05, 1.008080e-07, 1.313601e-05, 3.524404e-01, 
                      3.699813e-07, 1.008080e-07, 3.699813e-07, 4.277339e-01)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
    # With tapering and very large tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "tapering",
                                           y = y, X = X, cov_fct_taper_shape = 0, cov_fct_taper_range = 1e6,
                                           params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), num_it)
    # Prediction 
    pred <- predict(gp_model, gp_coords_pred = coord_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE_STRICT)
    
    # With tapering and smaller tapering range
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "tapering",
                                           y = y, X = X, cov_fct_taper_shape = 0, cov_fct_taper_range = 0.5,
                                           params = DEFAULT_OPTIM_PARAMS_STD), file='NUL')
    cov_pars_tap <- c(0.02593993, 0.07560715, 0.99435221, 0.21816716, 0.17712808, 0.09797175)
    coef_tap <- c(2.32410488, 0.20610507, 1.89498931, 0.09533541)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_tap)),TOLERANCE_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef_tap)),TOLERANCE_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 75)
  })
  
  test_that("Saving a GPModel and loading from file works ", {
    
    y <- eps + xi
    coord_test <- cbind(c(0.1,0.2,0.7),c(0.9,0.4,0.55))
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           y = y, params = DEFAULT_OPTIM_PARAMS_STD), 
                    file='NUL')
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    rm(gp_model)
    # Load from file and make predictions again
    gp_model_loaded <- loadGPModel(filename = filename)
    pred_loaded <- predict(gp_model_loaded, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    
    # With Vecchia approximation
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 20, num_neighbors_pred = 50,
                                           vecchia_ordering = "none", y = y, params = DEFAULT_OPTIM_PARAMS_STD), 
                    file='NUL')
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    rm(gp_model)
    # Load from file and make predictions again
    capture.output( gp_model_loaded <- loadGPModel(filename = filename), file='NUL')
    pred_loaded <- predict(gp_model_loaded, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    
    # With Vecchia approximation and random ordering
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "vecchia", num_neighbors = 20, num_neighbors_pred = 50,
                                           vecchia_ordering = "random", y = y, params = DEFAULT_OPTIM_PARAMS_STD), 
                    file='NUL')
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    rm(gp_model)
    # Load from file and make predictions again
    capture.output( gp_model_loaded <- loadGPModel(filename = filename), file='NUL')
    pred_loaded <- predict(gp_model_loaded, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
    
    # With Tapering
    capture.output( gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                                           gp_approx = "tapering", cov_fct_taper_range = 0.5, cov_fct_taper_shape = 1.,
                                           y = y, params = DEFAULT_OPTIM_PARAMS_STD), 
                    file='NUL')
    pred <- predict(gp_model, y=y, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    # Save model to file
    filename <- tempfile(fileext = ".json")
    saveGPModel(gp_model, filename = filename)
    rm(gp_model)
    # Load from file and make predictions again
    capture.output( gp_model_loaded <- loadGPModel(filename = filename), file='NUL')
    pred_loaded <- predict(gp_model_loaded, gp_coords_pred = coord_test, predict_cov_mat = TRUE)
    expect_equal(pred$mu, pred_loaded$mu)
    expect_equal(pred$cov, pred_loaded$cov)
  })
  
}

