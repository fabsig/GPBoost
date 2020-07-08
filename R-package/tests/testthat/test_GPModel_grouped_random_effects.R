context("GPModel_grouped_random_effects")

# Function that simulates uniform random variables
sim_rand_unif <- function(n, init_c=0.1){
  mod_lcg <- 134456 # modulus for linear congruential generator (random0 used)
  sim <- rep(NA, n)
  sim[1] <- floor(init_c * mod_lcg)
  for(i in 2:n) sim[i] <- (8121 * sim[i-1] + 28411) %% mod_lcg
  return(sim / mod_lcg)
}

# Create data
n <- 10 # number of samples
# First grouped random effects model
m <- 5 # number of categories / levels for grouping variable
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
b1 <- 1:m-3
# Second random effect
n_obs_gr <- n/m # number of sampels per group
group2 <- rep(1,n) # grouping variable
for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
b2 <- c(-1,1)
# Random slope / coefficient
x <- cos((1:n-n/2)^2*5*pi/n) # covariate data for random slope
x <- cos((1:n-n/2)^2*5.5*pi/n) # covariate data for random slope
Z3 <- diag(x) %*% Z1
b3 <- sin((1:m-3))^2/3
# Error term
xi <- qnorm(sim_rand_unif(n=n, init_c=0.1)) / 50
# Data for linear mixed effects model
X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # desing matrix / covariate data for fixed effect
beta <- c(2,2) # regression coefficents
# cluster_ids 
cluster_ids <- c(rep(1,0.4*n),rep(2,0.6*n))


test_that("single level grouped random effects model ", {

  y <- as.vector(Z1 %*% b1) + xi
  # Estimation
  gp_model <- GPModel(group_data = group)
  fit(gp_model, y = y, std_dev = TRUE, params = list(optimizer_cov = "fisher_scoring"))
  cov_pars <- c(0.0006328951, 0.0004002780, 2.0227167347, 1.2794785465)
  cov_pars_est <- as.vector(gp_model$get_cov_pars())
  expect_lt(sum(abs(cov_pars_est-cov_pars)),1E-6)
  expect_equal(dim(gp_model$get_cov_pars())[2], 2)
  expect_equal(dim(gp_model$get_cov_pars())[1], 2)
  expect_equal(gp_model$get_num_optim_iter(), 13)
  
  # Other optimization settings
  gp_model <- fitGPModel(group_data = group, y = y, std_dev = FALSE,
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                       maxit = 1000))
  cov_pars_est <- as.vector(gp_model$get_cov_pars())
  expect_lt(sum(abs(cov_pars_est-cov_pars[c(1,3)])),1E-5)
  expect_equal(class(cov_pars_est), "numeric")
  expect_equal(length(cov_pars_est), 2)
  expect_equal(gp_model$get_num_optim_iter(), 124)
  
  gp_model <- fitGPModel(group_data = group, y = y, std_dev = FALSE,
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                       acc_rate_cov = 0.5, maxit = 1000))
  cov_pars_est <- as.vector(gp_model$get_cov_pars())
  expect_lt(sum(abs(cov_pars_est-cov_pars[c(1,3)])),1E-5)
  expect_equal(class(cov_pars_est), "numeric")
  expect_equal(length(cov_pars_est), 2)
  expect_equal(gp_model$get_num_optim_iter(), 52)
  
  # Prediction 
  gp_model <- GPModel(group_data = group)
  group_test <- c(1,2,7)
  pred <- predict(gp_model, y=y, group_data_pred = group_test,
                  cov_pars = c(0.5,1.5), predict_cov_mat = TRUE)
  expected_mu <- c(-1.7303927, -0.8722256,  0.0000000)
  expected_cov <- c(0.7142857, 0.0000000, 0.0000000, 0.0000000,
                    0.7142857, 0.0000000, 0.0000000, 0.0000000, 2.0000000)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Prediction from fitted model
  gp_model <- fitGPModel(group_data = group, y = y, std_dev = FALSE,
                         params = list(optimizer_cov = "fisher_scoring"))
  group_test <- c(1,2,7)
  pred <- predict(gp_model, group_data_pred = group_test, predict_cov_mat = TRUE)
  expected_mu <- c(-2.018476, -1.017437, 0.0000000)
  expected_cov <- c(0.0009492931, 0.0000000000, 0.0000000000, 0.0000000000,
                    0.0009492931, 0.0000000000, 0.0000000000, 0.0000000000, 2.0233496298)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)

  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1),y=y)
  expect_lt(abs(nll-10.12033),1E-5)
  
  # Do optimization using optim and e.g. Nelder-Mead
  gp_model <- GPModel(group_data = group)
  opt <- optim(par=c(1,1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
  expect_lt(sum(abs(opt$par-cov_pars[c(1,3)])),1E-3)
  expect_lt(abs(opt$value-(-0.7292666)),1E-5)
  expect_equal(as.integer(opt$counts[1]), 145)
  
  # Use non-ordered grouping data
  shuffle_ind <- c(10, 2, 1, 9, 7, 8, 4, 5, 3, 6)
  gp_model <- GPModel(group_data = group[shuffle_ind])
  fit(gp_model, y = y[shuffle_ind], std_dev = TRUE, params = list(optimizer_cov = "fisher_scoring"))
  cov_pars_est <- as.vector(gp_model$get_cov_pars())
  expect_lt(sum(abs(cov_pars_est-cov_pars)),1E-6)
})


test_that("linear mixed effects model with grouped random effects ", {

  y <- Z1 %*% b1 + X%*%beta + xi
  # Fit model
  gp_model <- fitGPModel(group_data = group,
                         y = y, X = X, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring",
                                       optimizer_coef = "wls"))
  cov_pars <- c(0.0005890585, 0.0003725533, 2.0227180604, 1.2794655172)
  coef <- c(1.99219041, 0.63608373, 2.01453829, 0.02383676)
  cov_pars_est <- as.vector(gp_model$get_cov_pars())
  coef_est <- as.vector(gp_model$get_coef())
  expect_lt(sum(abs(cov_pars_est-cov_pars)),1E-6)
  expect_lt(sum(abs(coef_est-coef)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 13)
  
  # Prediction 
  group_test <- c(1,2,7)
  X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
  pred <- predict(gp_model, group_data_pred = group_test,
                  X_pred = X_test, predict_cov_mat = TRUE)
  expected_mu <- c(-1.017224, 1.376914, 2.798006)
  expected_cov <- c(0.0008835448, 0.0000000000, 0.0000000000, 0.0000000000,
                    0.0008835448, 0.0000000000, 0.0000000000, 0.0000000000, 2.0233071188)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})


# Ignore [GPBoost] [Warning]
test_that("two crossed random effects and a random slope ", {
  
  y <- Z1%*%b1 + Z2%*%b2 + Z3%*%b3 + xi
  # Fisher scoring
  gp_model <- fitGPModel(group_data = cbind(group,group2),
                         group_rand_coef_data = x,
                         ind_effect_group_rand_coef = 1,
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring", maxit=5))
  expected_values <- c(0.01328599, 0.05137801, 2.42699005, 1.70170676,
                       1.23536565, 1.54308011, 0.10065015, 0.12494428)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 5)
  
  # Prediction
  gp_model <- GPModel(group_data = cbind(group,group2),
                      group_rand_coef_data = x, ind_effect_group_rand_coef = 1)
  group_data_pred = cbind(c(1,1,7),c(2,1,3))
  group_rand_coef_data_pred = c(0,0.1,0.3)
  pred <- gp_model$predict(y = y, group_data_pred=group_data_pred,
                   group_rand_coef_data_pred=group_rand_coef_data_pred,
                   cov_pars = c(0.1,1,2,1.5), predict_cov_mat = TRUE)
  expected_mu <- c(-0.01123019, -1.88978313, 0.000000000)
  expected_cov <- c(0.8005528, 0.6179886, 0.0000000, 0.6179886,
                    0.7798839, 0.0000000, 0.0000000, 0.0000000, 3.2350000)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Gradient descent
  gp_model <- fitGPModel(group_data = cbind(group,group2),
                         group_rand_coef_data = x,
                         ind_effect_group_rand_coef = 1,
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc= FALSE, maxit=5))
  expected_values <- c(0.6435813, 0.9308820, 1.0414605, 1.0717530,
                       0.6866298, 1.0360369, 0.5270886, 1.2441796)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 5)
  
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,2,1.5),y=y)
  expect_lt(abs(nll-18.06998),1E-5)
})


test_that("not constant cluster_id's for grouped random effects ", {
  
  y <- Z1 %*% b1 + xi
  gp_model <- fitGPModel(group_data = group, cluster_ids = cluster_ids,
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring", maxit=100))
  expected_values <- c(0.0006328951, 0.0004002780, 2.0227167465, 1.2794785483)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 13)
  
  # gradient descent
  gp_model <- fitGPModel(group_data = group, cluster_ids = cluster_ids,
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc = FALSE, maxit = 1000))
  expected_values <- c(0.0006328973, 0.0004002794, 2.0227113725, 1.2794751488)
  cov_pars_est <- gp_model$get_cov_pars()
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 124)
  
  # Prediction
  group_data_pred = c(1,1,7)
  cluster_ids_pred = c(1,3,1)
  gp_model <- GPModel(group_data = group, cluster_ids = cluster_ids)
  pred <- gp_model$predict(y = y, group_data_pred = group_data_pred,
                           cluster_ids_pred = cluster_ids_pred,
                           cov_pars = c(0.75,1.25), predict_cov_mat = TRUE)
  expected_mu <- c(-1.552917, 0.000000, 0.000000)
  expected_cov <- c(1.038462, 0.000000, 0.000000, 0.000000,
                    2.000000, 0.000000, 0.000000, 0.000000, 2.000000)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})
