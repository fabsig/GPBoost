context("GPModel")

# Create data
n <- 10 # number of samples
# Data for single-level grouped random effects model
m <- 5 # number of categories / levels for grouping variable
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
b1 <- 1:5-3
# Second random effect
n_obs_gr <- n/m # number of sampels per group
group2 <- rep(1,n) # grouping variable
for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
b2 <- c(-1,1)
# Random slope
x <- cos((1:n-n/2)^2*5*pi/n) # covariate data for random slope
x <- cos((1:n-n/2)^2*5.5*pi/n) # covariate data for random slope
Z3 <- diag(x) %*% Z1
b3 <- sin((1:5-3))^2/3
# Data for linear mixed effects model
X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # desing matrix / covariate data for fixed effect
beta <- c(2,2) # regression coefficents
# Error term
xi <- (1:5-2)/100
# cluster_ids 
cluster_ids <- c(rep(1,4),rep(2,6))


test_that("grouped random effects model ", {
  
  # Single-level grouped random effects model
  y <- Z1 %*% b1 + xi
  
  # Estimation
  gp_model <- GPModel(group_data = group)
  fit(gp_model, y = y, std_dev = TRUE, params = list(optimizer_cov = "fisher_scoring"))
  cov_pars <- rbind(c(0.0002000001,2.020100),c(0.0001264912,1.277687))
  cov_pars_est <- gp_model$get_cov_pars()
  expect_lt(sum(abs(cov_pars_est-cov_pars)),1E-6)
  expect_equal(dim(cov_pars_est)[2], 2)
  expect_equal(dim(cov_pars_est)[1], 2)
  expect_equal(gp_model$get_num_optim_iter(), 14)
  
  # Other optimization settings
  gp_model <- fitGPModel(group_data = group, y = y, std_dev = FALSE,
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc = FALSE,
                                       maxit = 1000))
  cov_pars_est <- gp_model$get_cov_pars()
  expect_lt(sum(abs(cov_pars_est-cov_pars[1,])),1E-5)
  expect_equal(class(cov_pars_est), "numeric")
  expect_equal(length(cov_pars_est), 2)
  expect_equal(gp_model$get_num_optim_iter(), 128)
  
  gp_model <- fitGPModel(group_data = group, y = y, std_dev = FALSE,
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc = TRUE,
                                       acc_rate_cov = 0.5, maxit = 1000))
  cov_pars_est <- gp_model$get_cov_pars()
  expect_lt(sum(abs(cov_pars_est-cov_pars[1,])),1E-4)
  expect_equal(class(cov_pars_est), "numeric")
  expect_equal(length(cov_pars_est), 2)
  expect_equal(gp_model$get_num_optim_iter(), 43)
  
  # Prediction 
  gp_model <- GPModel(group_data = group)
  group_test <- c(1,2,7)
  pred <- predict(gp_model, y=y, group_data_pred = group_test,
                  cov_pars = c(0.5,1.5), predict_cov_mat = TRUE)
  expected_mu <- c(-1.7185714, -0.8442857, 0.0000000)
  expected_cov <- c(0.7142857, 0.0000000, 0.0000000, 0.0000000,
                    0.7142857, 0.0000000, 0.0000000, 0.0000000, 2.0000000)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
  
  # Prediction from fitted model
  gp_model <- fitGPModel(group_data = group, y = y, std_dev = FALSE,
                         params = list(optimizer_cov = "fisher_scoring"))
  group_test <- c(1,2,7)
  pred <- predict(gp_model, group_data_pred = group_test, predict_cov_mat = TRUE)
  expected_mu <- c(-2.0049008,-0.9849512,0.0000000)
  expected_cov <- c(0.0002999951, 0.0000000000, 0.0000000000, 0.0000000000,
                    0.0002999951, 0.0000000000, 0.0000000000, 0.0000000000, 2.0202999334)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)

  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1),y=y)
  expect_lt(abs(nll-10.10277),1E-5)
  
  # Do optimization using optim and e.g. Nelder-Mead
  gp_model <- GPModel(group_data = group)
  opt <- optim(par=c(1,1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
  expect_lt(sum(abs(opt$par-cov_pars[1,])),1E-3)
  expect_lt(abs(opt$value-(-3.612738)),1E-5)
  expect_equal(as.integer(opt$counts[1]), 121)
})


test_that("linear mixed effects model with grouped random effects ", {

  y <- Z1 %*% b1 + xi + X%*%beta 
  # Fit model
  gp_model <- fitGPModel(group_data = group,
                         y = y, X = X, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring",
                                       optimizer_coef = "wls"))
  cov_pars <- rbind(c(0.0001583333,2.020097),c(0.0001001388,1.277672))
  coef <- rbind(c(2.0100000,1.98582397),c(0.6356377,0.01235927))
  cov_pars_est <- gp_model$get_cov_pars()
  coef_est <- gp_model$get_coef()
  expect_lt(sum(abs(cov_pars_est-cov_pars)),1E-6)
  expect_lt(sum(abs(coef_est-coef)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 15)
  
  # Prediction 
  group_test <- c(1,2,7)
  X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
  pred <- predict(gp_model, group_data_pred = group_test,
                  X_pred = X_test, predict_cov_mat = TRUE)
  expected_mu <- c(-1.006165, 1.420536, 2.804330)
  expected_cov <- c(0.0002374969, 0.0000000000, 0.0000000000, 0.0000000000,
                    0.0002374969, 0.0000000000, 0.0000000000, 0.0000000000, 2.0202555439)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})


test_that("two crossed random effects and a random slope ", {
  
  y <- Z1%*%b1 + Z2%*%b2 + Z3%*%b3 + xi
  # Fisher scoring
  gp_model <- fitGPModel(group_data = cbind(group,group2),
                         group_rand_coef_data = x,
                         ind_effect_group_rand_coef = 1,
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring", maxit=5))
  expected_values <- c(0.01600543, 0.05277050, 2.42786777, 1.70178763,
                       1.21670023, 1.52318022, 0.09156979, 0.11999864)
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
  expected_mu <- c(-0.005910141, -1.876016819, 0.000000000)
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
  expected_values <- c(0.6401681, 0.9257807, 1.0408789, 1.0689150,
                       0.6810692, 1.0289172, 0.5225324, 1.2367742)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 5)
  
  # Evaluate negative log-likelihood
  nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,2,1.5),y=y)
  expect_lt(abs(nll-18.07796),1E-5)
})


test_that("not constant cluster_id's for grouped random effects ", {
  
  y <- Z1 %*% b1 + xi
  gp_model <- fitGPModel(group_data = group, cluster_ids = cluster_ids,
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "fisher_scoring", maxit=100))
  expected_values <- c(0.0002000000, 0.0001264911, 2.0201000000, 1.2776866674)
  expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),1E-6)
  expect_equal(gp_model$get_num_optim_iter(), 14)
  
  # gradient descent
  gp_model <- fitGPModel(group_data = group, cluster_ids = cluster_ids,
                         y = y, std_dev = TRUE,
                         params = list(optimizer_cov = "gradient_descent",
                                       lr_cov = 0.1, use_nesterov_acc = FALSE, maxit = 1000))
  expected_values <- c(0.0002000000, 0.0001264911, 2.0201000000, 1.2776866674)
  cov_pars_est <- gp_model$get_cov_pars()
  expect_lt(sum(abs(cov_pars_est-expected_values)),1E-5)
  expect_equal(gp_model$get_num_optim_iter(), 128)
  
  # Prediction
  group_data_pred = c(1,1,7)
  cluster_ids_pred = c(1,3,1)
  gp_model <- GPModel(group_data = group, cluster_ids = cluster_ids)
  pred <- gp_model$predict(y = y, group_data_pred = group_data_pred,
                           cluster_ids_pred = cluster_ids_pred,
                           cov_pars = c(0.75,1.25), predict_cov_mat = TRUE)
  expected_mu <- c(-1.542308, 0.000000, 0.000000)
  expected_cov <- c(1.038462, 0.000000, 0.000000, 0.000000,
                    2.000000, 0.000000, 0.000000, 0.000000, 2.000000)
  expect_lt(sum(abs(pred$mu-expected_mu)),1E-6)
  expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),1E-6)
})
