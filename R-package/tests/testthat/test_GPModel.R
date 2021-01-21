context("GPModel")

# Simulate data
n <- 10 # number of samples
m <- 5 # number of categories / levels for grouping variable
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
b1 <- 1:5-3
eps <- Z1 %*% b1
y <- Z1 %*% b1 + (1:5-2)/100

windows_flag = grepl('Windows', Sys.info()[['sysname']])

test_that("grouped random effects model", {
  #Estimation
  gp_model <- GPModel(group_data = group)
  fit(gp_model, y = y, std_dev = TRUE)
  cov_pars <- rbind(c(0.0002000001,2.020100),c(0.0001264912,1.277687))
  cov_pars_est <- gp_model$get_cov_pars()
  
  expect_lt(sum(abs(cov_pars_est-cov_pars)),1E-6)
  expect_equal(dim(cov_pars)[2], 2)
  expect_equal(dim(cov_pars)[1], 2)
  
  #Prediction
  group_test <- c(1,2,7)
  pred <- predict(gp_model, group_data_pred = group_test, predict_cov_mat = TRUE)
  pred_exp <- c(-2.0049008,-0.9849512,0.0000000)
  expect_lt(sum(abs(pred$mu-pred_exp)),1E-6)
  
  expect_equal(pred$cov[1,2], 0)
  expect_equal(pred$cov[3,1], 0)
  diag_exp <- c(0.0002999952,0.0002999952,2.0203000001)
  expect_lt(sum(abs(diag(pred$cov)-diag_exp)),1E-6)
  
})


