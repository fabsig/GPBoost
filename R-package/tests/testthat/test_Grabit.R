context("Grabit")

TOLERANCE <- 1E-3

# Function that simulates uniform random variables
sim_rand_unif <- function(n, init_c=0.1){
  mod_lcg <- 134456 # modulus for linear congruential generator (random0 used)
  sim <- rep(NA, n)
  sim[1] <- floor(init_c * mod_lcg)
  for(i in 2:n) sim[i] <- (8121 * sim[i-1] + 28411) %% mod_lcg
  return(sim / mod_lcg)
}
# Function for non-linear mean
sim_friedman3 <- function(n, n_irrelevant=5){
  X <- matrix(sim_rand_unif(4*n,init_c=0.54234),ncol=4)
  X[,1] <- 100*X[,1]
  X[,2] <- X[,2]*pi*(560-40)+40*pi
  X[,4] <- X[,4]*10+1
  f <- sqrt(10)*atan((X[,2]*X[,3]-1/(X[,2]*X[,4]))/X[,1])
  X <- cbind(rep(1,n),X)
  if(n_irrelevant>0) X <- cbind(X,matrix(sim_rand_unif(n_irrelevant*n,init_c=0.74534),ncol=n_irrelevant))
  return(list(X=X,f=f))
}

n <- 1000
sim_train <- sim_friedman3(n=n)
sim_test <- sim_friedman3(n=n)
X <- sim_train$X
y <- sim_train$f
X_test <- sim_test$X
y_test <- sim_test$f
# apply censoring
yu <- 4.8
yl <- 3.5
y[y>=yu] <- yu
y[y<=yl] <- yl
# # censoring fractions
# sum(y==yu) / n
# sum(y==yl) / n

expect_lt(sum(abs(tail(y)-c(4.594936, 3.500000, 3.500000,
                            3.500000, 4.800000, 4.724953))),TOLERANCE)

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  # train model and make predictions
  dtrain <- gpb.Dataset(data = X, label = y)
  bst <- gpb.train(data = dtrain, nrounds = 100, objective = "tobit",
                   verbose = 0, yl = yl, yu = yu)
  y_pred <- predict(bst, data = X_test)
  expect_lt(sum(abs(tail(y_pred)-c(4.5605215, 2.0462860, -0.4051916, 
                                   1.6789510, 8.4034647, 4.7509841))),TOLERANCE)
  # applying no censoring
  bst <- gpb.train(data = dtrain, nrounds = 100, objective = "tobit",
                   verbose = 0, yl = -Inf, yu = Inf)
  y_pred_no_censor <- predict(bst, data = X_test)
  bst <- gpb.train(data = dtrain, nrounds = 100, objective = "regression_l2",
                   verbose = 0)
  y_pred_l2 <- predict(bst, data = X_test)
  expect_lt(sum(abs(y_pred_no_censor - y_pred_l2)),TOLERANCE)
  # not providing limits = no censoring
  bst <- gpb.train(data = dtrain, nrounds = 100, objective = "tobit",
                   verbose = 0)
  y_pred_no_limits <- predict(bst, data = X_test)
  expect_lt(sum(abs(y_pred_no_limits - y_pred_l2)),TOLERANCE)
  
}
