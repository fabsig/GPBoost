context("generalized_GPBoost_combined_boosting_GP_random_effects")

# Function that simulates uniform random variables
sim_rand_unif <- function(n, init_c=0.1){
  mod_lcg <- 134456 # modulus for linear congruential generator (random0 used)
  sim <- rep(NA, n)
  sim[1] <- floor(init_c * mod_lcg)
  for(i in 2:n) sim[i] <- (8121 * sim[i-1] + 28411) %% mod_lcg
  return(sim / mod_lcg)
}
# Function for non-linear mean
sim_friedman3=function(n, n_irrelevant=5, init_c=0.2644234){
  X <- matrix(sim_rand_unif(4*n,init_c=init_c),ncol=4)
  X[,1] <- 100*X[,1]
  X[,2] <- X[,2]*pi*(560-40)+40*pi
  X[,4] <- X[,4]*10+1
  f <- sqrt(10)*atan((X[,2]*X[,3]-1/(X[,2]*X[,4]))/X[,1])
  X <- cbind(rep(1,n),X)
  if(n_irrelevant>0) X <- cbind(X,matrix(sim_rand_unif(n_irrelevant*n,init_c=0.6543),ncol=n_irrelevant))
  return(list(X=X,f=f))
}
f1d <- function(x) 2*(1.5*(1/(1+exp(-(x-0.5)*20))+0.75*x)-1.3)
sim_non_lin_f=function(n){
  X <- matrix(sim_rand_unif(2*n,init_c=0.4596534),ncol=2)
  f <- f1d(X[,1])
  return(list(X=X,f=f))
}


# Make plot of fitted boosting ensemble ("manual test")
n <- 1000
m <- 100
sim_data <- sim_non_lin_f(n=n)
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
b1 <- qnorm(sim_rand_unif(n=m, init_c=0.3242))
eps <- b1[group]
eps <- eps - mean(eps)
probs <- pnorm(sim_data$f+eps)
y <- as.numeric(sim_rand_unif(n=n, init_c=0.6352) < probs)

nrounds <- 200
learning_rate <- 0.2
min_data_in_leaf <- 50
gp_model <- GPModel(group_data = group, likelihood = "bernoulli_probit")
bst <- gpboost(data = sim_data$X, label = y, gp_model = gp_model,
               objective = "binary", nrounds=200, learning_rate=learning_rate,
               train_gp_model_cov_pars=TRUE, min_data_in_leaf=min_data_in_leaf,verbose=0,
               metric="approx_neg_marginal_log_likelihood")
# summary(gp_model)
nplot <- 200
X_test_plot <- cbind(seq(from=0,to=1,length.out=nplot),rep(0.5,nplot))
group_data_pred <- rep(-9999,dim(X_test_plot)[1])
pred_prob <- predict(bst, data = X_test_plot, group_data_pred = group_data_pred, rawscore = FALSE)$response_mean
pred <- predict(bst, data = X_test_plot, group_data_pred = group_data_pred, rawscore = TRUE)
x <- seq(from=0,to=1,length.out=200)
plot(x,f1d(x),type="l",lwd=3,col=2,main="Data, true and fitted function")
points(sim_data$X[,1],y)
lines(X_test_plot[,1],pred$fixed_effect,col=4,lwd=3)
lines(X_test_plot[,1],pred_prob,col=3,lwd=3)
legend(legend=c("True","Pred F","Pred p"),"bottomright",bty="n",lwd=3,col=c(2,4,3))

# ## Compare to standard, independent boosting
# bst_std <- gpboost(data = sim_data$X, label = y,verbose=0,
#                objective = "binary", nrounds=200, learning_rate=learning_rate,
#                train_gp_model_cov_pars=FALSE, min_data_in_leaf=min_data_in_leaf)
# pred <- predict(bst_std, data = X_test_plot, rawscore=TRUE)
# lines(X_test_plot[,1],pred,col=5,lwd=3, lty=2)



if(R.Version()$arch != "i386"){##32-bit version is not supported by the tree-boosting part (LightGBM)
  
  print("Ignore [GPBoost] [Warning]")
  test_that("Combine tree-boosting and grouped random effects model for binary classification ", {
    
    ntrain <- ntest <- 1000
    n <- ntrain + ntest
    # Simulate fixed effects
    sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.2644234)
    f <- sim_data$f
    f <- f - mean(f)
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
    n_new <- 3# number of new random effects in test data
    group[(length(group)-n_new+1):length(group)] <- rep(99999,n_new)
    Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
    b1 <- sqrt(sigma2_1) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.5542))
    # Second random effect
    n_obs_gr <- ntrain/m# number of sampels per group
    group2 <- rep(1,ntrain) # grouping variable
    for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
    group2 <- c(group2,group2)
    group2[(length(group2)-n_new+1):length(group2)] <- rep(99999,n_new)
    Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
    b2 <- sqrt(sigma2_2) * qnorm(sim_rand_unif(n=length(unique(group2)), init_c=0.82354))
    eps <- Z1 %*% b1 + Z2 %*% b2
    eps <- eps - mean(eps)
    group_data <- cbind(group,group2)
    # Observed data
    probs <- pnorm(f + eps)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.574) < probs)
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
    # Data for Booster
    dtrain <- gpb.Dataset(data = X_train, label = y_train)
    dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
    valids <- list(test = dtest)
    params <- list(learning_rate = 0.1, objective = "binary")
    # Folds for CV
    group_aux <- rep(1,ntrain) # grouping variable
    for(i in 1:(ntrain/4)) group_aux[(1:4)+4*(i-1)] <- 1:4
    folds <- list()
    for(i in 1:4) folds[[i]] <- as.integer(which(group_aux==i))
    
    # Label needs to have correct format
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(maxit=2, optimizer_cov="gradient_descent"))
    expect_error(gpboost(data = X_train, label = probs[1:ntrain], gp_model = gp_model,
                         objective = "binary", nrounds=1))
    # Only gradient descent can be used
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(maxit=2, optimizer_cov="fisher_scoring"))
    expect_error(gpboost(data = X_train, label = y_train, gp_model = gp_model,
                         objective = "binary", verbose=0, nrounds=1))
    # Prediction data needs to be set when use_gp_model_for_validation=TRUE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    expect_error(gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                         objective = "binary", train_gp_model_cov_pars=FALSE, nrounds=1, valids=valids))
    
    # Validation metrics for training data
    # Default metric is "Approx. negative marginal log-likelihood" if there is only one training set
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                   objective = "binary", train_gp_model_cov_pars=FALSE, nrounds=1)
    record_results <- gpb.get.eval.result(bst, "train", "Approx. negative marginal log-likelihood")
    expect_lt(abs(record_results[1]-599.795), 1e-3)
    bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                   objective = "regression", train_gp_model_cov_pars=FALSE, nrounds=1)
    record_results <- gpb.get.eval.result(bst, "train", "Approx. negative marginal log-likelihood")
    expect_lt(abs(record_results[1]-599.795), 1e-3)
    # Can also use other metrics
    bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                   objective = "binary", train_gp_model_cov_pars=FALSE, nrounds=1,
                   eval=list("binary_logloss","binary_error"), use_gp_model_for_validation = FALSE)
    record_results <- gpb.get.eval.result(bst, "train", "binary_logloss")
    expect_lt(abs(record_results[1]-0.674702), 1e-3)
    record_results <- gpb.get.eval.result(bst, "train", "binary_error")
    expect_lt(abs(record_results[1]-0.466), 1e-3)
    bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                   objective = "regression", train_gp_model_cov_pars=FALSE, nrounds=1,
                   eval=list("l2","binary_error"), use_gp_model_for_validation = FALSE)
    record_results <- gpb.get.eval.result(bst, "train", "l2")
    expect_lt(abs(record_results[1]-0.240814), 1e-3)
    record_results <- gpb.get.eval.result(bst, "train", "binary_error")
    expect_lt(abs(record_results[1]-0.466), 1e-3)
    
    # Find number of iterations using validation data with use_gp_model_for_validation=FALSE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.1, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=FALSE, eval = "binary_error",
                     early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "binary_error")
    expect_lt(abs(min(record_results)-0.326), 1e-6)
    expect_equal(which.min(record_results), 8)
    
    # Find number of iterations using validation data with use_gp_model_for_validation=TRUE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_prediction_data(group_data_pred = group_data_test)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.1, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "binary_error",
                     early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "binary_error")
    expect_lt(abs(min(record_results)-0.239), 1e-6)
    expect_equal(which.min(record_results), 21)
    # Compare to when ignoring random effects part
    bst <- gpb.train(data = dtrain, nrounds=100, valids=valids,
                     learning_rate=0.1, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "binary_error", early_stopping_rounds=10)
    expect_lt(abs(bst$best_score-.345), 1e-6)
    
    # Other metrics / losses
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_prediction_data(group_data_pred = group_data_test)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.5, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "binary_logloss",
                     early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "binary_logloss")
    expect_lt(abs(min(record_results)-0.4917417), 1e-5)
    expect_equal(which.min(record_results), 6)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.5, objective = "regression", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "l2", early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "l2")
    expect_lt(abs(min(record_results)-0.1643653), 1e-4)
    expect_equal(which.min(record_results), 6)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.5, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "l2", early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "l2")
    expect_lt(abs(min(record_results)-0.1643653), 1e-4)
    expect_equal(which.min(record_results), 6)
    
    # CV for finding number of boosting iterations when use_gp_model_for_validation = FALSE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    cvbst <- gpb.cv(params = params,
                    data = dtrain,
                    gp_model=gp_model,
                    nrounds = 100,
                    nfold = 4,
                    eval = "binary_error",
                    early_stopping_rounds = 5,
                    use_gp_model_for_validation = FALSE,
                    fit_GP_cov_pars_OOS = FALSE,
                    folds = folds,
                    verbose = 0)
    expect_equal(cvbst$best_iter, 5)
    expect_lt(abs(cvbst$best_score-0.353), 1E-4)
    # same thing but "wrong" likelihood given in gp_model
    gp_model <- GPModel(group_data = group_data_train)
    cvbst <- gpb.cv(params = params,
                    data = dtrain,
                    gp_model=gp_model,
                    nrounds = 100,
                    nfold = 4,
                    eval = "binary_error",
                    early_stopping_rounds = 5,
                    use_gp_model_for_validation = FALSE,
                    fit_GP_cov_pars_OOS = FALSE,
                    folds = folds,
                    verbose = 0)
    expect_equal(cvbst$best_iter, 5)
    expect_lt(abs(cvbst$best_score-0.353), 1E-4)
    # CV for finding number of boosting iterations when use_gp_model_for_validation = TRUE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(use_nesterov_acc=FALSE))
    cvbst <- gpb.cv(params = params,
                    data = dtrain,
                    gp_model = gp_model,
                    nrounds = 100,
                    nfold = 4,
                    eval = "binary_error",
                    early_stopping_rounds = 5,
                    use_gp_model_for_validation = TRUE,
                    fit_GP_cov_pars_OOS = FALSE,
                    folds = folds,
                    verbose = 0)
    expect_equal(cvbst$best_iter, 30)
    expect_lt(abs(cvbst$best_score-0.243), 1E-4)
    
    # Create random effects model and train GPBoost model
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(use_nesterov_acc=FALSE))
    bst <- gpboost(data = X_train,
                   label = y_train,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "binary",
                   verbose = 0)
    cov_pars <- c(0.4581991, 0.3429415)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-6)

    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = TRUE)
    expect_lt(sqrt(mean((pred$fixed_effect - f_test)^2)),0.663)
    expect_lt(sum(abs(head(pred$fixed_effect)-c(0.4791272, -0.1744148, 0.9284747,
                                                0.8183078, -0.6986425, 0.8669949))),1E-6)
    expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-1.092220, -1.024815, -1.203819,
                                                      rep(0,n_new)))),1E-6)
    expect_lt(sum(abs(head(pred$random_effect_mean)-c(1.5492283, 0.1948716, 0.6561371,
                                                      1.2325282, 0.2249754, 0.5894416))),1E-6)
    expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.1292745, 0.1284435, 0.1291609,
                                                      rep(0.8011406,n_new)))),1E-6)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = FALSE)
    expect_equal(mean(as.numeric(pred$response_mean>0.5) != y_test),0.229)
    expect_lt(sum(abs(tail(pred$response_var)-c(0.146077725, 0.179428082, 0.012855503,
                                                0.230808713, 0.161401483, 0.241088278))),1E-6)
    
    # Prediction when having only one grouped random effect
    group_1 <- rep(1,ntrain) # grouping variable
    for(i in 1:m) group_1[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
    probs_1 <- pnorm(f[1:ntrain] + b1[group_1])
    y_1 <- as.numeric(sim_rand_unif(n=ntrain, init_c=0.574) < probs_1)
    gp_model <- GPModel(group_data = group_1, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(use_nesterov_acc=FALSE))
    bst <- gpboost(data = X_train,
                   label = y_1,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "binary",
                   verbose = 0,
                   leaves_newton_update = FALSE)
    pred <- predict(bst, data = X_test[1:length(unique(b1)),], group_data_pred = 1:length(unique(b1)), rawscore = TRUE)
    # plot(pred$random_effect_mean,b1)
    expect_lt(abs(sqrt(sum((pred$random_effect_mean - b1)^2))-1.665886),1E-6)
    expect_lt(abs(cor(pred$random_effect_mean,b1)-0.942044),1E-6)
    # Prediction for only new groups
    group_test <- c(-1,-1,-2,-2)
    pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test, rawscore = TRUE)
    fix_eff <- c(0.2179516, 0.3181951, 0.6611486, 0.4956463)
    expect_lt(sum(abs(pred$fixed_effect-fix_eff)),1E-6)
    expect_lt(sum(abs(pred$random_effect_mean-rep(0,4))),1E-6)
    pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test, rawscore = FALSE)
    resp <- c(0.5702958, 0.6020259, 0.7044731, 0.6564551)
    expect_lt(sum(abs(pred$response_mean-resp)),1E-6)
    # Prediction for only new cluster_ids
    cluster_ids_pred <- c(-1,-1,-2,-2)
    group_test <- c(1,3,3,9999)
    pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test,
                    cluster_ids_pred = cluster_ids_pred, rawscore = TRUE)
    expect_lt(sum(abs(pred$random_effect_mean-rep(0,4))),1E-6)
    expect_lt(sum(abs(pred$fixed_effect-fix_eff)),1E-6)
    pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test,
                    cluster_ids_pred = cluster_ids_pred, rawscore = FALSE)
    expect_lt(sum(abs(pred$response_mean-resp)),1E-6)
    
    # Train tree-boosting model while holding the GPModel fix
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    bst <- gpboost(data = X_train,
                   label = y_train,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "binary",
                   train_gp_model_cov_pars = FALSE,
                   verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(1, 1))),1E-6)
    # GPBoostOOS algorithm: fit parameters on out-of-sample data
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(use_nesterov_acc=FALSE))
    cvbst <- gpb.cv(params = params,
                    data = dtrain,
                    gp_model = gp_model,
                    nrounds = 100,
                    nfold = 4,
                    eval = "binary_error",
                    early_stopping_rounds = 5,
                    use_gp_model_for_validation = TRUE,
                    fit_GP_cov_pars_OOS = TRUE,
                    folds = folds,
                    verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.4786203, 0.3436059))),1E-3)
    expect_equal(cvbst$best_iter, 30)
    expect_lt(abs(cvbst$best_score-0.243), 1E-4)
    
    # Use of validation data and cross-validation with custom metric
    bin_cust_error <- function(preds, dtrain) {
      labels <- getinfo(dtrain, "label")
      predsbin <- preds > 0.55
      error <- mean(predsbin!=labels)#mean((preds-labels)^4)
      return(list(name="bin_cust_error",value=error,higher_better=FALSE))
    }
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.1, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=FALSE,
                     early_stopping_rounds=10, eval = bin_cust_error, metric = "bin_cust_error")
    expect_equal(bst$best_iter, 27)
    expect_lt(abs(bst$best_score - 0.356),1E-6)
    # CV
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    cvbst <- gpb.cv(params = params,
                    data = dtrain,
                    gp_model=gp_model,
                    nrounds = 100,
                    nfold = 4,
                    early_stopping_rounds = 5,
                    use_gp_model_for_validation = FALSE,
                    fit_GP_cov_pars_OOS = FALSE,
                    folds = folds,
                    verbose = 0,
                    eval = bin_cust_error, metric = "bin_cust_error")
    expect_equal(cvbst$best_iter, 21)
    expect_lt(abs(cvbst$best_score-0.356), 1E-4)
  })
  
  print("Ignore [GPBoost] [Warning]")
  test_that("Combine tree-boosting and Gaussian process model for binary classification ", {
    
    ntrain <- ntest <- 500
    n <- ntrain + ntest
    # Simulate fixed effects
    sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.69)
    f <- sim_data$f
    f <- f - mean(f)
    X <- sim_data$X
    # Simulate spatial Gaussian process
    sigma2_1 <- 1 # marginal variance of GP
    rho <- 0.1 # range parameter
    d <- 2 # dimension of GP locations
    coords <- matrix(sim_rand_unif(n=n*d, init_c=0.63), ncol=d)
    D <- as.matrix(dist(coords))
    Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
    C <- t(chol(Sigma))
    b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.987864))
    eps <- as.vector(C %*% b_1)
    # Observed data
    probs <- pnorm(f + eps)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.52574) < probs)
    # Split into training and test data
    y_train <- y[1:ntrain]
    X_train <- X[1:ntrain,]
    coords_train <- coords[1:ntrain,]
    dtrain <- gpb.Dataset(data = X_train, label = y_train)
    y_test <- y[1:ntest+ntrain]
    X_test <- X[1:ntest+ntrain,]
    f_test <- f[1:ntest+ntrain]
    coords_test <- coords[1:ntest+ntrain,]
    eps_test <- eps[1:ntest+ntrain]

    # Train model
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(maxit=10))
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 2,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    cov_pars_est <- c(0.3049804, 0.1050220)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),1E-6)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, rawscore = TRUE)
    expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-0.7347603),1E-6)
    expect_lt(abs(sqrt(mean((pred$random_effect_mean - eps_test)^2))-0.8476357),1E-6)
    # Predict response
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, rawscore = FALSE)
    expect_equal(mean(as.numeric(pred$response_mean>0.5) != y_test),0.306)
    
    # Use validation set to determine number of boosting iteration with use_gp_model_for_validation = FALSE
    dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
    valids <- list(test = dtest)
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(maxit=10))
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 10,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0,
                     valids = valids,
                     early_stopping_rounds = 2,
                     use_gp_model_for_validation = FALSE)
    expect_equal(bst$best_iter, 2)
    expect_lt(abs(bst$best_score - 0.6368231),1E-6)
    
    # Also use GPModel for calculating validation error
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(maxit=20, use_nesterov_acc=FALSE))
    gp_model$set_prediction_data(gp_coords_pred = coords_test)
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 10,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0,
                     valids = valids,
                     early_stopping_rounds = 2,
                     use_gp_model_for_validation = TRUE)
    expect_equal(bst$best_iter, 4)
    expect_lt(abs(bst$best_score - 0.5901579),1E-6)
  })
  
  print("Ignore [GPBoost] [Warning]")
  test_that("Combine tree-boosting and Gaussian process model with Vecchia approximation for binary classification", {
    
    ntrain <- ntest <- 100
    n <- ntrain + ntest
    # Simulate fixed effects
    sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.69)
    f <- sim_data$f
    f <- f - mean(f)
    X <- sim_data$X
    # Simulate grouped random effects
    sigma2_1 <- 1 # marginal variance of GP
    rho <- 0.1 # range parameter
    d <- 2 # dimension of GP locations
    coords <- matrix(sim_rand_unif(n=n*d, init_c=0.63), ncol=d)
    D <- as.matrix(dist(coords))
    Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
    C <- t(chol(Sigma))
    b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.987864))
    eps <- as.vector(C %*% b_1)
    # Observed data
    probs <- pnorm(f + eps)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.52574) < probs)
    # Split in training and test data
    y_train <- y[1:ntrain]
    X_train <- X[1:ntrain,]
    coords_train <- coords[1:ntrain,]
    dtrain <- gpb.Dataset(data = X_train, label = y_train)
    y_test <- y[1:ntest+ntrain]
    X_test <- X[1:ntest+ntrain,]
    f_test <- f[1:ntest+ntrain]
    coords_test <- coords[1:ntest+ntrain,]
    eps_test <- eps[1:ntest+ntrain]
    
    # Train model
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(maxit=20, lr_cov=0.1, use_nesterov_acc=FALSE))
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 5,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    cov_pars_est <- c(0.009769709, 0.102480262)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),1E-6)
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, rawscore = TRUE)
    expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-0.9995755),1E-5)
    expect_lt(abs(sqrt(mean((pred$random_effect_mean - eps_test)^2))-1.08882),1E-5)
    # Same thing with Vecchia approximation
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_probit", vecchia_approx =TRUE, num_neighbors = ntrain-1)
    gp_model$set_optim_params(params=list(maxit=20, lr_cov=0.1, use_nesterov_acc=FALSE))
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 5,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),1E-6)
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, rawscore = TRUE)
    expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-0.9995755),1E-6)
    expect_lt(abs(sqrt(mean((pred$random_effect_mean - eps_test)^2))-1.08882),1E-6)
  })
  
  
  test_that("Combine tree-boosting and Gaussian process model for binary classification with logit link", {
    
    ntrain <- ntest <- 500
    n <- ntrain + ntest
    # Simulate fixed effects
    sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.69)
    f <- sim_data$f
    f <- f - mean(f)
    X <- sim_data$X
    # Simulate spatial Gaussian process
    sigma2_1 <- 1 # marginal variance of GP
    rho <- 0.1 # range parameter
    d <- 2 # dimension of GP locations
    coords <- matrix(sim_rand_unif(n=n*d, init_c=0.63), ncol=d)
    D <- as.matrix(dist(coords))
    Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
    C <- t(chol(Sigma))
    b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.987864))
    eps <- as.vector(C %*% b_1)
    # Observed data
    probs <- 1/(1+exp(-(f+eps)))
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.52574) < probs)
    # Split into training and test data
    y_train <- y[1:ntrain]
    X_train <- X[1:ntrain,]
    coords_train <- coords[1:ntrain,]
    dtrain <- gpb.Dataset(data = X_train, label = y_train)
    y_test <- y[1:ntest+ntrain]
    X_test <- X[1:ntest+ntrain,]
    f_test <- f[1:ntest+ntrain]
    coords_test <- coords[1:ntest+ntrain,]
    eps_test <- eps[1:ntest+ntrain]
    
    # Train model
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_logit")
    gp_model$set_optim_params(params=list(maxit=10))
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 2,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    cov_pars_est <- c(0.4529232, 0.1110128)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),1E-6)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    predict_var = TRUE, rawscore = TRUE)
    expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-0.7699441),1E-6)
    expect_lt(abs(sqrt(mean((pred$random_effect_mean - eps_test)^2))-0.8463743),1E-6)
    expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.3243953, 0.2990569, 0.3091693,
                                                     0.3093641, 0.2914468, 0.2957911))),1E-6)
    # Predict response
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, 
                    predict_var = TRUE, rawscore = FALSE)
    expect_equal(mean(as.numeric(pred$response_mean>0.5) != y_test),0.374)
    expect_lt(sum(abs(tail(pred$response_var)-c(0.2318384, 0.2178964, 0.2399617,
                                                0.2494777, 0.2060365, 0.2499535))),1E-6)
    
  })
  
  test_that("Combine tree-boosting and random effects for Poisson regression", {
    
    ntrain <- ntest <- 1000
    n <- ntrain + ntest
    # Simulate fixed effects
    sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.2644234)
    f <- sim_data$f
    f <- f - mean(f)
    X <- sim_data$X
    # Simulate grouped random effects
    sigma2_1 <- 0.6 # variance of first random effect
    sigma2_2 <- 0.4 # variance of second random effect
    m <- 40 # number of categories / levels for grouping variable
    # first random effect
    group <- rep(1,ntrain) # grouping variable
    for(i in 1:m) group[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
    group <- c(group, group)
    n_new <- 3# number of new random effects in test data
    group[(length(group)-n_new+1):length(group)] <- rep(99999,n_new)
    Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
    b1 <- sqrt(sigma2_1) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.5542))
    # Second random effect
    n_obs_gr <- ntrain/m# number of sampels per group
    group2 <- rep(1,ntrain) # grouping variable
    for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
    group2 <- c(group2,group2)
    group2[(length(group2)-n_new+1):length(group2)] <- rep(99999,n_new)
    Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
    b2 <- sqrt(sigma2_2) * qnorm(sim_rand_unif(n=length(unique(group2)), init_c=0.82354))
    eps <- Z1 %*% b1 + Z2 %*% b2
    eps <- eps - mean(eps)
    group_data <- cbind(group,group2)
    # Observed data
    mu <- exp(f + eps)
    y <- qpois(sim_rand_unif(n=n, init_c=0.04532), lambda = mu)
    # Split in training and test data
    y_train <- y[1:ntrain]
    X_train <- X[1:ntrain,]
    group_data_train <- group_data[1:ntrain,]
    y_test <- y[1:ntest+ntrain]
    X_test <- X[1:ntest+ntrain,]
    f_test <- f[1:ntest+ntrain]
    group_data_test <- group_data[1:ntest+ntrain,]
    eps_test <- eps[1:ntest+ntrain]
    # Data for Booster
    dtrain <- gpb.Dataset(data = X_train, label = y_train)
    dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
    
    # Train model
    gp_model <- GPModel(group_data = group_data_train, likelihood = "poisson")
    bst <- gpboost(data = dtrain,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "poisson",
                   verbose = 0)
    cov_pars_est <- c(0.5459914, 0.3763641)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),1E-6)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = TRUE)
    expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-0.5600951),1E-3)
    expect_lt(abs(sqrt(mean((pred$random_effect_mean - eps_test)^2))-0.2989241),1E-6)
    expect_lt(sum(abs(tail(pred$fixed_effect)-c(0.2741840, 0.2159258, -2.2611556,
                                                0.7262528, -1.1725225, 0.4201358))),1E-6)
    expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-0.7801564, -0.7557558, -0.8661279, rep(0,3)))),1E-6)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = FALSE)
    expect_lt(abs(sqrt(mean((pred$response_mean - y_test)^2))-1.838055),1E-4)
    expect_lt(sum(abs(tail(pred$response_mean)-c(0.6236602, 0.6022954, 0.0454444,
                                                 3.2786463, 0.4909835, 2.4140687))),1E-6)
    expect_lt(sum(abs(tail(pred$response_var)-c(0.65088143, 0.62690803, 0.04559866,
                                                19.56640952, 0.85624692, 11.24428895))),1E-6)
  })
  
  test_that("Combine tree-boosting and random effects for gamma regression", {
    
    ntrain <- ntest <- 1000
    n <- ntrain + ntest
    # Simulate fixed effects
    sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.2644234)
    f <- sim_data$f
    f <- f - mean(f)
    X <- sim_data$X
    # Simulate grouped random effects
    sigma2_1 <- 0.6 # variance of first random effect
    sigma2_2 <- 0.4 # variance of second random effect
    m <- 40 # number of categories / levels for grouping variable
    # first random effect
    group <- rep(1,ntrain) # grouping variable
    for(i in 1:m) group[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
    group <- c(group, group)
    n_new <- 3# number of new random effects in test data
    group[(length(group)-n_new+1):length(group)] <- rep(99999,n_new)
    Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
    b1 <- sqrt(sigma2_1) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.5542))
    # Second random effect
    n_obs_gr <- ntrain/m# number of sampels per group
    group2 <- rep(1,ntrain) # grouping variable
    for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
    group2 <- c(group2,group2)
    group2[(length(group2)-n_new+1):length(group2)] <- rep(99999,n_new)
    Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
    b2 <- sqrt(sigma2_2) * qnorm(sim_rand_unif(n=length(unique(group2)), init_c=0.82354))
    eps <- Z1 %*% b1 + Z2 %*% b2
    eps <- eps - mean(eps)
    group_data <- cbind(group,group2)
    # Observed data
    mu <- exp(f + eps)
    shape <- 1
    y <- qgamma(sim_rand_unif(n=n, init_c=0.652), scale = mu/shape, shape = shape)
    # Split in training and test data
    y_train <- y[1:ntrain]
    X_train <- X[1:ntrain,]
    group_data_train <- group_data[1:ntrain,]
    y_test <- y[1:ntest+ntrain]
    X_test <- X[1:ntest+ntrain,]
    f_test <- f[1:ntest+ntrain]
    group_data_test <- group_data[1:ntest+ntrain,]
    eps_test <- eps[1:ntest+ntrain]
    # Data for Booster
    dtrain <- gpb.Dataset(data = X_train, label = y_train)
    dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
    
    # Train model
    gp_model <- GPModel(group_data = group_data_train, likelihood = "gamma")
    bst <- gpboost(data = dtrain,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "gamma",
                   verbose = 0)
    cov_pars_est <- c(0.5537789, 0.4654065)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),1E-6)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = TRUE)
    expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-0.4737707),1E-6)
    expect_lt(abs(sqrt(mean((pred$random_effect_mean - eps_test)^2))-0.347549),1E-6)
    expect_lt(sum(abs(tail(pred$fixed_effect)-c( 0.8115548, 0.1776281, -1.5159576,
                                                 0.4539956, -1.2554634, 0.3017937))),1E-6)
    expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-1.309823, -1.003674, -1.420617, rep(0,3)))),1E-6)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = FALSE)
    expect_lt(abs(sqrt(mean((pred$response_mean - y_test)^2))-3.121299),1E-6)
    expect_lt(sum(abs(tail(pred$response_mean)-c(0.62762983, 0.45201814, 0.05481417,
                                                 2.62108499, 0.47432113, 2.25102673))),1E-6)
    expect_lt(sum(abs(tail(pred$response_var)-c(0.446767916, 0.231339865, 0.003411605,
                                                31.203064977, 1.021833158, 23.014225038))),1E-6)
    
  })
  
}

