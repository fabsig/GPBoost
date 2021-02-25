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



# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
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
    expect_value <- 599.8544
    expect_lt(abs(record_results[1]-expect_value), 1e-3)
    bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                   objective = "regression", train_gp_model_cov_pars=FALSE, nrounds=1)
    record_results <- gpb.get.eval.result(bst, "train", "Approx. negative marginal log-likelihood")
    expect_lt(abs(record_results[1]-expect_value), 1e-3)
    # Can also use other metrics
    bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                   objective = "binary", train_gp_model_cov_pars=FALSE, nrounds=1,
                   eval=list("binary_logloss","binary_error"), use_gp_model_for_validation = FALSE)
    record_results <- gpb.get.eval.result(bst, "train", "binary_logloss")
    expect_lt(abs(record_results[1]-0.6796555), 1e-3)
    record_results <- gpb.get.eval.result(bst, "train", "binary_error")
    expect_lt(abs(record_results[1]-0.466), 1e-3)
    bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                   objective = "regression", train_gp_model_cov_pars=FALSE, nrounds=1,
                   eval=list("l2","binary_error"), use_gp_model_for_validation = FALSE)
    record_results <- gpb.get.eval.result(bst, "train", "l2")
    expect_lt(abs(record_results[1]-0.2434136), 1e-3)
    record_results <- gpb.get.eval.result(bst, "train", "binary_error")
    expect_lt(abs(record_results[1]-0.466), 1e-3)
    
    # Find number of iterations using validation data with use_gp_model_for_validation=FALSE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(lr_cov=0.01))
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.1, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=FALSE, eval = "binary_error",
                     early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "binary_error")
    expect_lt(abs(min(record_results)-0.33), 1e-3)
    expect_equal(which.min(record_results), 13)
    
    # Find number of iterations using validation data with use_gp_model_for_validation=TRUE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(lr_cov=0.01))
    gp_model$set_prediction_data(group_data_pred = group_data_test)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.1, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "binary_error",
                     early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "binary_error")
    expect_lt(abs(min(record_results)-0.242), 1e-3)
    expect_equal(which.min(record_results), 17)
    # Compare to when ignoring random effects part
    bst <- gpb.train(data = dtrain, nrounds=100, valids=valids,
                     learning_rate=0.1, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "binary_error", early_stopping_rounds=10)
    expect_lt(abs(bst$best_score-.345), 1e-3)
    
    # Other metrics / losses
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(lr_cov=0.01))
    gp_model$set_prediction_data(group_data_pred = group_data_test)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.5, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "binary_logloss",
                     early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "binary_logloss")
    expect_lt(abs(min(record_results)-0.4909814), 1e-3)
    expect_equal(which.min(record_results), 7)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.5, objective = "regression", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "l2", early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "l2")
    expect_lt(abs(min(record_results)-0.1643213), 1e-3)
    expect_equal(which.min(record_results), 7)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.5, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "l2", early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "l2")
    expect_lt(abs(min(record_results)-0.1643213), 1e-3)
    expect_equal(which.min(record_results), 7)
    
    # CV for finding number of boosting iterations when use_gp_model_for_validation = FALSE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(lr_cov=0.01))
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
    expect_equal(cvbst$best_iter, 15)
    expect_lt(abs(cvbst$best_score-0.352), 1E-3)
    # same thing but "wrong" likelihood given in gp_model
    gp_model <- GPModel(group_data = group_data_train, likelihood="gaussian")
    gp_model$set_optim_params(params=list(lr_cov=0.01))
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
    expect_equal(cvbst$best_iter, 15)
    expect_lt(abs(cvbst$best_score-0.352), 1E-3)
    # CV for finding number of boosting iterations when use_gp_model_for_validation = TRUE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(use_nesterov_acc=FALSE, lr_cov=0.01))
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
    expect_equal(cvbst$best_iter, 27)
    expect_lt(abs(cvbst$best_score-0.24), 1E-3)
    
    # Create random effects model and train GPBoost model
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(use_nesterov_acc=FALSE, lr_cov=0.01))
    bst <- gpboost(data = X_train,
                   label = y_train,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "binary",
                   verbose = 0)
    cov_pars <- c(0.4624550, 0.3472107)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),1E-3)
    
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = TRUE)
    expect_lt(sum(abs(head(pred$fixed_effect)-c(0.61920543, -0.04315145, 1.08458970,
                                                0.94743365, -0.55021366, 0.92315482))),1E-3)
    expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-1.204733, -1.151827, -1.325256,
                                                      rep(0,n_new)))),1E-3)
    expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.1295074, 0.1286363, 0.1290317,
                                                     rep(0.8096657,n_new)))),1E-3)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = FALSE)
    expect_equal(mean(as.numeric(pred$response_mean>0.5) != y_test),0.24)
    expect_lt(sum(abs(tail(pred$response_var)-c(0.13912283, 0.17045664, 0.01650304, 0.22656233, 0.17169528, 0.22360806))),1E-3)
    
    # Prediction when having only one grouped random effect
    group_1 <- rep(1,ntrain) # grouping variable
    for(i in 1:m) group_1[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
    probs_1 <- pnorm(f[1:ntrain] + b1[group_1])
    y_1 <- as.numeric(sim_rand_unif(n=ntrain, init_c=0.574) < probs_1)
    gp_model <- GPModel(group_data = group_1, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(use_nesterov_acc=FALSE, lr_cov=0.01))
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
    expect_lt(abs(sqrt(sum((pred$random_effect_mean - b1)^2))-1.690961),1E-3)
    # Prediction for only new groups
    group_test <- c(-1,-1,-2,-2)
    pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test, rawscore = TRUE)
    fix_eff <- c(0.2561277, 0.5479570, 0.6756243, 0.5795295)
    expect_lt(sum(abs(pred$fixed_effect-fix_eff)),1E-3)
    expect_lt(sum(abs(pred$random_effect_mean-rep(0,4))),1E-3)
    pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test, rawscore = FALSE)
    resp <- c(0.5823307, 0.6717290, 0.7082572, 0.6809350)
    expect_lt(sum(abs(pred$response_mean-resp)),1E-3)
    # Prediction for only new cluster_ids
    cluster_ids_pred <- c(-1L,-1L,-2L,-2L)
    group_test <- c(1,3,3,9999)
    pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test,
                    cluster_ids_pred = cluster_ids_pred, rawscore = TRUE)
    expect_lt(sum(abs(pred$random_effect_mean-rep(0,4))),1E-3)
    expect_lt(sum(abs(pred$fixed_effect-fix_eff)),1E-3)
    pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test,
                    cluster_ids_pred = cluster_ids_pred, rawscore = FALSE)
    expect_lt(sum(abs(pred$response_mean-resp)),1E-3)
    
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(1, 1))),1E-3)
    # GPBoostOOS algorithm: fit parameters on out-of-sample data
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(use_nesterov_acc=FALSE, lr_cov=0.01))
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.4680810, 0.3371244))),1E-3)
    expect_equal(cvbst$best_iter, 27)
    expect_lt(abs(cvbst$best_score-0.24), 1E-3)
    
    # Use of validation data and cross-validation with custom metric
    bin_cust_error <- function(preds, dtrain) {
      labels <- getinfo(dtrain, "label")
      predsbin <- preds > 0.55
      error <- mean(predsbin!=labels)#mean((preds-labels)^4)
      return(list(name="bin_cust_error",value=error,higher_better=FALSE))
    }
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(lr_cov=0.01))
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.1, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=FALSE,
                     early_stopping_rounds=10, eval = bin_cust_error, metric = "bin_cust_error")
    expect_equal(bst$best_iter, 8)
    expect_lt(abs(bst$best_score - 0.326),1E-3)
    # CV
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(lr_cov=0.01))
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
    expect_equal(cvbst$best_iter, 5)
    expect_lt(abs(cvbst$best_score-0.357), 1E-3)
  })
  
  
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
    gp_model$set_optim_params(params=list(maxit=10, lr_cov=0.01))
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 2,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    cov_pars_est <- c(0.32547158, 0.09894853)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),1E-3)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, rawscore = TRUE)
    expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-0.8028029),1E-3)
    expect_lt(abs(sqrt(mean((pred$random_effect_mean - eps_test)^2))-0.8899398),1E-3)
    # Predict response
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, rawscore = FALSE)
    expect_equal(mean(as.numeric(pred$response_mean>0.5) != y_test),0.302)
    
    # Use validation set to determine number of boosting iteration with use_gp_model_for_validation = FALSE
    dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
    valids <- list(test = dtest)
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(maxit=10, lr_cov=0.01))
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
    expect_equal(bst$best_iter, 3)
    expect_lt(abs(bst$best_score - 0.6386544),1E-3)
    
    # Also use GPModel for calculating validation error
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(maxit=20, use_nesterov_acc=TRUE, lr_cov=0.1))
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
    expect_lt(abs(bst$best_score - 0.5963344),1E-3)
  })
  
  
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
    gp_model$set_optim_params(params=list(maxit=20, lr_cov=0.01, use_nesterov_acc=FALSE))
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 5,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    cov_pars_est <- c(0.2768962, 0.4783053)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),1E-3)
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, rawscore = TRUE)
    expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-1.376468),1E-3)
    expect_lt(abs(sqrt(mean((pred$random_effect_mean - eps_test)^2))-1.295551),1E-3)
    # Same thing with Vecchia approximation
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_probit", vecchia_approx =TRUE, num_neighbors = ntrain-1)
    gp_model$set_optim_params(params=list(maxit=20, lr_cov=0.01, use_nesterov_acc=FALSE))
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 5,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),1E-3)
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, rawscore = TRUE)
    expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-1.376468),1E-2)
    expect_lt(abs(sqrt(mean((pred$random_effect_mean - eps_test)^2))-1.295551),1E-3)
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
    gp_model$set_optim_params(params=list(maxit=10, lr_cov=0.01))
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 2,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    cov_pars_est <- c(0.41970989, 0.07741939)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),1E-3)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    predict_var = TRUE, rawscore = TRUE)
    expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-0.831294),1E-3)
    expect_lt(abs(sqrt(mean((pred$random_effect_mean - eps_test)^2))-0.9268517),1E-3)
    expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.3494496, 0.3252931, 0.3400189, 0.3231344, 0.3158659, 0.3251662))),1E-3)
    # Predict response
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, 
                    predict_var = TRUE, rawscore = FALSE)
    expect_equal(mean(as.numeric(pred$response_mean>0.5) != y_test),0.358)
    expect_lt(sum(abs(tail(pred$response_var)-c(0.2273978, 0.2134153, 0.2359810, 0.2498721, 0.2028934, 0.2495832))),1E-3)
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
    
    # Train model
    gp_model <- GPModel(group_data = group_data_train, likelihood = "poisson")
    gp_model$set_optim_params(params=list(lr_cov=0.01))
    bst <- gpboost(data = dtrain,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "poisson",
                   verbose = 0)
    cov_pars_est <- c(0.551014, 0.396748)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),1E-3)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = TRUE)
    expect_lt(sum(abs(tail(pred$fixed_effect)-c( 0.5762045, 0.4024273, -1.7571674,
                                                 0.9201089, -0.7181154, 0.7605264))),1E-3)
    expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-1.166206, -1.077694, -1.248224, rep(0,3)))),1E-3)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean)-c(0.57327886, 0.52581855, 0.05131457, 4.03090085, 0.78330399, 3.43634136))),1E-3)
    expect_lt(sum(abs(tail(pred$response_var)-c(0.59613634, 0.54436787, 0.05150895, 29.70184658, 1.75269350, 22.09284402))),1E-3)
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
    
    # Train model
    gp_model <- GPModel(group_data = group_data_train, likelihood = "gamma")
    gp_model$set_optim_params(params=list(lr_cov=0.01))
    bst <- gpboost(data = dtrain,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "gamma",
                   verbose = 0)
    cov_pars_est <- c(0.6270329, 0.577390)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),1E-3)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = TRUE)
    expect_lt(sum(abs(tail(pred$fixed_effect)-c(1.2900949, 0.6858358, -1.2370813, 1.0135403, -0.7627855, 0.6990146))),1E-3)
    expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-1.765800, -1.488995, -1.841810, rep(0,3)))),1E-3)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean)-c(0.64226518, 0.46276689, 0.04756658, 5.03166989, 0.85165486, 3.67379858))),1E-3)
    expect_lt(sum(abs(tail(pred$response_var)-c(4.687067e-01, 2.430343e-01, 2.573891e-03, 1.435430e+02, 4.112302e+00, 7.652238e+01))),1E-3)
  })
  

  test_that("Saving and loading a booster with a gp_model for non-Gaussian data ", {
    
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
    # Split in training and test data
    y_train <- y[1:ntrain]
    X_train <- X[1:ntrain,]
    group_data_train <- group_data[1:ntrain,]
    y_test <- y[1:ntest+ntrain]
    X_test <- X[1:ntest+ntrain,]
    f_test <- f[1:ntest+ntrain]
    group_data_test <- group_data[1:ntest+ntrain,]
    
    # Train model and make predictions
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(use_nesterov_acc=FALSE, lr_cov=0.01))
    bst <- gpboost(data = X_train,
                   label = y_train,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "binary",
                   verbose = 0)
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = TRUE)
    # Predict response
    pred_resp <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, rawscore = FALSE)
    # Save to file
    filename <- tempfile(fileext = ".model")
    gpb.save(bst, filename=filename)
    
    # finalize and destroy models
    bst$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)
    gp_model$finalize()
    expect_null(gp_model$.__enclos_env__$private$handle)
    rm(gp_model)
    
    # Load from file and make predictions again
    bst_loaded <- gpb.load(filename = filename)
    pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                           predict_var= TRUE, rawscore = TRUE)
    pred_resp_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                           predict_var= TRUE, rawscore = FALSE)
    expect_equal(pred$fixed_effect, pred_loaded$fixed_effect)
    expect_equal(pred$random_effect_mean, pred_loaded$random_effect_mean)
    expect_equal(pred$random_effect_cov, pred_loaded$random_effect_cov)
    expect_equal(pred_resp$response_mean, pred_resp_loaded$response_mean)
    expect_equal(pred_resp$response_var, pred_resp_loaded$response_var)
    
  })
  
}