context("generalized_GPBoost_combined_boosting_GP_random_effects")

TOLERANCE <- 1E-3
DEFAULT_OPTIM_PARAMS <- list(optimizer_cov="gradient_descent", use_nesterov_acc=TRUE,
                             delta_rel_conv=1E-6, lr_cov=0.1, lr_coef=0.1)
DEFAULT_OPTIM_PARAMS_V2 <- list(optimizer_cov="gradient_descent", use_nesterov_acc=TRUE,
                                delta_rel_conv=1E-6, lr_cov=0.01, lr_coef=0.1)
DEFAULT_OPTIM_PARAMS_NO_NESTEROV <- list(optimizer_cov="gradient_descent", use_nesterov_acc=FALSE,
                                         delta_rel_conv=1E-6, lr_cov=0.01, lr_coef=0.1)
DEFAULT_OPTIM_PARAMS_EARLY_STOP <- list(maxit=10, lr_cov=0.1, optimizer_cov="gradient_descent", lr_coef=0.1)
DEFAULT_OPTIM_PARAMS_EARLY_STOP_NO_NESTEROV <- list(maxit=20, lr_cov=0.01, use_nesterov_acc=FALSE,
                                                    optimizer_cov="gradient_descent", lr_coef=0.1)

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
sim_non_lin_f=function(n, init_c=0.4596534){
  X <- matrix(sim_rand_unif(2*n,init_c=init_c),ncol=2)
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
pred_prob <- predict(bst, data = X_test_plot, group_data_pred = group_data_pred, pred_latent = FALSE)$response_mean
pred <- predict(bst, data = X_test_plot, group_data_pred = group_data_pred, pred_latent = TRUE)
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
# pred <- predict(bst_std, data = X_test_plot, pred_latent=TRUE)
# lines(X_test_plot[,1],pred,col=5,lwd=3, lty=2)



# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  test_that("GPBoost algorithm with grouped random effects model for binary classification ", {
    
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
    capture.output( expect_error(gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                                         objective = "binary", train_gp_model_cov_pars=FALSE, nrounds=1, valids=valids)), file='NUL')
    
    # Validation metrics for training data
    # Default metric is "Approx. negative marginal log-likelihood" if there is only one training set
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                                   objective = "binary", train_gp_model_cov_pars=FALSE, nrounds=1), file='NUL')
    record_results <- gpb.get.eval.result(bst, "train", "Approx. negative marginal log-likelihood")
    expect_value <- 599.7875
    expect_lt(abs(record_results[1]-expect_value), TOLERANCE)
    capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                                   objective = "regression", train_gp_model_cov_pars=FALSE, nrounds=1), file='NUL')
    record_results <- gpb.get.eval.result(bst, "train", "Approx. negative marginal log-likelihood")
    expect_lt(abs(record_results[1]-expect_value), TOLERANCE)
    # Can also use other metrics
    capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                                   objective = "binary", train_gp_model_cov_pars=FALSE, nrounds=1,
                                   eval=list("binary_logloss","binary_error"), use_gp_model_for_validation = FALSE), file='NUL')
    record_results <- gpb.get.eval.result(bst, "train", "binary_logloss")
    expect_lt(abs(record_results[1]-0.6749475), TOLERANCE)
    record_results <- gpb.get.eval.result(bst, "train", "binary_error")
    expect_lt(abs(record_results[1]-0.466), TOLERANCE)
    capture.output( bst <- gpboost(data = X_train, label = y_train, gp_model = gp_model, verbose = 1,
                                   objective = "regression", train_gp_model_cov_pars=FALSE, nrounds=1,
                                   eval=list("l2","binary_error"), use_gp_model_for_validation = FALSE), file='NUL')
    record_results <- gpb.get.eval.result(bst, "train", "l2")
    expect_lt(abs(record_results[1]-0.2409613), TOLERANCE)
    record_results <- gpb.get.eval.result(bst, "train", "binary_error")
    expect_lt(abs(record_results[1]-0.466), TOLERANCE)
    
    # Find number of iterations using validation data with use_gp_model_for_validation=FALSE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_V2)
    capture.output( bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                                     learning_rate=0.1, objective = "binary", verbose = 0,
                                     use_gp_model_for_validation=FALSE, eval = "binary_error",
                                     early_stopping_rounds=10), file='NUL')
    record_results <- gpb.get.eval.result(bst, "test", "binary_error")
    expect_lt(abs(min(record_results)-0.323), TOLERANCE)
    expect_equal(which.min(record_results), 11)
    
    # Find number of iterations using validation data with use_gp_model_for_validation=TRUE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_V2)
    gp_model$set_prediction_data(group_data_pred = group_data_test)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.1, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "binary_error",
                     early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "binary_error")
    expect_lt(abs(min(record_results)-0.241), TOLERANCE)
    expect_equal(which.min(record_results), 16)
    # Compare to when ignoring random effects part
    bst <- gpb.train(data = dtrain, nrounds=100, valids=valids,
                     learning_rate=0.1, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "binary_error", early_stopping_rounds=10)
    expect_lt(abs(bst$best_score-.345), TOLERANCE)
    
    # Other metrics / losses
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_V2)
    gp_model$set_prediction_data(group_data_pred = group_data_test)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.5, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "binary_logloss",
                     early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "binary_logloss")
    expect_lt(abs(min(record_results)-0.4917727), TOLERANCE)
    expect_equal(which.min(record_results), 6)
    capture.output( bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                                     learning_rate=0.5, objective = "regression", verbose = 0,
                                     use_gp_model_for_validation=TRUE, eval = "l2", early_stopping_rounds=10), file='NUL')
    record_results <- gpb.get.eval.result(bst, "test", "l2")
    expect_lt(abs(min(record_results)-0.1643671), TOLERANCE)
    expect_equal(which.min(record_results), 6)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.5, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "l2", early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "l2")
    expect_lt(abs(min(record_results)-0.1643671), TOLERANCE)
    expect_equal(which.min(record_results), 6)
    
    # CV for finding number of boosting iterations when use_gp_model_for_validation = FALSE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_V2)
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
    expect_iter <- 9
    expect_score <- 0.352
    expect_equal(cvbst$best_iter, expect_iter)
    expect_lt(abs(cvbst$best_score-expect_score), TOLERANCE)
    # same thing but "wrong" likelihood given in gp_model
    gp_model <- GPModel(group_data = group_data_train, likelihood="gaussian")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_V2)
    capture.output( cvbst <- gpb.cv(params = params,
                                    data = dtrain,
                                    gp_model=gp_model,
                                    nrounds = 100,
                                    nfold = 4,
                                    eval = "binary_error",
                                    early_stopping_rounds = 5,
                                    use_gp_model_for_validation = FALSE,
                                    fit_GP_cov_pars_OOS = FALSE,
                                    folds = folds,
                                    verbose = 0), file='NUL')
    expect_equal(cvbst$best_iter, expect_iter)
    expect_lt(abs(cvbst$best_score-expect_score), TOLERANCE)
    # CV for finding number of boosting iterations when use_gp_model_for_validation = TRUE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_NO_NESTEROV)
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
    expect_iter <- 15
    expect_score <- 0.242
    expect_equal(cvbst$best_iter, expect_iter)
    expect_lt(abs(cvbst$best_score-expect_score), TOLERANCE)
    
    # Create random effects model and train GPBoost model
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_NO_NESTEROV)
    bst <- gpboost(data = X_train,
                   label = y_train,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "binary",
                   verbose = 0)
    cov_pars <- c(0.4590874, 0.3459219)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(head(pred$fixed_effect)-c(0.51870398, -0.03819206, 0.99821096, 0.86094202, -0.65647551, 0.83694023))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-1.114939, -1.063664, -1.239394,
                                                      rep(0,n_new)))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.1292918, 0.1284037, 0.1291332,
                                                     rep(0.8050093,n_new)))),TOLERANCE)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.0171176, 0.6237437, 0.1986085, 0.6377248))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.01682459, 0.23468750, 0.15916318, 0.23103189))),TOLERANCE)
    
    # Prediction when having only one grouped random effect
    group_1 <- rep(1,ntrain) # grouping variable
    for(i in 1:m) group_1[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
    probs_1 <- pnorm(f[1:ntrain] + b1[group_1])
    y_1 <- as.numeric(sim_rand_unif(n=ntrain, init_c=0.574) < probs_1)
    gp_model <- GPModel(group_data = group_1, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_NO_NESTEROV)
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
    pred <- predict(bst, data = X_test[1:length(unique(b1)),], group_data_pred = 1:length(unique(b1)), pred_latent = TRUE)
    expect_lt(abs(sqrt(sum((pred$random_effect_mean - b1)^2))-1.667843),TOLERANCE)
    # Prediction for only new groups
    group_test <- c(-1,-1,-2,-2)
    pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test, pred_latent = TRUE)
    fix_eff <- c(0.2292592, 0.3296304, 0.6725046, 0.5069731)
    expect_lt(sum(abs(pred$fixed_effect-fix_eff)),TOLERANCE)
    expect_lt(sum(abs(pred$random_effect_mean-rep(0,4))),TOLERANCE)
    pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test, pred_latent = FALSE)
    resp <- c(0.5739159, 0.6056269, 0.7076881, 0.6598638)
    expect_lt(sum(abs(pred$response_mean-resp)),TOLERANCE)
    # Prediction for only new cluster_ids
    cluster_ids_pred <- c(-1L,-1L,-2L,-2L)
    group_test <- c(1,3,3,9999)
    pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test,
                    cluster_ids_pred = cluster_ids_pred, pred_latent = TRUE)
    expect_lt(sum(abs(pred$random_effect_mean-rep(0,4))),TOLERANCE)
    expect_lt(sum(abs(pred$fixed_effect-fix_eff)),TOLERANCE)
    pred <- predict(bst, data = X_test[1:4,], group_data_pred = group_test,
                    cluster_ids_pred = cluster_ids_pred, pred_latent = FALSE)
    expect_lt(sum(abs(pred$response_mean-resp)),TOLERANCE)
    
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(1, 1))),TOLERANCE)
    # LaGaBoostOOS algorithm
    #   1. Run LaGaBoost algorithm separately on every fold and fit parameters on out-of-sample data
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_NO_NESTEROV)
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
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.4255016, 0.3026152))),TOLERANCE)
    expect_equal(cvbst$best_iter, 15)
    expect_lt(abs(cvbst$best_score-0.242), TOLERANCE)
    #   2. Run LaGaBoost algorithm on entire data while holding covariance parameters fixed
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 15,
                     params = params, train_gp_model_cov_pars = FALSE, verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.4255016, 0.3026152))),TOLERANCE)
    #   3. Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(head(pred$fixed_effect, n=4)-c(0.4455938, -0.2227164, 0.8109617, 0.6144774))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-1.050472, -1.025383, -1.187068,
                                                      rep(0,n_new)))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.1165838, 0.1175573, 0.1174311,
                                                     rep(0.7282491,n_new)))),TOLERANCE)
    
    # Use of validation data and cross-validation with custom metric
    bin_cust_error <- function(preds, dtrain) {
      labels <- getinfo(dtrain, "label")
      predsbin <- preds > 0.55
      error <- mean(predsbin!=labels)#mean((preds-labels)^4)
      return(list(name="bin_cust_error",value=error,higher_better=FALSE))
    }
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_V2)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.1, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=FALSE,
                     early_stopping_rounds=10, eval = bin_cust_error, metric = "bin_cust_error")
    expect_equal(bst$best_iter, 17)
    expect_lt(abs(bst$best_score - 0.358),TOLERANCE)
    # CV
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_V2)
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
    expect_equal(cvbst$best_iter, 7)
    expect_lt(abs(cvbst$best_score-0.365), TOLERANCE)
    
    # Training using Nelder-Mead
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead"))
    bst <- gpboost(data = X_train,
                   label = y_train,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "binary",
                   verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.4682746, 0.3544995))),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(head(pred$fixed_effect,n=4)-c(0.53963543, -0.09143685, 0.97199209, 0.82756999))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-1.121577, -1.057764, -1.243746,
                                                      rep(0,n_new)))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.1294601, 0.1286418, 0.1289668,
                                                     rep(0.8227741,n_new)))),TOLERANCE)
    
    # # Training using BFGS (sometimes crashes)
    # gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    # gp_model$set_optim_params(params=list(optimizer_cov="bfgs"))
    # bst <- gpboost(data = X_train,
    #                label = y_train,
    #                gp_model = gp_model,
    #                nrounds = 1,
    #                learning_rate = 0.1,
    #                max_depth = 6,
    #                min_data_in_leaf = 5,
    #                objective = "binary",
    #                verbose = 0)
    # expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.3806467, 0.2682585))),TOLERANCE)
  
  })
  
  test_that("GPBoost algorithm for binary classification when having only one grouping variable", {
    
    ntrain <- ntest <- 1000
    n <- ntrain + ntest
    # Simulate fixed effects
    sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.2644234)
    f <- sim_data$f
    f <- f - mean(f)
    X <- sim_data$X
    # Simulate grouped random effects
    sigma2_1 <- 1 # variance of random effect
    m <- 40 # number of categories / levels for grouping variable
    # first random effect
    group <- rep(1,ntrain) # grouping variable
    for(i in 1:m) group[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
    group <- c(group, group)
    n_new <- 3# number of new random effects in test data
    group[(length(group)-n_new+1):length(group)] <- rep(99999,n_new)
    Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
    b1 <- sqrt(sigma2_1) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.5542))
    eps <- Z1 %*% b1
    eps <- eps - mean(eps)
    group_data <- group
    # Observed data
    probs <- pnorm(f + eps)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.574) < probs)
    # Signal-to-noise ratio of approx. 1
    # var(f) / var(eps)
    # Split in training and test data
    y_train <- y[1:ntrain]
    X_train <- X[1:ntrain,]
    group_data_train <- group_data[1:ntrain]
    y_test <- y[1:ntest+ntrain]
    X_test <- X[1:ntest+ntrain,]
    f_test <- f[1:ntest+ntrain]
    group_data_test <- group_data[1:ntest+ntrain]
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
    
    # Find number of iterations using validation data with use_gp_model_for_validation=FALSE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.1, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=FALSE, eval = "binary_error",
                     early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "binary_error")
    expect_lt(abs(min(record_results)-0.356), TOLERANCE)
    expect_equal(which.min(record_results), 17)
    # Find number of iterations using validation data with use_gp_model_for_validation=TRUE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    gp_model$set_prediction_data(group_data_pred = group_data_test)
    bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                     learning_rate=0.1, objective = "binary", verbose = 0,
                     use_gp_model_for_validation=TRUE, eval = "binary_error",
                     early_stopping_rounds=10)
    record_results <- gpb.get.eval.result(bst, "test", "binary_error")
    expect_lt(abs(min(record_results)-0.263), TOLERANCE)
    expect_equal(which.min(record_results), 31)
    # Find number of iterations using validation when specifying "wrong" likelihood in gp_model
    gp_model <- GPModel(group_data = group_data_train, likelihood = "gaussian")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    gp_model$set_prediction_data(group_data_pred = group_data_test)
    capture.output( bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                                     learning_rate=0.1, objective = "binary", verbose = 0,
                                     use_gp_model_for_validation=TRUE, eval = "binary_error",
                                     early_stopping_rounds=10), file='NUL')
    record_results <- gpb.get.eval.result(bst, "test", "binary_error")
    expect_lt(abs(min(record_results)-0.263), TOLERANCE)
    expect_equal(which.min(record_results), 31)
    # Find number of iterations using validation when specifying "wrong" objective in gpb.train
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    gp_model$set_prediction_data(group_data_pred = group_data_test)
    capture.output( bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds=100, valids=valids,
                                     learning_rate=0.1, objective = "regression_l2", verbose = 0,
                                     use_gp_model_for_validation=TRUE, eval = "binary_error",
                                     early_stopping_rounds=10), file='NUL')
    record_results <- gpb.get.eval.result(bst, "test", "binary_error")
    expect_lt(abs(min(record_results)-0.263), TOLERANCE)
    expect_equal(which.min(record_results), 31)
    
    # CV for finding number of boosting iterations when use_gp_model_for_validation = FALSE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
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
    expect_equal(cvbst$best_iter, 12)
    expect_lt(abs(cvbst$best_score-0.383), TOLERANCE)
    # CV for finding number of boosting iterations when use_gp_model_for_validation = TRUE
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    cvbst <- gpb.cv(params = params,
                    data = dtrain,
                    gp_model = gp_model,
                    nrounds = 100,
                    nfold = 4,
                    eval = "binary_error",
                    early_stopping_rounds = 5,
                    use_gp_model_for_validation = TRUE,
                    folds = folds,
                    verbose = 0)
    expect_equal(cvbst$best_iter, 5)
    expect_lt(abs(cvbst$best_score-0.259), TOLERANCE)
    # same thing but "wrong" likelihood in gp_model
    gp_model <- GPModel(group_data = group_data_train, likelihood = "gaussian")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    capture.output( cvbst <- gpb.cv(params = params,
                                    data = dtrain,
                                    gp_model = gp_model,
                                    nrounds = 100,
                                    nfold = 4,
                                    eval = "binary_error",
                                    early_stopping_rounds = 5,
                                    use_gp_model_for_validation = TRUE,
                                    folds = folds,
                                    verbose = 0), file='NUL')
    expect_equal(cvbst$best_iter, 5)
    expect_lt(abs(cvbst$best_score-0.259), TOLERANCE)
    # same thing but "wrong" objective in gpb.cv
    params_w <- params
    params_w[["objective"]] <- "regression_l2"
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    capture.output( cvbst <- gpb.cv(params = params_w,
                                    data = dtrain,
                                    gp_model = gp_model,
                                    nrounds = 100,
                                    nfold = 4,
                                    eval = "binary_error",
                                    early_stopping_rounds = 5,
                                    use_gp_model_for_validation = TRUE,
                                    folds = folds,
                                    verbose = 0), file='NUL')
    expect_equal(cvbst$best_iter, 5)
    expect_lt(abs(cvbst$best_score-0.259), TOLERANCE)
    
    # Create random effects model and train GPBoost model
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    bst <- gpboost(data = X_train,
                   label = y_train,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "binary",
                   verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.9865279)),TOLERANCE)
    
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(head(pred$fixed_effect,n=4)-c(0.3656899, 0.5207786, 0.6272630, 0.5435066))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-2.004453, -2.004453, -2.004453,
                                                      rep(0,n_new)))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_cov)-c(0.2156130, 0.2156130, 0.2156130,
                                                     rep(0.9865279,n_new)))),TOLERANCE)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean,n=4)-c(0.003516566, 0.589670698, 0.262059586, 0.409466628))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$response_var,n=4)-c(0.0035042, 0.2419592, 0.1933844, 0.2418037))),TOLERANCE)
    
    # Training using Nelder-Mead
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead"))
    bst <- gpboost(data = X_train,
                   label = y_train,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "binary",
                   verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.9927358)),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean,n=4)-c(0.003543759, 0.588232583, 0.262807287, 0.401855180))),TOLERANCE)
    
    # Training using BFGS
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(optimizer_cov="bfgs"))
    bst <- gpboost(data = X_train,
                   label = y_train,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "binary",
                   verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.9855134)),TOLERANCE)
    
  })
  
  # This is a slow test
  test_that("GPBoost algorithm with Gaussian process model for binary classification ", {
    
    ntrain <- ntest <- 500
    n <- ntrain + ntest
    # Simulate fixed effects
    sim_data <- sim_non_lin_f(n=n, init_c=0.78345)
    f <- sim_data$f/2
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
    eps <- eps - mean(eps)
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
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 9,
                     learning_rate = 0.2,
                     max_depth = 10,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    cov_pars_est <- c(0.1776908, 0.1887078)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-c(-0.25248234, 0.07336944, 0.19282985, 0.04100225))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-c(0.09672839, 0.10432856, 0.09164587, 0.09215657))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-c(0.4087100, -0.5570364, -0.7904685, 0.5055812))),TOLERANCE)
    # Predict response
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean,n=4)-c(0.5592939, 0.3226671, 0.2836602, 0.6995181))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$response_var,n=4)-c(0.2464842, 0.2185530, 0.2031971, 0.2101925))),TOLERANCE)
    
    # Use validation set to determine number of boosting iteration with use_gp_model_for_validation = FALSE
    dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
    valids <- list(test = dtest)
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 20,
                     learning_rate = 0.2,
                     max_depth = 10,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0,
                     valids = valids,
                     early_stopping_rounds = 2,
                     use_gp_model_for_validation = FALSE)
    expect_equal(bst$best_iter, 9)
    expect_lt(abs(bst$best_score - 0.6020659),TOLERANCE)
    # Also use GPModel for calculating validation error
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    gp_model$set_prediction_data(gp_coords_pred = coords_test)
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 20,
                     learning_rate = 0.2,
                     max_depth = 10,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0,
                     valids = valids,
                     early_stopping_rounds = 2,
                     use_gp_model_for_validation = TRUE)
    expect_equal(bst$best_iter, 9)
    expect_lt(abs(bst$best_score - 0.5785662),TOLERANCE)
    
    # Training with Vecchia approximation
    capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                        likelihood = "bernoulli_probit", vecchia_approx =TRUE, num_neighbors = 30),
                    file='NUL')
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 9,
                     learning_rate = 0.2,
                     max_depth = 10,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    cov_pars_est <- c(0.1545302, 0.2631829)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-c(-0.2716762, 0.2615965, 0.3166944, 0.1967180))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-c(0.04289627, 0.06221441, 0.05985171, 0.05598201))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-c(0.2586964, -0.8518375, -0.8887175, 0.3031393))),TOLERANCE)
    
    # Training with Wendland covariance
    capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "wendland",
                                        cov_fct_shape=1, cov_fct_taper_range=0.2,
                                        likelihood = "bernoulli_probit"), file='NUL')
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 9,
                     learning_rate = 0.2,
                     max_depth = 10,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.1632674)),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-c(-0.26118832, -0.04498759, 0.19187178, 0.15228845))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-c(0.1364641, 0.1208750, 0.1170557, 0.1251195))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-c(0.4521438, -0.6149245, -0.5831075, 0.4807347))),TOLERANCE)
    
    # Wendland covariance and Nelder-Mead
    capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "wendland",
                                        cov_fct_shape=1, cov_fct_taper_range=0.2,
                                        likelihood = "bernoulli_probit"), file='NUL')
    gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead"))
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 9,
                     learning_rate = 0.2,
                     max_depth = 10,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-0.1633442)),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-c(-0.26069987, -0.04457552, 0.19229076, 0.15268263))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-c(0.1364151, 0.1208361, 0.1170151, 0.1250687))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-c(0.4510624, -0.6160705, -0.5842803, 0.4796462))),TOLERANCE)
    
    # Tapering
    capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential_tapered",
                                        likelihood = "bernoulli_probit", cov_fct_shape=1, cov_fct_taper_range=10) )
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 9,
                     learning_rate = 0.2,
                     max_depth = 10,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    cov_pars_est <- c(0.1777562, 0.1898083)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-c(-0.25264933, 0.07306853, 0.19296519, 0.04058235))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-c(0.09654161, 0.10422011, 0.09149145, 0.09198057))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-c(0.4089283, -0.5569100, -0.7903136, 0.5057746))),TOLERANCE)
  })
  
  
  test_that("GPBoost algorithm with GP model for binary classification with multiple observations at the same location", {
    
    ntrain <- ntest <- 400
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
    coords <- matrix(sim_rand_unif(n=n*d/8, init_c=0.12), ncol=d)
    coords <- rbind(coords,coords,coords,coords,coords,coords,coords,coords)
    D <- as.matrix(dist(coords))
    Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-15,n)
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
    # Folds for CV
    group_aux <- rep(1,ntrain) # grouping variable
    for(i in 1:(ntrain/4)) group_aux[(1:4)+4*(i-1)] <- 1:4
    folds <- list()
    for(i in 1:4) folds[[i]] <- as.integer(which(group_aux==i))
    
    # Train model
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(maxit=10, lr_cov=0.01, optimizer_cov="gradient_descent",
                                          lr_coef=0.1))
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 2,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.6094175, 0.1137471))),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$random_effect_mean, n=4)-c(-0.5066947, -0.4004523, -1.3271121, -1.0627929))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c(0.1536139, 0.2059765, 0.2395216, 0.2395186))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(0.4461313, 0.5287861, 0.5025805, 0.5287861))),TOLERANCE)
    # Predict response
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.4775558, 0.5465922, 0.2294873, 0.3157580))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.2494963, 0.2478292, 0.1768229, 0.2160549))),TOLERANCE)
    
    # Use validation set to determine number of boosting iteration
    dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
    valids <- list(test = dtest)
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_EARLY_STOP)
    gp_model$set_prediction_data(gp_coords_pred = coords_test)
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 10,
                     learning_rate = 0.1,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0,
                     valids = valids,
                     early_stopping_rounds = 2,
                     use_gp_model_for_validation = TRUE)
    expect_equal(bst$best_iter, 10)
    expect_lt(abs(bst$best_score - 0.6129572),TOLERANCE)
    # same thing but "wrong" likelihood in gp_model
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "gaussian")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_EARLY_STOP)
    gp_model$set_prediction_data(gp_coords_pred = coords_test)
    capture.output( bst <- gpb.train(data = dtrain,
                                     gp_model = gp_model,
                                     nrounds = 10,
                                     learning_rate = 0.1,
                                     max_depth = 6,
                                     min_data_in_leaf = 5,
                                     objective = "binary",
                                     verbose = 0,
                                     valids = valids,
                                     early_stopping_rounds = 2,
                                     use_gp_model_for_validation = TRUE), file='NUL')
    expect_equal(bst$best_iter, 10)
    expect_lt(abs(bst$best_score - 0.6129572),TOLERANCE)
    # same thing but "wrong" objective in gpb.train
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_EARLY_STOP)
    gp_model$set_prediction_data(gp_coords_pred = coords_test)
    capture.output( bst <- gpb.train(data = dtrain,
                                     gp_model = gp_model,
                                     nrounds = 10,
                                     learning_rate = 0.1,
                                     max_depth = 6,
                                     min_data_in_leaf = 5,
                                     objective = "regression_l2",
                                     verbose = 0,
                                     valids = valids,
                                     early_stopping_rounds = 2,
                                     use_gp_model_for_validation = TRUE,
                                     metrics = "binary_logloss"), file='NUL')
    expect_equal(bst$best_iter, 10)
    expect_lt(abs(bst$best_score - 0.6129572),TOLERANCE)
    
    # CV for finding number of boosting iterations when use_gp_model_for_validation = TRUE
    gp_model <- GPModel(gp_coords = coords_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_EARLY_STOP)
    cvbst <- gpb.cv(data = dtrain,
                    gp_model = gp_model,
                    nrounds = 10,
                    learning_rate = 0.1,
                    max_depth = 6,
                    min_data_in_leaf = 5,
                    objective = "binary",
                    eval = "binary_error",
                    early_stopping_rounds = 5,
                    use_gp_model_for_validation = TRUE,
                    folds = folds,
                    verbose = 0)
    expcet_iter <- 9
    expcet_score <- 0.315
    expect_equal(cvbst$best_iter, expcet_iter)
    expect_lt(abs(cvbst$best_score-expcet_score), TOLERANCE)
    # same thing but "wrong" likelihood in gp_model
    gp_model <- GPModel(gp_coords = coords_train, likelihood = "gaussian")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_EARLY_STOP)
    capture.output( cvbst <- gpb.cv(data = dtrain,
                                    gp_model = gp_model,
                                    nrounds = 10,
                                    learning_rate = 0.1,
                                    max_depth = 6,
                                    min_data_in_leaf = 5,
                                    objective = "binary",
                                    eval = "binary_error",
                                    early_stopping_rounds = 5,
                                    use_gp_model_for_validation = TRUE,
                                    folds = folds,
                                    verbose = 0), file='NUL')
    expect_equal(cvbst$best_iter, expcet_iter)
    expect_lt(abs(cvbst$best_score-expcet_score), TOLERANCE)
    # same thing but "wrong" objective in gpb.cv
    gp_model <- GPModel(gp_coords = coords_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_EARLY_STOP)
    capture.output( cvbst <- gpb.cv(data = dtrain,
                                    gp_model = gp_model,
                                    nrounds = 10,
                                    learning_rate = 0.1,
                                    max_depth = 6,
                                    min_data_in_leaf = 5,
                                    objective = "regression_l2",
                                    eval = "binary_error",
                                    early_stopping_rounds = 5,
                                    use_gp_model_for_validation = TRUE,
                                    folds = folds,
                                    verbose = 0), file='NUL')
    expect_equal(cvbst$best_iter, expcet_iter)
    expect_lt(abs(cvbst$best_score-expcet_score), TOLERANCE)
  })
  
  
  # This is a slow test
  test_that("GPBoost algorithm for binary classification with combined Gaussian process and grouped random effects model", {
    
    ntrain <- ntest <- 500
    n <- ntrain + ntest
    # Simulate fixed effects
    sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.6549)
    f <- sim_data$f
    f <- f - mean(f)
    X <- sim_data$X
    # Simulate spatial Gaussian process
    sigma2_1 <- 1 # marginal variance of GP
    rho <- 0.1 # range parameter
    d <- 2 # dimension of GP locations
    coords <- matrix(sim_rand_unif(n=n*d, init_c=0.633), ncol=d)
    D <- as.matrix(dist(coords))
    Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
    C <- t(chol(Sigma))
    b_1 <- qnorm(sim_rand_unif(n=n, init_c=0.67))
    eps <- as.vector(C %*% b_1)
    # Simulate grouped random effects
    sigma2_grp <- 1 # variance of random effect
    m <- 50 # number of categories / levels for grouping variable
    # first random effect
    group <- rep(1,ntrain) # grouping variable
    for(i in 1:m) group[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
    group <- c(group, group)
    n_new <- 3# number of new random effects in test data
    group[(length(group)-n_new+1):length(group)] <- rep(99999,n_new)
    Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
    b_grp <- sqrt(sigma2_grp) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.52))
    eps <- C %*% b_1 + Z1 %*% b_grp
    group_data <- group
    eps <- eps - mean(eps)
    # Observed data
    probs <- pnorm(f + eps)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.234) < probs)
    # Split into training and test data
    y_train <- y[1:ntrain]
    X_train <- X[1:ntrain,]
    coords_train <- coords[1:ntrain,]
    group_data_train <- group_data[1:ntrain]
    dtrain <- gpb.Dataset(data = X_train, label = y_train)
    y_test <- y[1:ntest+ntrain]
    X_test <- X[1:ntest+ntrain,]
    f_test <- f[1:ntest+ntrain]
    coords_test <- coords[1:ntest+ntrain,]
    group_data_test <- group_data[1:ntest+ntrain]
    eps_test <- eps[1:ntest+ntrain]
    
    # Train model
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 5,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.2389226, 0.2944397, 0.3476084))),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.7599814, 0.5543266, 0.1063388, 0.5439135))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.18240965, 0.24704862, 0.09503084, 0.24807160))),TOLERANCE)
    
    # The following is very slow
    # Train model using Nelder-Mead
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=list(optimizer_cov = "nelder_mead", delta_rel_conv=1E-8))
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 5,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-c(0.2390776, 0.2966670, 0.3499098))),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.7600335, 0.5543040, 0.1062553, 0.5437832))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.18238257, 0.24705107, 0.09496514, 0.24808303))),TOLERANCE)
    
    # Use validation set to determine number of boosting iteration
    dtest <- gpb.Dataset.create.valid(dtrain, data = X_test, label = y_test)
    valids <- list(test = dtest)
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    gp_model$set_prediction_data(gp_coords_pred = coords_test, group_data_pred = group_data_test)
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 100,
                     learning_rate = 0.1,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0,
                     valids = valids,
                     early_stopping_rounds = 2,
                     use_gp_model_for_validation = TRUE)
    expect_equal(bst$best_iter, 12)
    expect_lt(abs(bst$best_score - 0.5826652),TOLERANCE)
    # same thing but "wrong" likelihood given in gp_model
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        group_data = group_data_train, likelihood = "gaussian")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    gp_model$set_prediction_data(gp_coords_pred = coords_test, group_data_pred = group_data_test)
    capture.output( bst <- gpb.train(data = dtrain,
                                     gp_model = gp_model,
                                     nrounds = 100,
                                     learning_rate = 0.1,
                                     max_depth = 6,
                                     min_data_in_leaf = 5,
                                     objective = "binary",
                                     verbose = 0,
                                     valids = valids,
                                     early_stopping_rounds = 2,
                                     use_gp_model_for_validation = TRUE), file='NUL')
    expect_equal(bst$best_iter, 12)
    expect_lt(abs(bst$best_score - 0.5826652),TOLERANCE)
    # same thing but "wrong" objective given in gpb.train
    gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                        group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    gp_model$set_prediction_data(gp_coords_pred = coords_test, group_data_pred = group_data_test)
    capture.output( bst <- gpb.train(data = dtrain,
                                     gp_model = gp_model,
                                     nrounds = 100,
                                     learning_rate = 0.1,
                                     max_depth = 6,
                                     min_data_in_leaf = 5,
                                     objective = "regression_l2",
                                     eval = "binary_logloss",
                                     verbose = 0,
                                     valids = valids,
                                     early_stopping_rounds = 2,
                                     use_gp_model_for_validation = TRUE), file='NUL')
    expect_equal(bst$best_iter, 12)
    expect_lt(abs(bst$best_score - 0.5826652),TOLERANCE)
    
  })
  
  
  test_that("GPBoost algorithm for binary classification: equivalence of Vecchia approximation", {
    
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
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_EARLY_STOP_NO_NESTEROV)
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 5,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    cov_pars_est <- c(0.1195943, 0.1479688)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    predict_var = TRUE, pred_latent = TRUE)
    P_RE_mean <- c(-0.03827765, -0.15611348, 0.04603207, -0.03903325)
    P_RE_cov <- c(0.1013040, 0.1029115, 0.1098251, 0.1142902)
    P_F <- c(0.2807203, 0.9713023, -0.2379479, 1.1268341)
    expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-P_RE_mean)),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-P_RE_cov)),TOLERANCE)
    expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-P_F)),TOLERANCE)
    
    # Same thing with Vecchia approximation
    capture.output( gp_model <- GPModel(gp_coords = coords_train, cov_function = "exponential",
                                        likelihood = "bernoulli_probit", vecchia_approx =TRUE, num_neighbors = ntrain-1), file='NUL')
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_EARLY_STOP_NO_NESTEROV)
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 5,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$random_effect_mean,n=4)-P_RE_mean)),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_cov,n=4)-P_RE_cov)),TOLERANCE)
    expect_lt(sum(abs(tail(pred$fixed_effect,n=4)-P_F)),TOLERANCE)
  })
  
  test_that("GPBoost algorithm with Gaussian process model for binary classification with logit link", {
    
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
    gp_model$set_optim_params(params=list(maxit=10, lr_cov=0.01, optimizer_cov="gradient_descent",
                                          lr_coef=0.1))
    bst <- gpb.train(data = dtrain,
                     gp_model = gp_model,
                     nrounds = 2,
                     learning_rate = 0.5,
                     max_depth = 6,
                     min_data_in_leaf = 5,
                     objective = "binary",
                     verbose = 0)
    cov_pars_est <- c(0.41398781, 0.07678912)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(abs(sqrt(mean((pred$fixed_effect - f_test)^2))-0.8197184),TOLERANCE)
    expect_lt(abs(sqrt(mean((pred$random_effect_mean - eps_test)^2))-0.9186907),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_cov, n=4)-c(0.3368866, 0.3202246, 0.3128022, 0.3221874))),TOLERANCE)
    # Predict response
    pred <- predict(bst, data = X_test, gp_coords_pred = coords_test, 
                    predict_var = TRUE, pred_latent = FALSE)
    expect_equal(mean(as.numeric(pred$response_mean>0.5) != y_test),0.362)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.2365583, 0.2499360, 0.2041193, 0.2496736))),TOLERANCE)
  })
  
  test_that("GPBoost algorithm with grouped random effects for Poisson regression", {
    
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
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_V2)
    bst <- gpboost(data = dtrain,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "poisson",
                   verbose = 0)
    cov_pars_est <- c(0.5298689, 0.3680592)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(-1.8305056, 0.9506364, -0.8736293, 0.4120991))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$random_effect_mean)-c(-0.9853151, -0.9234402, -1.0387300, rep(0,3)))),TOLERANCE)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.05880294, 4.05273488, 0.65385231, 2.36518123))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.05905864, 27.92579356, 1.27525196, 10.49611686))),TOLERANCE)
  })
  
  test_that("GPBoost algorithm with grouped random effects for gamma regression", {
    
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
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_V2)
    bst <- gpboost(data = dtrain,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "gamma",
                   verbose = 0)
    cov_pars_est <- c(0.5946253, 0.5044433)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_est)),TOLERANCE)
    # Prediction
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = TRUE)
    expect_lt(sum(abs(tail(pred$fixed_effect, n=4)-c(-1.4125121, 0.8533486, -1.1365429, 0.5067477))),TOLERANCE)
    # Predict response
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = FALSE)
    expect_lt(sum(abs(tail(pred$response_mean, n=4)-c(0.04967546, 4.06690743, 0.55598796, 2.87565940))),TOLERANCE)
    expect_lt(sum(abs(tail(pred$response_var, n=4)-c(0.002804865, 82.743971184, 1.546459557, 41.369729192))),TOLERANCE)
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
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS_NO_NESTEROV)
    bst <- gpboost(data = X_train,
                   label = y_train,
                   gp_model = gp_model,
                   nrounds = 30,
                   learning_rate = 0.1,
                   max_depth = 6,
                   min_data_in_leaf = 5,
                   objective = "binary",
                   verbose = 0)
    # Predict raw score and response
    pred <- predict(bst, data = X_test, group_data_pred = group_data_test,
                    predict_var = TRUE, pred_latent = TRUE)
    pred_resp <- predict(bst, data = X_test, group_data_pred = group_data_test,
                         predict_var = TRUE, pred_latent = FALSE)
    pred2 <- predict(bst, data = X_test, group_data_pred = group_data_test,
                     predict_var = TRUE, pred_latent = TRUE,
                     num_iteration = 22, start_iteration = 0)
    pred_resp2 <- predict(bst, data = X_test, group_data_pred = group_data_test,
                          predict_var = TRUE, pred_latent = FALSE,
                          num_iteration = 22, start_iteration = 0)
    pred3 <- predict(bst, data = X_test, group_data_pred = group_data_test,
                     predict_var = TRUE, pred_latent = TRUE,
                     num_iteration = 20, start_iteration = 5)
    pred_resp3 <- predict(bst, data = X_test, group_data_pred = group_data_test,
                          predict_var = TRUE, pred_latent = FALSE,
                          num_iteration = 20, start_iteration = 5)
    # Save to file
    filename <- tempfile(fileext = ".model")
    gpb.save(bst, filename=filename, save_raw_data = FALSE)
    filename_num_it <- tempfile(fileext = ".model")
    gpb.save(bst, filename=filename_num_it, save_raw_data = FALSE, num_iteration = 22, start_iteration = 0)
    filename2 <- tempfile(fileext = ".model")
    gpb.save(bst, filename=filename2, save_raw_data = TRUE)
    # finalize and destroy models
    bst$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)
    rm(gp_model)
    # Load from file and make predictions again with save_raw_data = FALSE option
    bst_loaded <- gpb.load(filename = filename)
    pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                           predict_var = TRUE, pred_latent = TRUE)
    pred_resp_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                                predict_var = TRUE, pred_latent = FALSE)
    expect_equal(pred$fixed_effect, pred_loaded$fixed_effect)
    expect_equal(pred$random_effect_mean, pred_loaded$random_effect_mean)
    expect_equal(pred$random_effect_cov, pred_loaded$random_effect_cov)
    expect_equal(pred_resp$response_mean, pred_resp_loaded$response_mean)
    expect_equal(pred_resp$response_var, pred_resp_loaded$response_var)
    # Different num_iteration when saving
    bst_loaded <- gpb.load(filename = filename_num_it)
    pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                           predict_var = TRUE, pred_latent = TRUE)
    pred_resp_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                                predict_var = TRUE, pred_latent = FALSE)
    expect_equal(pred2$fixed_effect, pred_loaded$fixed_effect)
    expect_equal(pred2$random_effect_mean, pred_loaded$random_effect_mean)
    expect_equal(pred2$random_effect_cov, pred_loaded$random_effect_cov)
    expect_equal(pred_resp2$response_mean, pred_resp_loaded$response_mean)
    expect_equal(pred_resp2$response_var, pred_resp_loaded$response_var)
    expect_error({
      pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                             predict_var= TRUE, start_iteration=5)
    })
    # Load from file and make predictions again with save_raw_data = TRUE option
    bst_loaded <- gpb.load(filename = filename2)
    pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                           predict_var= TRUE, pred_latent = TRUE)
    pred_resp_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                                predict_var= TRUE, pred_latent = FALSE)
    expect_equal(pred$fixed_effect, pred_loaded$fixed_effect)
    expect_equal(pred$random_effect_mean, pred_loaded$random_effect_mean)
    expect_equal(pred$random_effect_cov, pred_loaded$random_effect_cov)
    expect_equal(pred_resp$response_mean, pred_resp_loaded$response_mean)
    expect_equal(pred_resp$response_var, pred_resp_loaded$response_var)
    # Same num_iteration when saving but different one for prediction
    pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                           predict_var = TRUE, pred_latent = TRUE, num_iteration = 22, start_iteration = 0)
    pred_resp_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                                predict_var = TRUE, pred_latent = FALSE, num_iteration = 22, start_iteration = 0)
    expect_equal(pred2$fixed_effect, pred_loaded$fixed_effect)
    expect_equal(pred2$random_effect_mean, pred_loaded$random_effect_mean)
    expect_equal(pred2$random_effect_cov, pred_loaded$random_effect_cov)
    expect_equal(pred_resp2$response_mean, pred_resp_loaded$response_mean)
    expect_equal(pred_resp2$response_var, pred_resp_loaded$response_var)
    # Set num_iteration and start_iteration
    pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                           predict_var= TRUE, pred_latent = TRUE,
                           num_iteration = 20, start_iteration = 5)
    pred_resp_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test,
                                predict_var= TRUE, pred_latent = FALSE,
                                num_iteration = 20, start_iteration = 5)
    expect_equal(pred3$fixed_effect, pred_loaded$fixed_effect)
    expect_equal(pred3$random_effect_mean, pred_loaded$random_effect_mean)
    expect_equal(pred3$random_effect_cov, pred_loaded$random_effect_cov)
    expect_equal(pred_resp3$response_mean, pred_resp_loaded$response_mean)
    expect_equal(pred_resp3$response_var, pred_resp_loaded$response_var)
    
  })
  
  test_that("Paramter tuning for GPBoost algorithm ", {
    
    ntrain <- 1000
    n <- ntrain
    # Simulate fixed effects
    sim_data <- sim_friedman3(n=n, n_irrelevant=5, init_c=0.12644234)
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
    n_new <- 3# number of new random effects in test data
    group[(length(group)-n_new+1):length(group)] <- rep(99999,n_new)
    Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
    b1 <- sqrt(sigma2_1) * qnorm(sim_rand_unif(n=length(unique(group)), init_c=0.53542))
    # Second random effect
    n_obs_gr <- ntrain/m# number of sampels per group
    group2 <- rep(1,ntrain) # grouping variable
    for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
    group2[(length(group2)-n_new+1):length(group2)] <- rep(99999,n_new)
    Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
    b2 <- sqrt(sigma2_2) * qnorm(sim_rand_unif(n=length(unique(group2)), init_c=0.282354))
    eps <- Z1 %*% b1 + Z2 %*% b2
    eps <- eps - mean(eps)
    group_data <- cbind(group,group2)
    # Observed data
    probs <- pnorm(f + eps)
    y <- as.numeric(sim_rand_unif(n=n, init_c=0.6574) < probs)
    # Split in training and test data
    y_train <- y[1:ntrain]
    X_train <- X[1:ntrain,]
    group_data_train <- group_data[1:ntrain,]
    # Folds for CV
    group_aux <- rep(1,ntrain) # grouping variable
    for(i in 1:(ntrain/4)) group_aux[(1:4)+4*(i-1)] <- 1:4
    folds <- list()
    for(i in 1:4) folds[[i]] <- as.integer(which(group_aux==i))
    
    #Parameter tuning using cross-validation: deterministic and random grid search
    gp_model <- GPModel(group_data = group_data_train, likelihood = "bernoulli_probit")
    gp_model$set_optim_params(params=DEFAULT_OPTIM_PARAMS)
    dtrain <- gpb.Dataset(data = X, label = y)
    params <- list(objective = "binary", verbose = 0)
    param_grid = list("learning_rate" = c(0.5,0.11), "min_data_in_leaf" = c(20),
                      "max_depth" = c(5), "num_leaves" = 2^17, "max_bin" = c(255,500))
    opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid,
                                                  params = params,
                                                  num_try_random = NULL,
                                                  data = dtrain,
                                                  gp_model = gp_model,
                                                  verbose_eval = 1,
                                                  nrounds = 100,
                                                  early_stopping_rounds = 5,
                                                  eval = "binary_logloss",
                                                  folds = folds)
    
    expect_lt(abs(opt_params$best_score-0.5131497),TOLERANCE)
    expect_equal(opt_params$best_iter,31)
    expect_equal(opt_params$best_params$learning_rate,0.11)
    expect_equal(opt_params$best_params$max_bin,255)
    expect_equal(opt_params$best_params$max_depth,5)
  })
  
}