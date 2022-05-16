context("GPModel_grouped_random_effects")

TOLERANCE <- 1E-3
TOLERANCE2 <- 1E-2
TOLERANCE1 <- 1E-1
TOL_STRICT <- 1E-6

if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  # Function that simulates uniform random variables
  sim_rand_unif <- function(n, init_c=0.1){
    mod_lcg <- 134456 # modulus for linear congruential generator (random0 used)
    sim <- rep(NA, n)
    sim[1] <- floor(init_c * mod_lcg)
    for(i in 2:n) sim[i] <- (8121 * sim[i-1] + 28411) %% mod_lcg
    return(sim / mod_lcg)
  }
  
  # Create data
  n <- 1000 # number of samples
  # First grouped random effects model
  m <- 100 # number of categories / levels for grouping variable
  group <- rep(1,n) # grouping variable
  for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
  Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
  b1 <- qnorm(sim_rand_unif(n=m, init_c=0.546))
  # Second random effects
  n_gr <- n/20 # number of groups
  group2 <- rep(1,n) # grouping variable
  for(i in 1:(n/n_gr)) group2[(1:n_gr)+n_gr*(i-1)] <- 1:n_gr
  Z2 <- model.matrix(rep(1,n)~factor(group2)-1)
  b2 <- qnorm(sim_rand_unif(n=length(unique(group2)), init_c=0.46))
  # Random slope / coefficient
  x <- cos((1:n-n/2)^2*5.5*pi/n) # covariate data for random slope
  Z3 <- diag(x) %*% Z1
  b3 <- qnorm(sim_rand_unif(n=m, init_c=0.69))
  # Error term
  xi <- sqrt(0.5) * qnorm(sim_rand_unif(n=n, init_c=0.1))
  # Data for linear mixed effects model
  X <- cbind(rep(1,n),sin((1:n-n/2)^2*2*pi/n)) # design matrix / covariate data for fixed effect
  beta <- c(2,2) # regression coefficients
  # cluster_ids 
  cluster_ids <- c(rep(1,0.4*n),rep(2,0.6*n))
  
  test_that("single level grouped random effects model ", {
    
    y <- as.vector(Z1 %*% b1) + xi
    # Estimation using Fisher scoring
    gp_model <- GPModel(group_data = group)
    fit(gp_model, y = y, params = list(std_dev = TRUE, optimizer_cov = "fisher_scoring",
                                       convergence_criterion = "relative_change_in_parameters"))
    cov_pars <- c(0.49348532, 0.02326312, 1.22299521, 0.17995161)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOL_STRICT)
    expect_equal(dim(gp_model$get_cov_pars())[2], 2)
    expect_equal(dim(gp_model$get_cov_pars())[1], 2)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    # Using gradient descent instead of Fisher scoring
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "gradient_descent", std_dev = FALSE,
                                         lr_cov = 0.1, use_nesterov_acc = FALSE, maxit = 1000,
                                         convergence_criterion = "relative_change_in_parameters"))
    cov_pars_est <- as.vector(gp_model$get_cov_pars())
    expect_lt(sum(abs(cov_pars_est-cov_pars[c(1,3)])),1E-5)
    expect_equal(class(cov_pars_est), "numeric")
    expect_equal(length(cov_pars_est), 2)
    expect_equal(gp_model$get_num_optim_iter(), 9)
    # Using gradient descent with Nesterov acceleration
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "gradient_descent", std_dev = FALSE,
                                         lr_cov = 0.2, use_nesterov_acc = TRUE,
                                         acc_rate_cov = 0.1, maxit = 1000,
                                         convergence_criterion = "relative_change_in_parameters"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(1,3)])),1E-5)
    expect_equal(gp_model$get_num_optim_iter(), 10)
    # Using gradient descent and a too large learning rate
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "gradient_descent", std_dev = FALSE,
                                         lr_cov = 10, use_nesterov_acc = FALSE,
                                         maxit = 1000, convergence_criterion = "relative_change_in_parameters"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(1,3)])),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 32)
    # Different termination criterion
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                         convergence_criterion = "relative_change_in_log_likelihood"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    # Nelder-Mead
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "nelder_mead", std_dev = TRUE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
    expect_equal(gp_model$get_num_optim_iter(), 12)
    # Adam
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "adam", std_dev = TRUE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
    expect_equal(gp_model$get_num_optim_iter(), 275)
    
    # Evaluation of likelihood
    ll <- gp_model$neg_log_likelihood(y=y,cov_pars=gp_model$get_cov_pars()[1,])
    expect_lt(abs(ll-(1228.293)),1E-2)
    
    # Prediction 
    gp_model <- GPModel(group_data = group)
    group_test <- c(1,2,m+1)
    expect_error(predict(gp_model, y=y, cov_pars = c(0.5,1.5)))# group data not provided
    pred <- predict(gp_model, y=y, group_data_pred = group_test,
                    cov_pars = c(0.5,1.5), predict_cov_mat = TRUE)
    expected_mu <- c(-0.1553877, -0.3945731, 0)
    expected_cov <- c(0.5483871, 0.0000000, 0.0000000, 0.0000000,
                      0.5483871, 0.0000000, 0.0000000, 0.0000000, 2)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOL_STRICT)
    # Predict variances
    pred <- predict(gp_model, y=y, group_data_pred = group_test,
                    cov_pars = c(0.5,1.5), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOL_STRICT)
    
    # Prediction from fitted model
    gp_model <- fitGPModel(group_data = group, y = y,
                           params = list(optimizer_cov = "fisher_scoring",
                                         convergence_criterion = "relative_change_in_parameters"))
    group_test <- c(1,2,m+1)
    pred <- predict(gp_model, group_data_pred = group_test, predict_cov_mat = TRUE)
    expected_mu <- c(-0.1543396, -0.3919117, 0.0000000)
    expected_cov <- c(0.5409198 , 0.0000000000, 0.0000000000, 0.0000000000,
                      0.5409198 , 0.0000000000, 0.0000000000, 0.0000000000, 1.7164805)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOL_STRICT)
    
    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
    first_occurences <- match(unique(group), group)
    training_data_random_effects <- all_training_data_random_effects[first_occurences] 
    group_unique <- unique(group)
    pred_random_effects <- predict(gp_model, group_data_pred = group_unique)
    expect_lt(sum(abs(training_data_random_effects - pred_random_effects$mu)),TOL_STRICT)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1),y=y)
    expect_lt(abs(nll-2282.073),1E-2)
    
    # Do optimization using optim and e.g. Nelder-Mead
    gp_model <- GPModel(group_data = group)
    opt <- optim(par=c(1,1), fn=gp_model$neg_log_likelihood, y=y, method="Nelder-Mead")
    expect_lt(sum(abs(opt$par-cov_pars[c(1,3)])),TOLERANCE)
    expect_lt(abs(opt$value-(1228.293)),1E-2)
    expect_equal(as.integer(opt$counts[1]), 49)
    
    # Use non-ordered grouping data
    set.seed(1)
    shuffle_ind <- sample.int(n=n,size=n,replace=FALSE)
    gp_model <- GPModel(group_data = group[shuffle_ind])
    fit(gp_model, y = y[shuffle_ind], params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE,
                                                    convergence_criterion = "relative_change_in_parameters"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    
    # Inf in response variable data gives error
    y_Inf <- y
    y_Inf[1] = -Inf
    expect_error(gp_model <- fitGPModel(group_data = group, y = y_Inf))
  })
  
  
  test_that("linear mixed effects model with grouped random effects ", {
    
    y <- Z1 %*% b1 + X%*%beta + xi
    # Fit model
    gp_model <- fitGPModel(group_data = group,
                           y = y, X = X,
                           params = list(optimizer_cov = "fisher_scoring",
                                         optimizer_coef = "wls", std_dev = TRUE,
                                         convergence_criterion = "relative_change_in_parameters"))
    cov_pars <- c(0.49205230, 0.02319557, 1.22064076, 0.17959832)
    coef <- c(2.07499902, 0.11269252, 1.94766255, 0.03382472)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 7)
    
    # Prediction
    group_test <- c(1,2,m+1)
    X_test <- cbind(rep(1,3),c(-0.5,0.2,0.4))
    expect_error(predict(gp_model,group_data_pred = group_test))# covariate data not provided
    pred <- predict(gp_model, group_data_pred = group_test,
                    X_pred = X_test, predict_cov_mat = TRUE)
    expected_mu <- c(0.886494, 2.043259, 2.854064)
    expected_cov <- c(0.5393509 , 0.0000000000, 0.0000000000, 0.0000000000,
                      0.5393509 , 0.0000000000, 0.0000000000, 0.0000000000, 1.712693)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOL_STRICT)
    
    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
    first_occurences <- match(unique(group), group)
    training_data_random_effects <- all_training_data_random_effects[first_occurences] 
    group_unique <- unique(group)
    X_zero <- cbind(rep(0,length(group_unique)),rep(0,length(group_unique)))
    pred_random_effects <- predict(gp_model, group_data_pred = group_unique, X_pred = X_zero)
    expect_lt(sum(abs(training_data_random_effects - pred_random_effects$mu)),TOL_STRICT)
    
    # Fit model using gradient descent instead of wls for regression coefficients
    gp_model <- fitGPModel(group_data = group,
                           y = y, X = X,
                           params = list(optimizer_cov = "gradient_descent", maxit=1000, std_dev = TRUE,
                                         optimizer_coef = "gradient_descent", lr_coef=1, use_nesterov_acc=TRUE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 8)
    
    # Fit model using Nelder-Mead
    gp_model <- fitGPModel(group_data = group,
                           y = y, X = X,
                           params = list(optimizer_cov = "nelder_mead",
                                         optimizer_coef = "nelder_mead", std_dev = FALSE, delta_rel_conv=1e-6))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(1,3)])),TOLERANCE2)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef[c(1,3)])),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 125)
    # Fit model using Adam
    gp_model <- fitGPModel(group_data = group,
                           y = y, X = X,
                           params = list(optimizer_cov = "adam", std_dev = FALSE, delta_rel_conv=1e-6))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars[c(1,3)])),TOLERANCE2)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef[c(1,3)])),TOLERANCE2)
    expect_equal(gp_model$get_num_optim_iter(), 354)
    
    # Fit model using BFGS
    gp_model <- fitGPModel(group_data = group,
                           y = y, X = X,
                           params = list(optimizer_cov = "bfgs", std_dev = TRUE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 11)
    
    # Large data
    n_L <- 1e6 # number of samples
    m_L <- n_L/10 # number of categories / levels for grouping variable
    group_L <- rep(1,n_L) # grouping variable
    for(i in 1:m_L) group_L[((i-1)*n_L/m_L+1):(i*n_L/m_L)] <- i
    keps <- 1E-10
    b1_L <- qnorm(sim_rand_unif(n=m_L, init_c=0.846)*(1-keps) + keps/2)
    X_L <- cbind(rep(1,n_L),sim_rand_unif(n=n_L, init_c=0.341)) # design matrix / covariate data for fixed effect
    beta <- c(2,2) # regression coefficients
    xi_L <- sqrt(0.5) * qnorm(sim_rand_unif(n=m_L, init_c=0.321)*(1-keps) + keps/2)
    y_L <- b1_L[group_L] + X_L%*%beta + xi_L
    # Fit model using gradient descent instead of wls for regression coefficients
    gp_model <- fitGPModel(group_data = group_L,
                           y = y_L, X = X_L,
                           params = list(optimizer_cov = "gradient_descent", maxit=1000, std_dev = TRUE,
                                         optimizer_coef = "gradient_descent", lr_coef=0.1, use_nesterov_acc=TRUE))
    cov_pars <- c(0.5005068978, 0.0007461116, 0.9982863693, 0.0046888995)
    coef <- c(1.994283503, 0.003484753, 2.004006668, 0.002577150)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 7)
    
    gp_model <- fitGPModel(group_data = group_L,
                           y = y_L, X = X_L,
                           params = list(optimizer_cov = "nelder_mead", maxit=1000))
    cov_pars <- c(0.5004681, 0.9988288)
    coef <- c(2.000747, 1.999343)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(gp_model$get_coef())-coef)),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 152)
    
  })
  

  test_that("Multiple grouped random effects ", {
    
    ## Two crossed random effects
    y <- Z1%*%b1 + Z2%*%b2 + xi
    # Fisher scoring
    gp_model <- fitGPModel(group_data = cbind(group,group2), y = y, 
                           params = list(optimizer_cov = "fisher_scoring", std_dev = TRUE))
    expected_values <- c(0.49792062, 0.02408196, 1.21972166, 0.18357646, 1.06962710, 0.22567292)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    
    # Prediction after training
    group_data_pred = cbind(c(1,1,m+1),c(2,1,length(group2)+1))
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred, predict_var = TRUE)
    expected_mu <- c(0.7509175, -0.4208015, 0.0000000)
    expected_var <- c(0.5677178, 0.5677178, 2.7872694)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),1E-4)
    # Prediction without training and parameters given
    gp_model <- GPModel(group_data = cbind(group,group2))
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred,
                             cov_pars = c(0.1,1,2), predict_cov_mat = TRUE)
    expected_mu <- c(0.7631462, -0.4328551, 0.000000000)
    expected_cov <- c(0.114393721, 0.009406189, 0.0000000, 0.009406189,
                      0.114393721 , 0.0000000, 0.0000000, 0.0000000, 3.100000000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE)
    # Prediction for only existing random effects
    group_data_pred_in = cbind(c(1,1),c(2,1))
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred_in,
                             cov_pars = c(0.1,1,2), predict_cov_mat = TRUE)
    expected_mu <- c(0.7631462, -0.4328551)
    expected_cov <- c(0.114393721, 0.009406189, 0.009406189, 0.114393721)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE)
    # Prediction for only new random effects
    group_data_pred_out = cbind(c(m+1,m+1,m+1),c(length(group2)+1,length(group2)+2,length(group2)+1))
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred_out,
                             cov_pars = c(0.1,1,2), predict_cov_mat = TRUE)
    expected_mu <- c(rep(0,3))
    expected_cov <- c(3.1, 1.0, 3.0, 1.0, 3.1, 1.0, 3.0, 1.0, 3.1)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE)
    
    ## Two crossed random effects and a random slope
    y <- Z1%*%b1 + Z2%*%b2 + Z3%*%b3 + xi
    gp_model <- fitGPModel(group_data = cbind(group,group2),
                           group_rand_coef_data = x,
                           ind_effect_group_rand_coef = 1,
                           y = y,
                           params = list(optimizer_cov = "fisher_scoring", maxit=5, std_dev = TRUE))
    expected_values <- c(0.49554952, 0.02546769, 1.24880860, 0.18983953, 1.05505134, 0.22337199, 1.13840014, 0.17950490)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 5)
    
    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
    first_occurences_1 <- match(unique(group), group)
    first_occurences_2 <- match(unique(group2), group2)
    pred_random_effects <- all_training_data_random_effects[first_occurences_1,1]
    pred_random_slopes <- all_training_data_random_effects[first_occurences_1,3]
    pred_random_effects_crossed <- all_training_data_random_effects[first_occurences_2,2] 
    group_unique <- unique(group)
    group_data_pred = cbind(group_unique,rep(-1,length(group_unique)))
    x_pr = rep(0,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr)
    expect_lt(sum(abs(pred_random_effects - preds$mu)),TOL_STRICT)
    # Check whether random slopes are correct
    x_pr = rep(1,length(group_unique))
    preds2 <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr)
    expect_lt(sum(abs(pred_random_slopes - (preds2$mu-preds$mu))),TOL_STRICT)
    # Check whether crossed random effects are correct
    group_unique <- unique(group2)
    group_data_pred = cbind(rep(-1,length(group_unique)),group_unique)
    x_pr = rep(0,length(group_unique))
    preds <- predict(gp_model, group_data_pred=group_data_pred, group_rand_coef_data_pred=x_pr)
    expect_lt(sum(abs(pred_random_effects_crossed - preds$mu)),TOL_STRICT)
    
    # Prediction
    gp_model <- GPModel(group_data = cbind(group,group2),
                        group_rand_coef_data = x, ind_effect_group_rand_coef = 1)
    group_data_pred = cbind(c(1,1,m+1),c(2,1,length(group2)+1))
    group_rand_coef_data_pred = c(0,10,0.3)
    expect_error(gp_model$predict(group_data_pred = group_data_pred,
                                  cov_pars = c(0.1,1,2,1.5), y=y))# random slope data not provided
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred,
                             group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.1,1,2,1.5), predict_cov_mat = TRUE)
    expected_mu <- c(0.7579961, -0.2868530, 0.000000000)
    expected_cov <- c(0.11534086, -0.01988167, 0.0000000, -0.01988167, 2.4073302,
                      0.0000000, 0.0000000, 0.0000000, 3.235)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOLERANCE)
    # Predict variances
    pred <- gp_model$predict(y = y, group_data_pred=group_data_pred,
                             group_rand_coef_data_pred=group_rand_coef_data_pred,
                             cov_pars = c(0.1,1,2,1.5), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_cov[c(1,5,9)])),TOLERANCE)
    
    # Gradient descent
    gp_model <- fitGPModel(group_data = cbind(group,group2),
                           group_rand_coef_data = x,
                           ind_effect_group_rand_coef = 1,
                           y = y,
                           params = list(optimizer_cov = "gradient_descent", std_dev = FALSE))
    cov_pars <- c(0.4958303, 1.2493181, 1.0543343, 1.1388604 )
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(),8)
    
    # Nelder-Mead
    gp_model <- fitGPModel(group_data = cbind(group,group2),
                           group_rand_coef_data = x,
                           ind_effect_group_rand_coef = 1,
                           y = y,
                           params = list(optimizer_cov = "nelder_mead", std_dev = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE1)
    # BFGS
    gp_model <- fitGPModel(group_data = cbind(group,group2),
                           group_rand_coef_data = x,
                           ind_effect_group_rand_coef = 1,
                           y = y,
                           params = list(optimizer_cov = "bfgs", std_dev = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
    # Adam
    gp_model <- fitGPModel(group_data = cbind(group,group2),
                           group_rand_coef_data = x,
                           ind_effect_group_rand_coef = 1,
                           y = y,
                           params = list(optimizer_cov = "adam", std_dev = FALSE))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars)),TOLERANCE2)
    
    # Evaluate negative log-likelihood
    nll <- gp_model$neg_log_likelihood(cov_pars=c(0.1,1,2,1.5),y=y)
    expect_lt(abs(nll-2335.803),1E-2)
  })
  
  
  test_that("not constant cluster_id's for grouped random effects ", {
    
    y <- Z1 %*% b1 + xi
    gp_model <- fitGPModel(group_data = group, cluster_ids = cluster_ids,
                           y = y,
                           params = list(optimizer_cov = "fisher_scoring", maxit=100, std_dev = TRUE,
                                         convergence_criterion = "relative_change_in_parameters"))
    expected_values <- c(0.49348532, 0.02326312, 1.22299521, 0.17995161)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    
    # gradient descent
    gp_model <- fitGPModel(group_data = group, cluster_ids = cluster_ids,
                           y = y,
                           params = list(optimizer_cov = "gradient_descent", std_dev = TRUE,
                                         lr_cov = 0.1, use_nesterov_acc = FALSE, maxit = 1000,
                                         convergence_criterion = "relative_change_in_parameters"))
    cov_pars_expected <- c(0.49348532, 0.02326312, 1.22299520, 0.17995161)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_expected)),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 9)
    
    # Predict training data random effects
    all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
    first_occurences <- match(unique(group), group)
    training_data_random_effects <- all_training_data_random_effects[first_occurences] 
    group_unique <- unique(group)
    pred_random_effects <- predict(gp_model, group_data_pred = group_unique, 
                                   cluster_ids_pred = cluster_ids[first_occurences])
    expect_lt(sum(abs(training_data_random_effects - pred_random_effects$mu)),TOL_STRICT)
    
    # Prediction
    group_data_pred = c(1,1,m+1)
    cluster_ids_pred = c(1,3,1)
    gp_model <- GPModel(group_data = group, cluster_ids = cluster_ids)
    expect_error(gp_model$predict(group_data_pred = group_data_pred,
                                  cov_pars = c(0.75,1.25), y=y))# cluster_id's not provided
    pred <- gp_model$predict(y = y, group_data_pred = group_data_pred,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(0.75,1.25), predict_cov_mat = TRUE)
    expected_mu <- c(-0.1514786, 0.000000, 0.000000)
    expected_cov <- c(0.8207547, 0.000000, 0.000000, 0.000000,
                      2.000000, 0.000000, 0.000000, 0.000000, 2.000000)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOL_STRICT)
    
    # cluster_ids can be string 
    cluster_ids_string <- paste0(as.character(cluster_ids),"_s")
    gp_model <- fitGPModel(group_data = group, cluster_ids = cluster_ids_string,
                           y = y,
                           params = list(optimizer_cov = "fisher_scoring", maxit=100, std_dev = TRUE,
                                         convergence_criterion = "relative_change_in_parameters"))
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-cov_pars_expected)),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 6)
    # Prediction
    group_data_pred = c(1,1,m+1)
    cluster_ids_pred_string = paste0(as.character(c(1,3,1)),"_s")
    pred <- gp_model$predict(y = y, group_data_pred = group_data_pred,
                             cluster_ids_pred = cluster_ids_pred_string,
                             cov_pars = c(0.75,1.25), predict_cov_mat = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$cov)-expected_cov)),TOL_STRICT)
    
    # Prediction when cluster_ids are not in ascending order
    group_data_pred = c(1,1,m,m)
    cluster_ids_pred = c(2,2,1,2)
    gp_model <- GPModel(group_data = group, cluster_ids = cluster_ids)
    pred <- gp_model$predict(y = y, group_data_pred = group_data_pred,
                             cluster_ids_pred = cluster_ids_pred,
                             cov_pars = c(0.75,1.25), predict_var = TRUE)
    expected_mu <- c(rep(0,3), 1.179557)
    expected_var <- c(rep(2,3), 0.8207547)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOL_STRICT)
    # same thing when cluster_ids are strings
    group_data_pred = c(1,1,m,m)
    cluster_ids_pred_string = paste0(as.character(c(2,2,1,2)),"_s")
    gp_model <- GPModel(group_data = group, cluster_ids = cluster_ids_string)
    pred <- gp_model$predict(y = y, group_data_pred = group_data_pred,
                             cluster_ids_pred = cluster_ids_pred_string,
                             cov_pars = c(0.75,1.25), predict_var = TRUE)
    expect_lt(sum(abs(pred$mu-expected_mu)),TOL_STRICT)
    expect_lt(sum(abs(as.vector(pred$var)-expected_var)),TOL_STRICT)

    # cluster_ids and random coefficients
    y <- Z1%*%b1 + Z3%*%b3 + xi
    gp_model <- fitGPModel(group_data = group,
                           cluster_ids = cluster_ids,
                           group_rand_coef_data = x,
                           ind_effect_group_rand_coef = 1,
                           y = y,
                           params = list(optimizer_cov = "gradient_descent", std_dev = FALSE,
                                         lr_cov = 0.1, use_nesterov_acc = TRUE, maxit = 1000,
                                         convergence_criterion = "relative_change_in_parameters"))
    expected_values <- c(0.4927786, 1.2565095, 1.1333656)
    expect_lt(sum(abs(as.vector(gp_model$get_cov_pars())-expected_values)),TOL_STRICT)
    expect_equal(gp_model$get_num_optim_iter(), 14)
    
  })
  
}
