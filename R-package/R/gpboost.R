#' @title Train a GPBoost model
#' @name gpboost
#' @description Simple interface for training a GPBoost model.
#' @inheritParams gpb_shared_params
#' @param data A matrix with covariate (=feature) data or a \code{gpb.Dataset} object for training
#' @param ... Additional arguments passed to \code{\link{gpb.train}}. See the documentation (help) of \code{\link{gpb.train}}.
#' @return a trained booster model \code{gpb.Booster}.
#'
#' @examples
#' ## SEE ALSO THE HELP OF 'gpb.train' FOR MORE EXAMPLES
#' \dontrun{
#' library(gpboost)
#' 
#' #--------------------Example without a Gaussian process or random effects model--------------
#' # Non-linear function for simulation
#' f1d <- function(x) 1.7*(1/(1+exp(-(x-0.5)*20))+0.75*x)
#' x <- seq(from=0,to=1,length.out=200)
#' plot(x,f1d(x),type="l",lwd=2,col="red",main="Mean function")
#' # Function that simulates data. Two covariates of which only one has an effect
#' sim_data <- function(n){
#'   X=matrix(runif(2*n),ncol=2)
#'   # mean function plus noise
#'   y=f1d(X[,1])+rnorm(n,sd=0.1)
#'   return(list(X=X,y=y))
#' }
#' # Simulate data
#' n <- 1000
#' set.seed(1)
#' data <- sim_data(2 * n)
#' Xtrain <- data$X[1:n,]
#' ytrain <- data$y[1:n]
#' Xtest <- data$X[1:n + n,]
#' ytest <- data$y[1:n + n]
#'
#' # Train model
#' print("Train boosting model")
#' bst <- gpboost(data = Xtrain,
#'                label = ytrain,
#'                nrounds = 40,
#'                learning_rate = 0.1,
#'                max_depth = 6,
#'                min_data_in_leaf = 5,
#'                objective = "regression_l2",
#'                verbose = 0)
#'
#' # Make predictions and compare fit to truth
#' x <- seq(from=0,to=1,length.out=200)
#' Xtest_plot <- cbind(x,rep(0,length(x)))
#' pred_plot <- predict(bst, data = Xtest_plot)
#' plot(x,f1d(x),type="l",ylim = c(-0.25,3.25), col = "red", lwd = 2,
#'      main = "Comparison of true and fitte value")
#' lines(x,pred_plot, col = "blue", lwd = 2)
#' legend("bottomright", legend = c("truth", "fitted"),
#'        lwd=2, col = c("red", "blue"), bty = "n")
#' 
#' # Prediction accuracy       
#' pred <- predict(bst, data = Xtest)
#' err <- mean((ytest-pred)^2)
#' print(paste("test-RMSE =", err))
#'
#' 
#' #--------------------Combine tree-boosting and Gaussian process model----------------
#' # Simulate data
#' # Function for non-linear mean. Two covariates of which only one has an effect
#' f1d <- function(x) 1.7*(1/(1+exp(-(x-0.5)*20))+0.75*x)
#' set.seed(2)
#' n <- 200 # number of samples
#' X <- matrix(runif(2*n),ncol=2)
#' y <- f1d(X[,1]) # mean
#' # Add Gaussian process
#' sigma2_1 <- 1^2 # marginal variance of GP
#' rho <- 0.1 # range parameter
#' sigma2 <- 0.1^2 # error variance
#' coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
#' D <- as.matrix(dist(coords))
#' Sigma <- sigma2_1*exp(-D/rho)+diag(1E-20,n)
#' C <- t(chol(Sigma))
#' b_1 <- rnorm(n) # simulate random effect
#' eps <- C %*% b_1
#' xi <- sqrt(sigma2) * rnorm(n) # simulate error term
#' y <- y + eps + xi # add random effects and error to data
#' # Create Gaussian process model
#' gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
#' # Default optimizer for covariance parameters is Fisher scoring.
#' # This can be changed as follows:
#' # re_params <- list(optimizer_cov = "gradient_descent", lr_cov = 0.05,
#' #                   use_nesterov_acc = TRUE, acc_rate_cov = 0.5)
#' # gp_model$set_optim_params(params=re_params)
#'
#' # Train model
#' print("Train boosting with Gaussian process model")
#' bst <- gpboost(data = X,
#'                label = y,
#'                gp_model = gp_model,
#'                nrounds = 8,
#'                learning_rate = 0.1,
#'                max_depth = 6,
#'                min_data_in_leaf = 5,
#'                objective = "regression_l2",
#'                verbose = 0)
#' print("Estimated random effects model")
#' summary(gp_model)
#'
#' # Make predictions
#' set.seed(1)
#' ntest <- 5
#' Xtest <- matrix(runif(2*ntest),ncol=2)
#' # prediction locations (=features) for Gaussian process
#' coords_test <- cbind(runif(ntest),runif(ntest))/10
#' pred <- predict(bst, data = Xtest, gp_coords_pred = coords_test,
#'                 predict_cov_mat =TRUE)
#' print("Predicted (posterior) mean of GP")
#' pred$random_effect_mean
#' print("Predicted (posterior) covariance matrix of GP")
#' pred$random_effect_cov
#' print("Predicted fixed effect from tree ensemble")
#' pred$fixed_effect
#' }
#' @export
gpboost <- function(data,
                    label = NULL,
                    nrounds = 100,
                    obj = NULL,
                    gp_model = NULL,
                    train_gp_model_cov_pars = TRUE,
                    params = list(),
                    valids = list(),
                    early_stopping_rounds = NULL,
                    use_gp_model_for_validation = FALSE,
                    weight = NULL,
                    verbose = 1,
                    ...) {
  
  # Set data to a temporary variable
  dtrain <- data
  if (nrounds <= 0) {
    stop("nrounds should be greater than zero")
  }
  # Check whether data is gpb.Dataset, if not then create gpb.Dataset manually
  if (!gpb.is.Dataset(dtrain)) {
    dtrain <- gpb.Dataset(data, label = label, weight = weight)
  }
  
  # Set validation as oneself
  if (length(valids)) valids <- list()
  if (verbose > 0) {
    valids$train = dtrain
  }
  
  # Train a model using the regular way
  bst <- gpb.train(params = params, data = dtrain, nrounds = nrounds,
                   obj = obj, valids = valids, verbose = verbose,
                   early_stopping_rounds = early_stopping_rounds,
                   gp_model = gp_model, train_gp_model_cov_pars = train_gp_model_cov_pars,
                   use_gp_model_for_validation = use_gp_model_for_validation, ...)
  
  # Return booster
  return(bst)
}

