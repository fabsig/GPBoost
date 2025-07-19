## Examples on how to use GPBoost for 
##  - the Grabit model of Sigrist and Hirnschall (2019)
##  - mean-scale regression / heteroscedastic regression with a Gaussian likelihood
## Author: Fabio Sigrist

library(gpboost)

####################
## Grabit Model
####################
# Function for non-linear mean
sim_friedman3 <- function(n, n_irrelevant=5){
        X <- matrix(runif(4*n),ncol=4)
        X[,1] <- 100*X[,1]
        X[,2] <- X[,2]*pi*(560-40)+40*pi
        X[,4] <- X[,4]*10+1
        f <- sqrt(10)*atan((X[,2]*X[,3]-1/(X[,2]*X[,4]))/X[,1])
        X <- cbind(rep(1,n),X)
        if(n_irrelevant>0) X <- cbind(X,matrix(runif(n_irrelevant*n),ncol=n_irrelevant))
        return(list(X=X,f=f))
}

# simulate data
n <- 10000
set.seed(1)
sim_train <- sim_friedman3(n=n)
sim_test <- sim_friedman3(n=n)
X <- sim_train$X
lp <- sim_train$f
X_test <- sim_test$X
lp_test <- sim_test$f
y <- rnorm(n,mean=lp,sd=1)
y_test <- rnorm(n,mean=lp_test,sd=1)
# apply censoring
yu <- 5
yl <- 3.5
y[y>=yu] <- yu
y[y<=yl] <- yl
# censoring fractions
sum(y==yu) / n
sum(y==yl) / n

# train model and make predictions
dtrain <- gpb.Dataset(data = X, label = y)
bst <- gpb.train(data = dtrain, nrounds = 100, objective = "tobit",
                 verbose = 0, yl = yl, yu = yu, sigma = 1)
y_pred <- predict(bst, data = X_test)
# mean square error (approx. 1.0 for n=10'000)
print(paste0("Test error of Grabit: ", mean((y_pred - y_test)^2)))
# compare to standard least squares gradient boosting (approx. 1.5 for n=10'000)
bst <- gpb.train(data = dtrain, nrounds = 100, objective = "regression_l2",
                 verbose = 0)
y_pred_ls <- predict(bst, data = X_test)
print(paste0("Test error of standard least squares gradient boosting: ", mean((y_pred_ls - y_test)^2)))


####################
## Heteroscedastic mean-scale regression
####################
f1d <- function(x) {
  ## Non-linear fixed effects function for simulation
  return( 1 / (1 + exp(-(x - 0.5) * 10)) - 0.5)
}
n <- 1000
p <- 5 # number of predictor variables
# Simulate data
set.seed(10)
X <- matrix(runif(p*n), ncol=p)
f_mean <- f1d(X[,1])
f_stdev <- exp(X[,1]*3-4)
y <- f_mean + f_stdev * rnorm(f_stdev)

dtrain <- gpb.Dataset(data = X, label = y)

## Parameter tuning
param_grid <- list("learning_rate" = c(0.001, 0.01, 0.1, 1), 
                   "min_data_in_leaf" = c(1, 10, 100, 1000),
                   "max_depth" = c(-1), # -1 means no depth limit as we tune 'num_leaves'. Can also additionally tune 'max_depth', e.g., "max_depth" = c(-1, 1, 2, 3, 5, 10)
                   "num_leaves" = 2^(1:10),
                   "lambda_l2" = c(0, 1, 10, 100),
                   "max_bin" = c(250, 500, 1000, min(n,10000)))
metric = "crps_gaussian" # Define metric
set.seed(1)
# Run parameter optimization using random grid search and k-fold CV
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, data = dtrain,
                                              num_try_random = 100, nfold = 5,
                                              nrounds = 1000, early_stopping_rounds = 20,
                                              verbose_eval = 1, metric = metric, cv_seed = 4,
                                              objective = "mean_scale_regression")
print(paste0("Best parameters: ", paste0(unlist(lapply(seq_along(opt_params$best_params), 
                                                       function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=opt_params$best_params, 
                                                       n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", round(opt_params$best_score, digits=3)))

# Train model
params <- list(learning_rate = 0.1, max_depth = -1, num_leaves = 8, max_bin = 250, 
               lambda_l2 = 100, min_data_in_leaf = 100)
bst <- gpb.train(data = dtrain, nrounds = 100, params = params,
                 objective = "mean_scale_regression", verbose = 0)

# Make predictions
npred <- 100
X_test <- matrix(0, ncol=p, nrow=npred)
X_test[,1] <- seq(0,1,length.out=npred)
y_pred <- predict(bst, data = X_test)
pred_mean <- y_pred$pred_mean
pred_sd <- sqrt(y_pred$pred_var)

# Plot data and predictions
plot(X[,1], y)
lines(X_test[,1], pred_mean, col = "red", lwd = 3)
lines(X_test[,1], pred_mean + 2 * pred_sd, col = "red", lwd = 2 ,lty=2)
lines(X_test[,1], pred_mean - 2 * pred_sd, col = "red", lwd = 2 ,lty=2)

