library(gpboost)

#--------------------Combine tree-boosting and grouped random effects model----------------
# Simulate data
n <- 5000 # number of samples
m <- 500  # number of groups
# Simulate grouped random effects
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
b1 <- rnorm(m)
eps <- b1[group]
# Simulate fixed effects
# Function for non-linear mean. Two covariates of which only one has an effect
f1d <- function(x) 1.7*(1/(1+exp(-(x-0.5)*20))+0.75*x)
X <- matrix(runif(2*n),ncol=2)
f <- f1d(X[,1]) # mean
# Observed data
xi <- sqrt(0.01) * rnorm(n) # simulate error term
y <- f + eps + xi 

#--------------------Training using gpboost----------------
# Create random effects model
gp_model <- GPModel(group_data = group)
# The default optimizer for covariance parameters is Fisher scoring.
# This can be changed to e.g. Nesterov accelerated gradient descent as follows:
# re_params <- list(trace=TRUE,optimizer_cov="gradient_descent",
#                   lr_cov = 0.05, use_nesterov_acc = TRUE)
# gp_model$set_optim_params(params=re_params)

# print("Train boosting with random effects model")
bst <- gpboost(data = X,
               label = y,
               gp_model = gp_model,
               nrounds = 15,
               learning_rate = 0.05,
               max_depth = 6,
               min_data_in_leaf = 5,
               objective = "regression_l2",
               verbose = 0,
               leaves_newton_update = FALSE)

# Same thing using the gpb.train function
print("Training with gpb.train")
dtrain <- gpb.Dataset(data = X, label = y)
bst <- gpb.train(data = dtrain,
                 gp_model = gp_model,
                 nrounds = 15,
                 learning_rate = 0.05,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 verbose = 0)

print("Estimated random effects model")
summary(gp_model)


#--------------------Prediction using gpboost--------------
group_test <- 1:m
x_test <- seq(from=0,to=1,length.out=m)
Xtest <- cbind(x_test,rep(0,length(x_test)))
pred <- predict(bst, data = Xtest, group_data_pred = group_test)

# Compare fit to truth: random effects
pred_random_effect <- pred$random_effect_mean
plot(b1, pred_random_effect, xlab="truth", ylab="predicted",
     main="Comparison of true and predicted random effects")
abline(a=0,b=1)
# Compare fit to truth: fixed effect (mean function)
pred_mean <- pred$fixed_effect
x <- seq(from=0,to=1,length.out=200)
plot(x,f1d(x),type="l",ylim = c(-0.25,3.25), col = "red", lwd = 2,
     main = "Comparison of true and fitted value")
points(x_test,pred_mean, col = "blue", lwd = 2)
legend("bottomright", legend = c("truth", "fitted"),
       lwd=2, col = c("red", "blue"), bty = "n")


#--------------------Using validation set-------------------------
set.seed(1)
train_ind <- sample.int(n,size=900)
dtrain <- gpb.Dataset(data = X[train_ind,], label = y[train_ind])
dtest <- gpb.Dataset.create.valid(dtrain, data = X[-train_ind,], label = y[-train_ind])
valids <- list(test = dtest)
gp_model <- GPModel(group_data = group[train_ind])

# Include random effect predictions for validation (=default)
gp_model <- GPModel(group_data = group[train_ind])
gp_model$set_prediction_data(group_data_pred = group[-train_ind])
print("Training with validation data and use_gp_model_for_validation = TRUE ")
bst <- gpb.train(data = dtrain,
                 gp_model = gp_model,
                 nrounds = 100,
                 learning_rate = 0.05,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 verbose = 1,
                 valids = valids,
                 early_stopping_rounds = 5,
                 use_gp_model_for_validation = TRUE)
print(paste0("Optimal number of iterations: ", bst$best_iter,
             ", best test error: ", bst$best_score))
# Plot validation error
val_error <- unlist(bst$record_evals$test$l2$eval)
plot(1:length(val_error), val_error, type="l", lwd=2, col="blue",
     xlab="iteration", ylab="Validation error", main="Validation error vs. boosting iteration")

# Do not include random effect predictions for validation (observe the higher test error)
print("Training with validation data and use_gp_model_for_validation = FALSE")
bst <- gpb.train(data = dtrain,
                 gp_model = gp_model,
                 nrounds = 100,
                 learning_rate = 0.05,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 verbose = 1,
                 valids = valids,
                 early_stopping_rounds = 5,
                 use_gp_model_for_validation = FALSE)
print(paste0("Optimal number of iterations: ", bst$best_iter,
             ", best test error: ", bst$best_score))
# Plot validation error
val_error <- unlist(bst$record_evals$test$l2$eval)
plot(1:length(val_error), val_error, type="l", lwd=2, col="blue",
     xlab="iteration", ylab="Validation error", main="Validation error vs. boosting iteration")

#--------------------Do Newton updates for tree leaves---------------
print("Training with Newton updates for tree leaves")
bst <- gpb.train(data = dtrain,
                 gp_model = gp_model,
                 nrounds = 100,
                 learning_rate = 0.05,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 verbose = 1,
                 valids = valids,
                 early_stopping_rounds = 5,
                 use_gp_model_for_validation = TRUE,
                 leaves_newton_update = TRUE)
print(paste0("Optimal number of iterations: ", bst$best_iter,
             ", best test error: ", bst$best_score))
# Plot validation error
val_error <- unlist(bst$record_evals$test$l2$eval)
plot(1:length(val_error), val_error, type="l", lwd=2, col="blue",
     xlab="iteration", ylab="Validation error", main="Validation error vs. boosting iteration")

# Using gpboost function
bst <- gpboost(data = dtrain,
               gp_model = gp_model,
               nrounds = 1,
               objective = "regression_l2",
               verbose = 0,
               leaves_newton_update = TRUE)


#--------------------Combine tree-boosting and Gaussian process model----------------
# Simulate data
# Function for non-linear mean. Two covariates of which only one has an effect
f1d <- function(x) 1.7*(1/(1+exp(-(x-0.5)*20))+0.75*x)
set.seed(2)
n <- 200 # number of samples
X <- matrix(runif(2*n),ncol=2)
y <- f1d(X[,1]) # mean
# Add Gaussian process
sigma2_1 <- 1^2 # marginal variance of GP
rho <- 0.1 # range parameter
sigma2 <- 0.1^2 # error variance
coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
D <- as.matrix(dist(coords))
Sigma <- sigma2_1*exp(-D/rho)+diag(1E-20,n)
C <- t(chol(Sigma))
b_1 <- rnorm(n) # simulate random effect
eps <- C %*% b_1
xi <- sqrt(sigma2) * rnorm(n) # simulate error term
y <- y + eps + xi # add random effects and error to data

# Create Gaussian process model
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
# The default optimizer for covariance parameters is Fisher scoring.
# This can be changed to e.g. Nesterov accelerated gradient descent as follows:
# re_params <- list(optimizer_cov = "gradient_descent", lr_cov = 0.05,
#                   use_nesterov_acc = TRUE, acc_rate_cov = 0.5)
# gp_model$set_optim_params(params=re_params)

# Train model
print("Train boosting with Gaussian process model")
bst <- gpboost(data = X,
               label = y,
               gp_model = gp_model,
               nrounds = 8,
               learning_rate = 0.1,
               max_depth = 6,
               min_data_in_leaf = 5,
               objective = "regression_l2",
               verbose = 0)
print("Estimated random effects model")
summary(gp_model)

# Make predictions
set.seed(1)
ntest <- 5
Xtest <- matrix(runif(2*ntest),ncol=2)
# prediction locations (=features) for Gaussian process
coords_test <- cbind(runif(ntest),runif(ntest))/10
pred <- predict(bst, data = Xtest, gp_coords_pred = coords_test,
                predict_cov_mat = TRUE)
print("Predicted (posterior) mean of GP")
pred$random_effect_mean
print("Predicted (posterior) covariance matrix of GP")
pred$random_effect_cov
print("Predicted fixed effect from tree ensemble")
pred$fixed_effect


#--------------------GPBoostOOS algorithm: GP parameters estimated out-of-sample----------------
# Simulate data
f1d <- function(x) 1.7*(1/(1+exp(-(x-0.5)*20))+0.75*x)
set.seed(1)
n <- 1000 # number of samples
X <- matrix(runif(2*n),ncol=2)
y <- f1d(X[,1]) # mean
# Add grouped random effects
m <- 25 # number of categories / levels for grouping variable
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
sigma2_1 <- 1^2 # random effect variance
sigma2 <- 0.1^2 # error variance
# incidence matrix relating grouped random effects to samples
Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
b1 <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
eps <- Z1 %*% b1
xi <- sqrt(sigma2) * rnorm(n) # simulate error term
y <- y + eps + xi # add random effects and error to data
# Create random effects model and dataset
gp_model <- GPModel(group_data = group)
dtrain <- gpb.Dataset(X, label = y)
params <- list(learning_rate = 0.05,
               max_depth = 6,
               min_data_in_leaf = 5,
               objective = "regression_l2")
# Stage 1: run cross-validation to (i) determine to optimal number of iterations
#           and (ii) to estimate the GPModel on the out-of-sample data
cvbst <- gpb.cv(params = params,
              data = dtrain,
              gp_model = gp_model,
              nrounds = 100,
              nfold = 4,
              eval = "l2",
              early_stopping_rounds = 5,
              fit_GP_cov_pars_OOS = TRUE)
print(paste0("Optimal number of iterations: ", cvbst$best_iter))
# Fitted model (note: ideally, one would have to find the optimal combination of 
#               other tuning parameters such as the learning rate, tree depth, etc.)
summary(gp_model)
# Stage 2: Train tree-boosting model while holding the GPModel fix
bst <- gpb.train(data = dtrain,
                 gp_model = gp_model,
                 nrounds = cvbst$best_iter,
                 learning_rate = 0.05,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 verbose = 0,
                 train_gp_model_cov_pars = FALSE)
# The GPModel has not changed:
summary(gp_model)


#--------------------Saving a booster with a gp_model and loading it from a file----------------
data(GPBoost_data, package = "gpboost")
# Train model and make prediction
gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
bst <- gpboost(data = X,
               label = y,
               gp_model = gp_model,
               nrounds = 16,
               learning_rate = 0.05,
               max_depth = 6,
               min_data_in_leaf = 5,
               objective = "regression_l2",
               verbose = 0)
pred <- predict(bst, data = X_test, group_data_pred = group_data_test[,1],
                predict_var= TRUE)
# Save model to file
filename <- tempfile(fileext = ".RData")
gpb.save(bst,filename = filename)
# Load from file and make predictions again
bst_loaded <- gpb.load(filename = filename)
pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test[,1],
                       predict_var= TRUE)
# Check equality
pred$fixed_effect - pred_loaded$fixed_effect
pred$random_effect_mean - pred_loaded$random_effect_mean
pred$random_effect_cov - pred_loaded$random_effect_cov
