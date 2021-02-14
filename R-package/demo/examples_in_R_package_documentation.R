library(gpboost)
data(GPBoost_data, package = "gpboost")

########################################################
## Linear random effects and Gaussian process models
########################################################

#--------------------Grouped random effects model: single-level random effect----------------
gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
fit(gp_model, y = y, params = list(std_dev = TRUE))
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(group_data = group_data[,1], y = y, likelihood="gaussian",
                       params = list(std_dev = TRUE))
summary(gp_model)
# Make predictions
pred <- predict(gp_model, group_data_pred = group_data_test[,1], predict_var = TRUE)
pred$mu # Predicted mean
pred$var # Predicted variances
# Also predict covariance matrix
pred <- predict(gp_model, group_data_pred = group_data_test[,1], predict_cov_mat = TRUE)
pred$mu # Predicted mean
pred$cov # Predicted covariance


#--------------------Mixed effects model: random effects and linear fixed effects----------------
X1 <- cbind(rep(1,length(y)),X) # Add intercept column
gp_model <- fitGPModel(group_data = group_data[,1], likelihood="gaussian",
                       y = y, X = X1, params = list(std_dev = TRUE))
summary(gp_model)


#--------------------Two crossed random effects and a random slope----------------
gp_model <- fitGPModel(group_data = group_data, likelihood="gaussian",
                       group_rand_coef_data = X[,2],
                       ind_effect_group_rand_coef = 1,
                       y = y, params = list(std_dev = TRUE))
summary(gp_model)


#--------------------Gaussian process model----------------
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                    likelihood="gaussian")
fit(gp_model, y = y, params = list(std_dev = TRUE))
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                       likelihood="gaussian", y = y, params = list(std_dev = TRUE))
summary(gp_model)
# Make predictions
pred <- predict(gp_model, gp_coords_pred = coords_test, predict_cov_mat = TRUE)
# Predicted (posterior/conditional) mean of GP
pred$mu
# Predicted (posterior/conditional) covariance matrix of GP
pred$cov


#--------------------Gaussian process model with linear mean function----------------
X1 <- cbind(rep(1,length(y)),X) # Add intercept column
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                       likelihood="gaussian", y = y, X=X1, params = list(std_dev = TRUE))
summary(gp_model)


#--------------------Gaussian process model with Vecchia approximation----------------
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                    likelihood="gaussian", vecchia_approx = TRUE, num_neighbors = 30)
fit(gp_model, y = y)
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                       vecchia_approx = TRUE, num_neighbors = 30,
                       likelihood="gaussian", y = y)
summary(gp_model)


#--------------------Gaussian process model with random coefficents----------------
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                    gp_rand_coef_data = X[,2], likelihood = "gaussian")
fit(gp_model, y = y, params = list(std_dev = TRUE))
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                       gp_rand_coef_data = X[,2], y=y,
                       likelihood = "gaussian", params = list(std_dev = TRUE))
summary(gp_model)


#--------------------Combine Gaussian process with grouped random effects----------------
gp_model <- fitGPModel(group_data = group_data,
                       gp_coords = coords, cov_function = "exponential",
                       likelihood = "gaussian", y = y, params = list(std_dev = TRUE))
summary(gp_model)


#--------------------Saving a GPModel and loading it from a file----------------
gp_model <- fitGPModel(group_data = group_data[,1], y = y, likelihood="gaussian")
pred <- predict(gp_model, group_data_pred = group_data_test[,1], predict_var = TRUE)
# Save model to file
filename <- tempfile(fileext = ".RData")
saveGPModel(gp_model,filename = filename)
# Load from file and make predictions again
gp_model_loaded <- loadGPModel(filename = filename)
pred_loaded <- predict(gp_model_loaded, group_data_pred = group_data_test[,1], predict_var = TRUE)
# Check equality
pred$mu - pred_loaded$mu
pred$var - pred_loaded$var





#######################
## GPBoost algorithm
#######################

#--------------------Combine tree-boosting and grouped random effects model----------------
# Create random effects model
gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
# The default optimizer for covariance parameters for Gaussian data is Fisher scoring.
# For non-Gaussian data, gradient descent is used.
# Optimizer properties can be changed as follows:
# re_params <- list(optimizer_cov = "gradient_descent", use_nesterov_acc = TRUE)
# gp_model$set_optim_params(params=re_params)
# Use trace = TRUE to monitor convergence:
# re_params <- list(trace = TRUE)
# gp_model$set_optim_params(params=re_params)

# Train model
bst <- gpboost(data = X,
               label = y,
               gp_model = gp_model,
               nrounds = 16,
               learning_rate = 0.05,
               max_depth = 6,
               min_data_in_leaf = 5,
               objective = "regression_l2",
               verbose = 0)

# Same thing using the gpb.train function
dtrain <- gpb.Dataset(data = X, label = y)
bst <- gpb.train(data = dtrain,
                 gp_model = gp_model,
                 nrounds = 16,
                 learning_rate = 0.05,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 verbose = 0)
# Estimated random effects model
summary(gp_model)
# Make predictions
pred <- predict(bst, data = X_test, group_data_pred = group_data_test[,1],
                predict_var= TRUE)
pred$random_effect_mean # Predicted mean
pred$random_effect_cov # Predicted variances
pred$fixed_effect # Predicted fixed effect from tree ensemble
# Sum them up to otbain a single prediction
pred$random_effect_mean + pred$fixed_effect


#--------------------Combine tree-boosting and Gaussian process model----------------
# Create Gaussian process model
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                    likelihood = "gaussian")
# Train model
bst <- gpboost(data = X,
               label = y,
               gp_model = gp_model,
               nrounds = 8,
               learning_rate = 0.1,
               max_depth = 6,
               min_data_in_leaf = 5,
               objective = "regression_l2",
               verbose = 0)

dtrain <- gpb.Dataset(data = X, label = y)
bst <- gpb.train(data = dtrain,
                 gp_model = gp_model,
                 nrounds = 16,
                 learning_rate = 0.05,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 verbose = 0)
# Estimated random effects model
summary(gp_model)
# Make predictions
pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                predict_cov_mat =TRUE)
pred$random_effect_mean # Predicted (posterior) mean of GP
pred$random_effect_cov # Predicted (posterior) covariance matrix of GP
pred$fixed_effect # Predicted fixed effect from tree ensemble
# Sum them up to otbain a single prediction
pred$random_effect_mean + pred$fixed_effect


#--------------------Using validation data-------------------------
set.seed(1)
train_ind <- sample.int(length(y),size=250)
dtrain <- gpb.Dataset(data = X[train_ind,], label = y[train_ind])
dtest <- gpb.Dataset.create.valid(dtrain, data = X[-train_ind,], label = y[-train_ind])
valids <- list(test = dtest)
gp_model <- GPModel(group_data = group_data[train_ind,1], likelihood="gaussian")
# Need to set prediction data for gp_model
gp_model$set_prediction_data(group_data_pred = group_data[-train_ind,1])
# Training with validation data and use_gp_model_for_validation = TRUE
bst <- gpb.train(data = dtrain,
                 gp_model = gp_model,
                 nrounds = 100,
                 learning_rate = 0.05,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 verbose = 1,
                 valids = valids,
                 early_stopping_rounds = 10,
                 use_gp_model_for_validation = TRUE)
print(paste0("Optimal number of iterations: ", bst$best_iter,
             ", best test error: ", bst$best_score))
# Plot validation error
val_error <- unlist(bst$record_evals$test$l2$eval)
plot(1:length(val_error), val_error, type="l", lwd=2, col="blue",
     xlab="iteration", ylab="Validation error", main="Validation error vs. boosting iteration")


#--------------------Do Newton updates for tree leaves---------------
# Note: run the above examples first
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
                 use_gp_model_for_validation = FALSE,
                 leaves_newton_update = TRUE)
print(paste0("Optimal number of iterations: ", bst$best_iter,
             ", best test error: ", bst$best_score))
# Plot validation error
val_error <- unlist(bst$record_evals$test$l2$eval)
plot(1:length(val_error), val_error, type="l", lwd=2, col="blue",
     xlab="iteration", ylab="Validation error", main="Validation error vs. boosting iteration")


#--------------------GPBoostOOS algorithm: GP parameters estimated out-of-sample----------------
# Create random effects model and dataset
gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
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
                use_gp_model_for_validation = TRUE,
                fit_GP_cov_pars_OOS = TRUE)
print(paste0("Optimal number of iterations: ", cvbst$best_iter))
# Estimated random effects model
# Note: ideally, one would have to find the optimal combination of
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


#--------------------Cross validation-------------------------
# Create random effects model and dataset
gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
dtrain <- gpb.Dataset(X, label = y)
params <- list(learning_rate = 0.05,
               max_depth = 6,
               min_data_in_leaf = 5,
               objective = "regression_l2")
# Run CV
cvbst <- gpb.cv(params = params,
                data = dtrain,
                gp_model = gp_model,
                nrounds = 100,
                nfold = 4,
                eval = "l2",
                early_stopping_rounds = 5,
                use_gp_model_for_validation = TRUE)
print(paste0("Optimal number of iterations: ", cvbst$best_iter,
             ", best test error: ", cvbst$best_score))


#--------------------Saving a booster with a gp_model and loading it from a file----------------
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

