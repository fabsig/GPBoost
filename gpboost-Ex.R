pkgname <- "gpboost"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
options(pager = "console")
library('gpboost')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("GPModel")
### * GPModel

flush(stderr()); flush(stdout())

### Name: GPModel
### Title: Create a 'GPModel' object
### Aliases: GPModel

### ** Examples

# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples

data(GPBoost_data, package = "gpboost")

#--------------------Grouped random effects model: single-level random effect----------------
gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")

#--------------------Gaussian process model----------------
gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                    likelihood="gaussian")

#--------------------Combine Gaussian process with grouped random effects----------------
gp_model <- GPModel(group_data = group_data,
                    gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                    likelihood="gaussian")



cleanEx()
nameEx("dim")
### * dim

flush(stderr()); flush(stdout())

### Name: dim.gpb.Dataset
### Title: Dimensions of an 'gpb.Dataset'
### Aliases: dim.gpb.Dataset

### ** Examples

## No test: 
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)

stopifnot(nrow(dtrain) == nrow(train$data))
stopifnot(ncol(dtrain) == ncol(train$data))
stopifnot(all(dim(dtrain) == dim(train$data)))
## End(No test)



cleanEx()
nameEx("dimnames.gpb.Dataset")
### * dimnames.gpb.Dataset

flush(stderr()); flush(stdout())

### Name: dimnames.gpb.Dataset
### Title: Handling of column names of 'gpb.Dataset'
### Aliases: dimnames.gpb.Dataset dimnames<-.gpb.Dataset

### ** Examples

## No test: 
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
gpb.Dataset.construct(dtrain)
dimnames(dtrain)
colnames(dtrain)
colnames(dtrain) <- make.names(seq_len(ncol(train$data)))
print(dtrain, verbose = TRUE)
## End(No test)



cleanEx()
nameEx("fit.GPModel")
### * fit.GPModel

flush(stderr()); flush(stdout())

### Name: fit.GPModel
### Title: Fits a 'GPModel'
### Aliases: fit.GPModel

### ** Examples

# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples

## No test: 
data(GPBoost_data, package = "gpboost")
# Add intercept column
X1 <- cbind(rep(1,dim(X)[1]),X)
X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)

#--------------------Grouped random effects model: single-level random effect----------------
gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
fit(gp_model, y = y, X = X1)
summary(gp_model)
# Make predictions
pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
                X_pred = X_test1, predict_var = TRUE)
pred$mu # Predicted mean
pred$var # Predicted variances
# Also predict covariance matrix
pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
                X_pred = X_test1, predict_cov_mat = TRUE)
pred$mu # Predicted mean
pred$cov # Predicted covariance
 
#--------------------Gaussian process model----------------
gp_model <- GPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                    likelihood="gaussian")
fit(gp_model, y = y, X = X1)
summary(gp_model)
# Make predictions
pred <- predict(gp_model, gp_coords_pred = coords_test, 
                X_pred = X_test1, predict_cov_mat = TRUE)
pred$mu # Predicted (posterior) mean of GP
pred$cov # Predicted (posterior) covariance matrix of GP
## End(No test)




cleanEx()
nameEx("fitGPModel")
### * fitGPModel

flush(stderr()); flush(stdout())

### Name: fitGPModel
### Title: Fits a 'GPModel'
### Aliases: fitGPModel

### ** Examples

# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples

## No test: 
data(GPBoost_data, package = "gpboost")
# Add intercept column
X1 <- cbind(rep(1,dim(X)[1]),X)
X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)

#--------------------Grouped random effects model: single-level random effect----------------
gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1,
                       likelihood="gaussian")
summary(gp_model)
# Make predictions
pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
                X_pred = X_test1, predict_var = TRUE)
pred$mu # Predicted mean
pred$var # Predicted variances
# Also predict covariance matrix
pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
                X_pred = X_test1, predict_cov_mat = TRUE)
pred$mu # Predicted mean
pred$cov # Predicted covariance

#--------------------Two crossed random effects and a random slope----------------
gp_model <- fitGPModel(group_data = group_data, likelihood="gaussian",
                       group_rand_coef_data = X[,2],
                       ind_effect_group_rand_coef = 1,
                       y = y, X = X1)
summary(gp_model)

#--------------------Gaussian process model----------------
gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                       likelihood="gaussian", y = y, X = X1)
summary(gp_model)
# Make predictions
pred <- predict(gp_model, gp_coords_pred = coords_test, 
                X_pred = X_test1, predict_cov_mat = TRUE)
pred$mu # Predicted (posterior) mean of GP
pred$cov # Predicted (posterior) covariance matrix of GP

#--------------------Gaussian process model with Vecchia approximation----------------
gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                       gp_approx = "vecchia", num_neighbors = 20,
                       likelihood="gaussian", y = y)
summary(gp_model)

#--------------------Gaussian process model with random coefficients----------------
gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                       gp_rand_coef_data = X[,2], y=y,
                       likelihood = "gaussian")
summary(gp_model)

#--------------------Combine Gaussian process with grouped random effects----------------
gp_model <- fitGPModel(group_data = group_data,
                       gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                       likelihood = "gaussian", y = y, X = X1)
summary(gp_model)

#--------------------Continuous-time RW2 prior for a time series----------------
# The prior is anchored at time t = 0, i.e., b(0) = b'(0) = 0
time <- seq(0, 1, length.out = 100)
y_time <- sin(2 * pi * time) + rnorm(100, sd = 0.2)
X_time <- cbind(Intercept = 1, trend = time) # unpenalized intercept and linear trend
# The covariance parameters are (error variance, GP variance, H), and H = 1.5 is not estimated
gp_model <- fitGPModel(gp_coords = time, cov_function = "hurst", cov_fct_order = 2,
                       likelihood = "gaussian", y = y_time, X = X_time,
                       params = list(init_cov_pars = c(0.1, 1, 1.5),
                                     estimate_cov_par_index = c(1, 1, 0)))
summary(gp_model)
# Continuous-time RW1 prior: cov_fct_order = 1, H fixed to 0.5, and only an intercept
## End(No test)




cleanEx()
nameEx("get_aux_pars.GPModel")
### * get_aux_pars.GPModel

flush(stderr()); flush(stdout())

### Name: get_aux_pars.GPModel
### Title: Get (estimated) auxiliary (additional) parameters of the
###   likelihood
### Aliases: get_aux_pars.GPModel

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
X1 <- cbind(rep(1,dim(X)[1]),X) # Add intercept column
y_pos <- exp(y)
gp_model <- fitGPModel(group_data = group_data[,1], y = y_pos, X = X1, likelihood="gamma")
get_aux_pars(gp_model)
## End(No test)



cleanEx()
nameEx("get_aux_pars")
### * get_aux_pars

flush(stderr()); flush(stdout())

### Name: get_aux_pars
### Title: Get (estimated) auxiliary (additional) parameters of the
###   likelihood
### Aliases: get_aux_pars

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
X1 <- cbind(rep(1,dim(X)[1]),X) # Add intercept column
y_pos <- exp(y)
gp_model <- fitGPModel(group_data = group_data[,1], y = y_pos, X = X1, likelihood="gamma")
get_aux_pars(gp_model)
## End(No test)




cleanEx()
nameEx("get_coef.GPModel")
### * get_coef.GPModel

flush(stderr()); flush(stdout())

### Name: get_coef.GPModel
### Title: Get (estimated) linear regression coefficients
### Aliases: get_coef.GPModel

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
X1 <- cbind(rep(1,dim(X)[1]),X) # Add intercept column
gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
get_coef(gp_model)
## End(No test)



cleanEx()
nameEx("get_coef")
### * get_coef

flush(stderr()); flush(stdout())

### Name: get_coef
### Title: Get (estimated) linear regression coefficients
### Aliases: get_coef

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
X1 <- cbind(rep(1,dim(X)[1]),X) # Add intercept column
gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
get_coef(gp_model)
## End(No test)




cleanEx()
nameEx("get_cov_pars.GPModel")
### * get_cov_pars.GPModel

flush(stderr()); flush(stdout())

### Name: get_cov_pars.GPModel
### Title: Get (estimated) covariance parameters
### Aliases: get_cov_pars.GPModel

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
X1 <- cbind(rep(1,dim(X)[1]),X) # Add intercept column
gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
get_cov_pars(gp_model)
## End(No test)



cleanEx()
nameEx("get_cov_pars")
### * get_cov_pars

flush(stderr()); flush(stdout())

### Name: get_cov_pars
### Title: Get (estimated) covariance parameters
### Aliases: get_cov_pars

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
X1 <- cbind(rep(1,dim(X)[1]),X) # Add intercept column
gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
get_cov_pars(gp_model)
## End(No test)




cleanEx()
nameEx("get_nested_categories")
### * get_nested_categories

flush(stderr()); flush(stdout())

### Name: get_nested_categories
### Title: Auxiliary function to create categorical variables for nested
###   grouped random effects
### Aliases: get_nested_categories

### ** Examples

## No test: 
# Fit a model with Time as categorical fixed effects variables and Diet and Chick
#   as random effects, where Chick is nested in Diet using lme4
chick_nested_diet <- get_nested_categories(ChickWeight$Diet, ChickWeight$Chick)
fixed_effects_matrix <- model.matrix(weight ~ as.factor(Time), data = ChickWeight)
mod_gpb <- fitGPModel(X = fixed_effects_matrix, 
                      group_data = cbind(diet=ChickWeight$Diet, chick_nested_diet), 
                      y = ChickWeight$weight)
summary(mod_gpb)
# This does (almost) the same thing as the following code using lme4:
# mod_lme4 <-  lmer(weight ~ as.factor(Time) + (1 | Diet/Chick), data = ChickWeight, REML = FALSE)
# summary(mod_lme4)
## End(No test)



cleanEx()
nameEx("getinfo")
### * getinfo

flush(stderr()); flush(stdout())

### Name: getinfo
### Title: Get information of an 'gpb.Dataset' object
### Aliases: getinfo getinfo.gpb.Dataset

### ** Examples

## No test: 
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
gpb.Dataset.construct(dtrain)

labels <- gpboost::getinfo(dtrain, "label")
gpboost::setinfo(dtrain, "label", 1 - labels)

labels2 <- gpboost::getinfo(dtrain, "label")
stopifnot(all(labels2 == 1 - labels))
## End(No test)



cleanEx()
nameEx("gpb.Dataset")
### * gpb.Dataset

flush(stderr()); flush(stdout())

### Name: gpb.Dataset
### Title: Construct 'gpb.Dataset' object
### Aliases: gpb.Dataset

### ** Examples

## No test: 
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
data_file <- tempfile(fileext = ".data")
gpb.Dataset.save(dtrain, data_file)
dtrain <- gpb.Dataset(data_file)
gpb.Dataset.construct(dtrain)
## End(No test)



cleanEx()
nameEx("gpb.Dataset.construct")
### * gpb.Dataset.construct

flush(stderr()); flush(stdout())

### Name: gpb.Dataset.construct
### Title: Construct Dataset explicitly
### Aliases: gpb.Dataset.construct

### ** Examples

## No test: 
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
gpb.Dataset.construct(dtrain)
## End(No test)



cleanEx()
nameEx("gpb.Dataset.create.valid")
### * gpb.Dataset.create.valid

flush(stderr()); flush(stdout())

### Name: gpb.Dataset.create.valid
### Title: Construct validation data
### Aliases: gpb.Dataset.create.valid

### ** Examples

## No test: 
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
data(agaricus.test, package = "gpboost")
test <- agaricus.test
dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
## End(No test)



cleanEx()
nameEx("gpb.Dataset.save")
### * gpb.Dataset.save

flush(stderr()); flush(stdout())

### Name: gpb.Dataset.save
### Title: Save 'gpb.Dataset' to a binary file
### Aliases: gpb.Dataset.save

### ** Examples

## No test: 
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
gpb.Dataset.save(dtrain, tempfile(fileext = ".bin"))
## End(No test)



cleanEx()
nameEx("gpb.Dataset.set.categorical")
### * gpb.Dataset.set.categorical

flush(stderr()); flush(stdout())

### Name: gpb.Dataset.set.categorical
### Title: Set categorical feature of 'gpb.Dataset'
### Aliases: gpb.Dataset.set.categorical

### ** Examples

## No test: 
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
data_file <- tempfile(fileext = ".data")
gpb.Dataset.save(dtrain, data_file)
dtrain <- gpb.Dataset(data_file)
gpb.Dataset.set.categorical(dtrain, 1L:2L)
## End(No test)



cleanEx()
nameEx("gpb.Dataset.set.reference")
### * gpb.Dataset.set.reference

flush(stderr()); flush(stdout())

### Name: gpb.Dataset.set.reference
### Title: Set reference of 'gpb.Dataset'
### Aliases: gpb.Dataset.set.reference

### ** Examples

## No test: 
data(agaricus.train, package ="gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
data(agaricus.test, package = "gpboost")
test <- agaricus.test
dtest <- gpb.Dataset(test$data, test = train$label)
gpb.Dataset.set.reference(dtest, dtrain)
## End(No test)



cleanEx()
nameEx("gpb.convert_with_rules")
### * gpb.convert_with_rules

flush(stderr()); flush(stdout())

### Name: gpb.convert_with_rules
### Title: Data preparator for GPBoost datasets with rules (integer)
### Aliases: gpb.convert_with_rules

### ** Examples

## No test: 
data(iris)

str(iris)

new_iris <- gpb.convert_with_rules(data = iris)
str(new_iris$data)

data(iris) # Erase iris dataset
iris$Species[1L] <- "NEW FACTOR" # Introduce junk factor (NA)

# Use conversion using known rules
# Unknown factors become 0, excellent for sparse datasets
newer_iris <- gpb.convert_with_rules(data = iris, rules = new_iris$rules)

# Unknown factor is now zero, perfect for sparse datasets
newer_iris$data[1L, ] # Species became 0 as it is an unknown factor

newer_iris$data[1L, 5L] <- 1.0 # Put back real initial value

# Is the newly created dataset equal? YES!
all.equal(new_iris$data, newer_iris$data)

# Can we test our own rules?
data(iris) # Erase iris dataset

# We remapped values differently
personal_rules <- list(
  Species = c(
    "setosa" = 3L
    , "versicolor" = 2L
    , "virginica" = 1L
  )
)
newest_iris <- gpb.convert_with_rules(data = iris, rules = personal_rules)
str(newest_iris$data) # SUCCESS!
## End(No test)



cleanEx()
nameEx("gpb.cv")
### * gpb.cv

flush(stderr()); flush(stdout())

### Name: gpb.cv
### Title: CV function for number of boosting iterations
### Aliases: gpb.cv

### ** Examples

# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
## No test: 
library(gpboost)
data(GPBoost_data, package = "gpboost")

# Create random effects model and dataset
gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
dtrain <- gpb.Dataset(X, label = y)
params <- list(learning_rate = 0.05,
               max_depth = 6,
               min_data_in_leaf = 5)
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
## End(No test)



cleanEx()
nameEx("gpb.dump")
### * gpb.dump

flush(stderr()); flush(stdout())

### Name: gpb.dump
### Title: Dump GPBoost model to json
### Aliases: gpb.dump

### ** Examples

## No test: 
library(gpboost)
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
data(agaricus.test, package = "gpboost")
test <- agaricus.test
dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
params <- list(objective = "regression", metric = "l2")
valids <- list(test = dtest)
model <- gpb.train(
  params = params
  , data = dtrain
  , nrounds = 10L
  , valids = valids
  , min_data = 1L
  , learning_rate = 1.0
  , early_stopping_rounds = 5L
)
json_model <- gpb.dump(model)
## End(No test)



cleanEx()
nameEx("gpb.get.default.num.threads")
### * gpb.get.default.num.threads

flush(stderr()); flush(stdout())

### Name: gpb.get.default.num.threads
### Title: Get the number of threads used by models for which no number of
###   threads is specified
### Aliases: gpb.get.default.num.threads

### ** Examples

num_threads <- gpb.get.default.num.threads()



cleanEx()
nameEx("gpb.get.eval.result")
### * gpb.get.eval.result

flush(stderr()); flush(stdout())

### Name: gpb.get.eval.result
### Title: Get record evaluation result from booster
### Aliases: gpb.get.eval.result

### ** Examples

## No test: 
# train a regression model
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
data(agaricus.test, package = "gpboost")
test <- agaricus.test
dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
params <- list(objective = "regression", metric = "l2")
valids <- list(test = dtest)
model <- gpb.train(
  params = params
  , data = dtrain
  , nrounds = 5L
  , valids = valids
  , min_data = 1L
  , learning_rate = 1.0
)

# Examine valid data_name values
print(setdiff(names(model$record_evals), "start_iter"))

# Examine valid eval_name values for dataset "test"
print(names(model$record_evals[["test"]]))

# Get L2 values for "test" dataset
gpb.get.eval.result(model, "test", "l2")
## End(No test)



cleanEx()
nameEx("gpb.get.num.threads")
### * gpb.get.num.threads

flush(stderr()); flush(stdout())

### Name: gpb.get.num.threads
### Title: Get the number of threads used for parallelization
### Aliases: gpb.get.num.threads

### ** Examples

num_threads <- gpb.get.num.threads()



cleanEx()
nameEx("gpb.grid.search.tune.parameters")
### * gpb.grid.search.tune.parameters

flush(stderr()); flush(stdout())

### Name: gpb.grid.search.tune.parameters
### Title: Function for choosing tuning parameters
### Aliases: gpb.grid.search.tune.parameters

### ** Examples

# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
## No test: 
library(gpboost)
data(GPBoost_data, package = "gpboost")

n <- length(y)
param_grid <- list("learning_rate" = c(0.001, 0.01, 0.1, 1, 10), 
                   "min_data_in_leaf" = c(1, 10, 100, 1000),
                   "max_depth" = c(-1), 
                   "num_leaves" = 2^(1:10),
                   "lambda_l2" = c(0, 1, 10, 100),
                   "max_bin" = c(250, 500, 1000, min(n,10000)),
                   "line_search_step_length" = c(TRUE, FALSE))
# Note: "max_depth" = c(-1) means no depth limit as we tune 'num_leaves'. 
#    Can also additionally tune 'max_depth', e.g., "max_depth" = c(-1, 1, 2, 3, 5, 10)
metric = "mse" # Define metric
# Note: can also use metric = "test_neg_log_likelihood". 
# See https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
data_train <- gpb.Dataset(data = X, label = y)
set.seed(1)
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid,
                                              data = data_train, gp_model = gp_model,
                                              num_try_random = 100, nfold = 5,
                                              nrounds = 1000, early_stopping_rounds = 20,
                                              verbose_eval = 1, metric = metric, cv_seed = 4)
print(paste0("Best parameters: ",
             paste0(unlist(lapply(seq_along(opt_params$best_params), 
                                  function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, 
                                  y=opt_params$best_params, 
                                  n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", round(opt_params$best_score, digits=3)))
# Alternatively and faster: using manually defined validation data instead of cross-validation
# use 20% of the data as validation data
valid_tune_idx <- sample.int(length(y), as.integer(0.2*length(y))) 
folds <- list(valid_tune_idx)
opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid,
                                              data = data_train, gp_model = gp_model,
                                              num_try_random = 100, folds = folds,
                                              nrounds = 1000, early_stopping_rounds = 20,
                                              verbose_eval = 1, metric = metric, cv_seed = 4)

## End(No test)



cleanEx()
nameEx("gpb.importance")
### * gpb.importance

flush(stderr()); flush(stdout())

### Name: gpb.importance
### Title: Compute feature importance in a model
### Aliases: gpb.importance

### ** Examples

## No test: 
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)

params <- list(
  objective = "binary"
  , learning_rate = 0.1
  , max_depth = -1L
  , min_data_in_leaf = 1L
  , min_sum_hessian_in_leaf = 1.0
)
model <- gpb.train(
    params = params
    , data = dtrain
    , nrounds = 5L
)

tree_imp1 <- gpb.importance(model, percentage = TRUE)
tree_imp2 <- gpb.importance(model, percentage = FALSE)
## End(No test)



cleanEx()
nameEx("gpb.interprete")
### * gpb.interprete

flush(stderr()); flush(stdout())

### Name: gpb.interprete
### Title: Compute feature contribution of prediction
### Aliases: gpb.interprete

### ** Examples

## No test: 
Logit <- function(x) log(x / (1.0 - x))
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
setinfo(dtrain, "init_score", rep(Logit(mean(train$label)), length(train$label)))
data(agaricus.test, package = "gpboost")
test <- agaricus.test

params <- list(
    objective = "binary"
    , learning_rate = 0.1
    , max_depth = -1L
    , min_data_in_leaf = 1L
    , min_sum_hessian_in_leaf = 1.0
)
model <- gpb.train(
    params = params
    , data = dtrain
    , nrounds = 3L
)

tree_interpretation <- gpb.interprete(model, test$data, 1L:5L)
## End(No test)



cleanEx()
nameEx("gpb.load")
### * gpb.load

flush(stderr()); flush(stdout())

### Name: gpb.load
### Title: Load GPBoost model
### Aliases: gpb.load

### ** Examples

## No test: 
library(gpboost)
data(GPBoost_data, package = "gpboost")

# Train model and make prediction
gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = 16,
               learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
               verbose = 0)
pred <- predict(bst, data = X_test, group_data_pred = group_data_test[,1],
                predict_var= TRUE, pred_latent = TRUE)
# Save model to file
filename <- tempfile(fileext = ".json")
gpb.save(bst,filename = filename)
# Load from file and make predictions again
bst_loaded <- gpb.load(filename = filename)
pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test[,1],
                       predict_var= TRUE, pred_latent = TRUE)
# Check equality
pred$fixed_effect - pred_loaded$fixed_effect
pred$random_effect_mean - pred_loaded$random_effect_mean
pred$random_effect_cov - pred_loaded$random_effect_cov
## End(No test)



cleanEx()
nameEx("gpb.model.dt.tree")
### * gpb.model.dt.tree

flush(stderr()); flush(stdout())

### Name: gpb.model.dt.tree
### Title: Parse a GPBoost model json dump
### Aliases: gpb.model.dt.tree

### ** Examples

## No test: 
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)

params <- list(
  objective = "binary"
  , learning_rate = 0.01
  , num_leaves = 63L
  , max_depth = -1L
  , min_data_in_leaf = 1L
  , min_sum_hessian_in_leaf = 1.0
)
model <- gpb.train(params, dtrain, 10L)

tree_dt <- gpb.model.dt.tree(model)
## End(No test)



cleanEx()
nameEx("gpb.plot.importance")
### * gpb.plot.importance

flush(stderr()); flush(stdout())

### Name: gpb.plot.importance
### Title: Plot feature importance as a bar graph
### Aliases: gpb.plot.importance

### ** Examples

## No test: 
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)

params <- list(
    objective = "binary"
    , learning_rate = 0.1
    , min_data_in_leaf = 1L
    , min_sum_hessian_in_leaf = 1.0
)

model <- gpb.train(
    params = params
    , data = dtrain
    , nrounds = 5L
)

tree_imp <- gpb.importance(model, percentage = TRUE)
gpb.plot.importance(tree_imp, top_n = 5L, measure = "Gain")
## End(No test)



cleanEx()
nameEx("gpb.plot.interpretation")
### * gpb.plot.interpretation

flush(stderr()); flush(stdout())

### Name: gpb.plot.interpretation
### Title: Plot feature contribution as a bar graph
### Aliases: gpb.plot.interpretation

### ** Examples

## No test: 
Logit <- function(x) {
  log(x / (1.0 - x))
}
data(agaricus.train, package = "gpboost")
labels <- agaricus.train$label
dtrain <- gpb.Dataset(
  agaricus.train$data
  , label = labels
)
setinfo(dtrain, "init_score", rep(Logit(mean(labels)), length(labels)))

data(agaricus.test, package = "gpboost")

params <- list(
  objective = "binary"
  , learning_rate = 0.1
  , max_depth = -1L
  , min_data_in_leaf = 1L
  , min_sum_hessian_in_leaf = 1.0
)
model <- gpb.train(
  params = params
  , data = dtrain
  , nrounds = 5L
)

tree_interpretation <- gpb.interprete(
  model = model
  , data = agaricus.test$data
  , idxset = 1L:5L
)
gpb.plot.interpretation(
  tree_interpretation_dt = tree_interpretation[[1L]]
  , top_n = 3L
)
## End(No test)



cleanEx()
nameEx("gpb.plot.part.dep.interact")
### * gpb.plot.part.dep.interact

flush(stderr()); flush(stdout())

### Name: gpb.plot.part.dep.interact
### Title: Plot interaction partial dependence plots
### Aliases: gpb.plot.part.dep.interact

### ** Examples

## No test: 
library(gpboost)
data(GPBoost_data, package = "gpboost")
gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
gpboost_model <- gpboost(data = X,
                        label = y,
                        gp_model = gp_model,
                        nrounds = 16,
                        learning_rate = 0.05,
                        max_depth = 6,
                        min_data_in_leaf = 5,
                        verbose = 0)
gpb.plot.part.dep.interact(gpboost_model, X, variables = c(1,2))
## End(No test)



cleanEx()
nameEx("gpb.plot.partial.dependence")
### * gpb.plot.partial.dependence

flush(stderr()); flush(stdout())

### Name: gpb.plot.partial.dependence
### Title: Plot partial dependence plots
### Aliases: gpb.plot.partial.dependence

### ** Examples

## No test: 
library(gpboost)
data(GPBoost_data, package = "gpboost")

gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
gpboost_model <- gpboost(data = X,
                         label = y,
                         gp_model = gp_model,
                         nrounds = 16,
                         learning_rate = 0.05,
                         max_depth = 6,
                         min_data_in_leaf = 5,
                         verbose = 0)
gpb.plot.partial.dependence(gpboost_model, X, variable = 1)
## End(No test)



cleanEx()
nameEx("gpb.save")
### * gpb.save

flush(stderr()); flush(stdout())

### Name: gpb.save
### Title: Save GPBoost model (booster and GPModel)
### Aliases: gpb.save

### ** Examples

## No test: 
library(gpboost)
data(GPBoost_data, package = "gpboost")

# Train model and make prediction
gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = 16,
               learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
               verbose = 0)
pred <- predict(bst, data = X_test, group_data_pred = group_data_test[,1],
                predict_var= TRUE, pred_latent = TRUE)
# Save model to file
filename <- tempfile(fileext = ".json")
gpb.save(bst,filename = filename)
# Load from file and make predictions again
bst_loaded <- gpb.load(filename = filename)
pred_loaded <- predict(bst_loaded, data = X_test, group_data_pred = group_data_test[,1],
                       predict_var= TRUE, pred_latent = TRUE)
# Check equality
pred$fixed_effect - pred_loaded$fixed_effect
pred$random_effect_mean - pred_loaded$random_effect_mean
pred$random_effect_cov - pred_loaded$random_effect_cov
## End(No test)



cleanEx()
nameEx("gpb.set.default.num.threads")
### * gpb.set.default.num.threads

flush(stderr()); flush(stdout())

### Name: gpb.set.default.num.threads
### Title: Set the number of threads used by models for which no number of
###   threads is specified
### Aliases: gpb.set.default.num.threads

### ** Examples

num_threads_old <- gpb.get.default.num.threads()
gpb.set.default.num.threads(2L)
gpb.set.default.num.threads(num_threads_old)



cleanEx()
nameEx("gpb.set.num.threads")
### * gpb.set.num.threads

flush(stderr()); flush(stdout())

### Name: gpb.set.num.threads
### Title: Set the number of threads used for parallelization
### Aliases: gpb.set.num.threads

### ** Examples

num_threads_old <- gpb.get.num.threads()
gpb.set.num.threads(2L)
gpb.set.num.threads(num_threads_old)



cleanEx()
nameEx("gpb.train")
### * gpb.train

flush(stderr()); flush(stdout())

### Name: gpb.train
### Title: Main training logic for GBPoost
### Aliases: gpb.train

### ** Examples

# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples

## No test: 
library(gpboost)
data(GPBoost_data, package = "gpboost")

#--------------------Combine tree-boosting and grouped random effects model----------------
# Create random effects model
gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
# The default optimizer for covariance parameters (hyperparameters) is 
# Nesterov-accelerated gradient descent.
# This can be changed to, e.g., Nelder-Mead as follows:
# re_params <- list(optimizer_cov = "nelder_mead")
# gp_model$set_optim_params(params=re_params)
# Use trace = TRUE to monitor convergence:
# re_params <- list(trace = TRUE)
# gp_model$set_optim_params(params=re_params)
dtrain <- gpb.Dataset(data = X, label = y)
# Train model
bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 16,
                 learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
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
dtrain <- gpb.Dataset(data = X, label = y)
bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 16,
                 learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
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
bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 100,
                 learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
                 verbose = 1, valids = valids,
                 early_stopping_rounds = 10, use_gp_model_for_validation = TRUE)
print(paste0("Optimal number of iterations: ", bst$best_iter,
             ", best test error: ", bst$best_score))
# Plot validation error
val_error <- unlist(bst$record_evals$test$l2$eval)
plot(1:length(val_error), val_error, type="l", lwd=2, col="blue",
     xlab="iteration", ylab="Validation error", main="Validation error vs. boosting iteration")


#--------------------Do Newton updates for tree leaves---------------
# Note: run the above examples first
bst <- gpb.train(data = dtrain, gp_model = gp_model, nrounds = 100,
                 learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
                 verbose = 1, valids = valids,
                 early_stopping_rounds = 5, use_gp_model_for_validation = FALSE,
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
               min_data_in_leaf = 5)
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
                 verbose = 0,
                 train_gp_model_cov_pars = FALSE)
# The GPModel has not changed:
summary(gp_model)
## End(No test)



cleanEx()
nameEx("gpb.tune.num.threads")
### * gpb.tune.num.threads

flush(stderr()); flush(stdout())

### Name: gpb.tune.num.threads
### Title: Benchmark different numbers of threads and select a tuned
###   default
### Aliases: gpb.tune.num.threads

### ** Examples

## No test: 
# Benchmark all workloads and use the result for the rest of the session
results <- gpb.tune.num.threads()

# Benchmark only the model class that is mainly used
results <- gpb.tune.num.threads(workloads = "grouped_re")

# Only measure, without changing the default
results <- gpb.tune.num.threads(set_default = FALSE)
## End(No test)



cleanEx()
nameEx("gpb.tune.pars.bayesian.optimization")
### * gpb.tune.pars.bayesian.optimization

flush(stderr()); flush(stdout())

### Name: gpb.tune.pars.bayesian.optimization
### Title: Function for choosing tuning parameters using Bayesian
###   optimization with the 'mlrMBO' package
### Aliases: gpb.tune.pars.bayesian.optimization

### ** Examples

# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
## No test: 
if (requireNamespace("mlrMBO", quietly = TRUE) && requireNamespace("DiceKriging", quietly = TRUE) &&
    requireNamespace("rgenoud", quietly = TRUE) &&
    requireNamespace("randomForest", quietly = TRUE)) {
library(gpboost)
data(GPBoost_data, package = "gpboost")
n <- length(y)
# Define search space
# Note: if the best combination found below is close to the bounday for a paramter,
#   you might want to extend the corresponding range
search_space <- list("learning_rate" = c(0.001, 10),
                     "min_data_in_leaf" = c(1, 1000),
                     "max_depth" = c(-1, -1),
                     "num_leaves" = c(2, 2^10),
                     "lambda_l2" = c(0, 100),
                     "max_bin" = c(63, min(n,10000)),
                     "line_search_step_length" = c(TRUE, FALSE))
# Note: "max_depth" = c(-1, -1) means no depth limit as we tune 'num_leaves'.
#    Can also additionally tune 'max_depth', e.g., "max_depth" = c(-1, 10)
metric = "mse" # Define metric
# Note: can also use metric = "test_neg_log_likelihood".
# See https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
data_train <- gpb.Dataset(data = X, label = y)
# Run parameter optimization using Bayesian optimization and 5-fold CV
opt_params <- gpb.tune.pars.bayesian.optimization(search_space = search_space, n_iter = 100,
                                                  data = data_train, gp_model = gp_model,
                                                  nfold = 5, nrounds = 1000,
                                                  early_stopping_rounds = 20, metric = metric,
                                                  cv_seed = 4, verbose_eval = 1)
print(paste0("Best parameters: ",
             paste0(unlist(lapply(seq_along(opt_params$best_params),
                                  function(y, n, i) { paste0(n[[i]],": ", y[[i]]) },
                                  y=opt_params$best_params,
                                  n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", round(opt_params$best_score, digits=3)))
}
## End(No test)



cleanEx()
nameEx("gpb.tune.pars.rf.bayesian.optimization")
### * gpb.tune.pars.rf.bayesian.optimization

flush(stderr()); flush(stdout())

### Name: gpb.tune.pars.rf.bayesian.optimization
### Title: Function for choosing tuning parameters using an approximation
###   of SMAC with the 'mlr3mbo' package
### Aliases: gpb.tune.pars.rf.bayesian.optimization

### ** Examples

# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples
## No test: 
if (requireNamespace("mlr3mbo", quietly = TRUE) &&
    requireNamespace("mlr3learners", quietly = TRUE) &&
    requireNamespace("ranger", quietly = TRUE)) {
library(gpboost)
data(GPBoost_data, package = "gpboost")
n <- length(y)
# Define search space
# Note: if the best combination found below is close to the bounday for a paramter,
#   you might want to extend the corresponding range
search_space <- list("learning_rate" = c(0.001, 10),
                     "min_data_in_leaf" = c(1, 1000),
                     "max_depth" = c(-1, -1),
                     "num_leaves" = c(2, 2^10),
                     "lambda_l2" = c(0, 100),
                     "max_bin" = c(63, min(n,10000)),
                     "line_search_step_length" = c(TRUE, FALSE))
# Note: "max_depth" = c(-1, -1) means no depth limit as we tune 'num_leaves'.
#    Can also additionally tune 'max_depth', e.g., "max_depth" = c(-1, 10)
metric = "mse" # Define metric
# Note: can also use metric = "test_neg_log_likelihood".
# See https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#metric-parameters
gp_model <- GPModel(group_data = group_data[,1], likelihood="gaussian")
data_train <- gpb.Dataset(data = X, label = y)
# Run parameter optimization using the approximation of SMAC and 5-fold CV
opt_params <- gpb.tune.pars.rf.bayesian.optimization(search_space = search_space, n_trials = 100,
                                                     data = data_train, gp_model = gp_model,
                                                     nfold = 5, nrounds = 1000,
                                                     early_stopping_rounds = 20, metric = metric,
                                                     cv_seed = 4, verbose_eval = 1)
print(paste0("Best parameters: ",
             paste0(unlist(lapply(seq_along(opt_params$best_params),
                                  function(y, n, i) { paste0(n[[i]],": ", y[[i]]) },
                                  y=opt_params$best_params,
                                  n=names(opt_params$best_params))), collapse=", ")))
print(paste0("Best number of iterations: ", opt_params$best_iter))
print(paste0("Best score: ", round(opt_params$best_score, digits=3)))
}
## End(No test)



cleanEx()
nameEx("gpboost")
### * gpboost

flush(stderr()); flush(stdout())

### Name: gpboost
### Title: Train a GPBoost model
### Aliases: gpboost

### ** Examples

# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples

## No test: 
library(gpboost)
data(GPBoost_data, package = "gpboost")

#--------------------Combine tree-boosting and grouped random effects model----------------
# Create random effects model
gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
# The default optimizer for covariance parameters (hyperparameters) is 
# Nesterov-accelerated gradient descent.
# This can be changed to, e.g., Nelder-Mead as follows:
# re_params <- list(optimizer_cov = "nelder_mead")
# gp_model$set_optim_params(params=re_params)
# Use trace = TRUE to monitor convergence:
# re_params <- list(trace = TRUE)
# gp_model$set_optim_params(params=re_params)

# Train model
bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = 16,
               learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
               verbose = 0)
# Estimated random effects model
summary(gp_model)

# Make predictions
# Predict latent variables
pred <- predict(bst, data = X_test, group_data_pred = group_data_test[,1],
                predict_var = TRUE, pred_latent = TRUE)
pred$random_effect_mean # Predicted latent random effects mean
pred$random_effect_cov # Predicted random effects variances
pred$fixed_effect # Predicted fixed effects from tree ensemble
# Predict response variable
pred_resp <- predict(bst, data = X_test, group_data_pred = group_data_test[,1],
                     predict_var = TRUE, pred_latent = FALSE)
pred_resp$response_mean # Predicted response mean
# For Gaussian data: pred$random_effect_mean + pred$fixed_effect = pred_resp$response_mean
pred$random_effect_mean + pred$fixed_effect - pred_resp$response_mean

#--------------------Combine tree-boosting and Gaussian process model----------------
# Create Gaussian process model
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                    likelihood = "gaussian")
# Train model
bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = 8,
               learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5,
               verbose = 0)
# Estimated random effects model
summary(gp_model)
# Make predictions
pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                predict_var = TRUE, pred_latent = TRUE)
pred$random_effect_mean # Predicted latent random effects mean
pred$random_effect_cov # Predicted random effects variances
pred$fixed_effect # Predicted fixed effects from tree ensemble
# Predict response variable
pred_resp <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                     predict_var = TRUE, pred_latent = FALSE)
pred_resp$response_mean # Predicted response mean
## End(No test)



cleanEx()
nameEx("loadGPModel")
### * loadGPModel

flush(stderr()); flush(stdout())

### Name: loadGPModel
### Title: Load a 'GPModel' from a file
### Aliases: loadGPModel

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
# Add intercept column
X1 <- cbind(rep(1,dim(X)[1]),X)
X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)

gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
                X_pred = X_test1, predict_var = TRUE)
# Save model to file
filename <- tempfile(fileext = ".json")
saveGPModel(gp_model,filename = filename)
# Load from file and make predictions again
gp_model_loaded <- loadGPModel(filename = filename)
pred_loaded <- predict(gp_model_loaded, group_data_pred = group_data_test[,1], 
                       X_pred = X_test1, predict_var = TRUE)
# Check equality
pred$mu - pred_loaded$mu
pred$var - pred_loaded$var
## End(No test)



cleanEx()
nameEx("neg_log_likelihood.GPModel")
### * neg_log_likelihood.GPModel

flush(stderr()); flush(stdout())

### Name: neg_log_likelihood.GPModel
### Title: Evaluate the negative log-likelihood
### Aliases: neg_log_likelihood.GPModel

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
gp_model <- GPModel(group_data = group_data, likelihood="gaussian")
X1 <- cbind(rep(1,dim(X)[1]), X)
coef <- c(0.1, 0.1, 0.1)
fixed_effects <- as.numeric(X1 %*% coef)
neg_log_likelihood(gp_model, y = y, cov_pars = c(0.1,1,1), 
                   fixed_effects = fixed_effects)
## End(No test)



cleanEx()
nameEx("neg_log_likelihood")
### * neg_log_likelihood

flush(stderr()); flush(stdout())

### Name: neg_log_likelihood
### Title: Evaluate the negative log-likelihood
### Aliases: neg_log_likelihood

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
gp_model <- GPModel(group_data = group_data, likelihood="gaussian")
X1 <- cbind(rep(1,dim(X)[1]), X)
coef <- c(0.1, 0.1, 0.1)
fixed_effects <- as.numeric(X1 %*% coef)
neg_log_likelihood(gp_model, y = y, cov_pars = c(0.1,1,1), 
                   fixed_effects = fixed_effects)
## End(No test)



cleanEx()
nameEx("predict.GPModel")
### * predict.GPModel

flush(stderr()); flush(stdout())

### Name: predict.GPModel
### Title: Make predictions for a 'GPModel'
### Aliases: predict.GPModel

### ** Examples

# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples

## No test: 
data(GPBoost_data, package = "gpboost")
# Add intercept column
X1 <- cbind(rep(1,dim(X)[1]),X)
X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)

#--------------------Grouped random effects model: single-level random effect----------------
gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1,
                       likelihood="gaussian")
summary(gp_model)
# Make predictions
pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
                X_pred = X_test1, predict_var = TRUE)
pred$mu # Predicted mean
pred$var # Predicted variances
# Also predict covariance matrix
pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
                X_pred = X_test1, predict_cov_mat = TRUE)
pred$mu # Predicted mean
pred$cov # Predicted covariance


#--------------------Gaussian process model----------------
gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                       likelihood="gaussian", y = y, X = X1)
summary(gp_model)
# Make predictions
pred <- predict(gp_model, gp_coords_pred = coords_test, 
                X_pred = X_test1, predict_cov_mat = TRUE)
pred$mu # Predicted (posterior) mean of GP
pred$cov # Predicted (posterior) covariance matrix of GP
## End(No test)




cleanEx()
nameEx("predict.gpb.Booster")
### * predict.gpb.Booster

flush(stderr()); flush(stdout())

### Name: predict.gpb.Booster
### Title: Prediction function for 'gpb.Booster' objects
### Aliases: predict.gpb.Booster

### ** Examples


# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples

## No test: 
library(gpboost)
data(GPBoost_data, package = "gpboost")

#--------------------Combine tree-boosting and grouped random effects model----------------
# Create random effects model
gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
# The default optimizer for covariance parameters (hyperparameters) is 
# Nesterov-accelerated gradient descent.
# This can be changed to, e.g., Nelder-Mead as follows:
# re_params <- list(optimizer_cov = "nelder_mead")
# gp_model$set_optim_params(params=re_params)
# Use trace = TRUE to monitor convergence:
# re_params <- list(trace = TRUE)
# gp_model$set_optim_params(params=re_params)

# Train model
bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = 16,
               learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
               verbose = 0)
# Estimated random effects model
summary(gp_model)

# Make predictions
# Predict latent variables
pred <- predict(bst, data = X_test, group_data_pred = group_data_test[,1],
                predict_var = TRUE, pred_latent = TRUE)
pred$random_effect_mean # Predicted latent random effects mean
pred$random_effect_cov # Predicted random effects variances
pred$fixed_effect # Predicted fixed effects from tree ensemble
# Predict response variable
pred_resp <- predict(bst, data = X_test, group_data_pred = group_data_test[,1],
                     predict_var = TRUE, pred_latent = FALSE)
pred_resp$response_mean # Predicted response mean
# For Gaussian data: pred$random_effect_mean + pred$fixed_effect = pred_resp$response_mean
pred$random_effect_mean + pred$fixed_effect - pred_resp$response_mean

#--------------------Combine tree-boosting and Gaussian process model----------------
# Create Gaussian process model
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                    likelihood = "gaussian")
# Train model
bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = 8,
               learning_rate = 0.1, max_depth = 6, min_data_in_leaf = 5,
               verbose = 0)
# Estimated random effects model
summary(gp_model)
# Make predictions
pred <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                predict_var = TRUE, pred_latent = TRUE)
pred$random_effect_mean # Predicted latent random effects mean
pred$random_effect_cov # Predicted random effects variances
pred$fixed_effect # Predicted fixed effects from tree ensemble
# Predict response variable
pred_resp <- predict(bst, data = X_test, gp_coords_pred = coords_test,
                     predict_var = TRUE, pred_latent = FALSE)
pred_resp$response_mean # Predicted response mean
## End(No test)



cleanEx()
nameEx("predict_training_data_random_effects.GPModel")
### * predict_training_data_random_effects.GPModel

flush(stderr()); flush(stdout())

### Name: predict_training_data_random_effects.GPModel
### Title: Predict ("estimate") training data random effects for a
###   'GPModel'
### Aliases: predict_training_data_random_effects.GPModel
###   predict_training_data_random_effects.gpb.Booster

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
# Add intercept column
X1 <- cbind(rep(1,dim(X)[1]),X)
X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)

gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
first_occurences <- match(unique(group_data[,1]), group_data[,1])
unique_training_data_random_effects <- all_training_data_random_effects[first_occurences]
head(unique_training_data_random_effects)
## End(No test)



cleanEx()
nameEx("predict_training_data_random_effects")
### * predict_training_data_random_effects

flush(stderr()); flush(stdout())

### Name: predict_training_data_random_effects
### Title: Predict ("estimate") training data random effects for a
###   'GPModel'
### Aliases: predict_training_data_random_effects

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
# Add intercept column
X1 <- cbind(rep(1,dim(X)[1]),X)
X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)

gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
all_training_data_random_effects <- predict_training_data_random_effects(gp_model)
first_occurences <- match(unique(group_data[,1]), group_data[,1])
unique_training_data_random_effects <- all_training_data_random_effects[first_occurences]
head(unique_training_data_random_effects)
## End(No test)



cleanEx()
nameEx("readRDS.gpb.Booster")
### * readRDS.gpb.Booster

flush(stderr()); flush(stdout())

### Name: readRDS.gpb.Booster
### Title: readRDS for 'gpb.Booster' models
### Aliases: readRDS.gpb.Booster

### ** Examples

## No test: 
library(gpboost)
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
data(agaricus.test, package = "gpboost")
test <- agaricus.test
dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
params <- list(objective = "regression", metric = "l2")
valids <- list(test = dtest)
model <- gpb.train(
  params = params
  , data = dtrain
  , nrounds = 10L
  , valids = valids
  , min_data = 1L
  , learning_rate = 1.0
  , early_stopping_rounds = 5L
)
model_file <- tempfile(fileext = ".rds")
saveRDS.gpb.Booster(model, model_file)
new_model <- readRDS.gpb.Booster(model_file)
## End(No test)



cleanEx()
nameEx("saveGPModel")
### * saveGPModel

flush(stderr()); flush(stdout())

### Name: saveGPModel
### Title: Save a 'GPModel'
### Aliases: saveGPModel

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
# Add intercept column
X1 <- cbind(rep(1,dim(X)[1]),X)
X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)

gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1, likelihood="gaussian")
pred <- predict(gp_model, group_data_pred = group_data_test[,1], 
                X_pred = X_test1, predict_var = TRUE)
# Save model to file
filename <- tempfile(fileext = ".json")
saveGPModel(gp_model,filename = filename)
# Load from file and make predictions again
gp_model_loaded <- loadGPModel(filename = filename)
pred_loaded <- predict(gp_model_loaded, group_data_pred = group_data_test[,1], 
                       X_pred = X_test1, predict_var = TRUE)
# Check equality
pred$mu - pred_loaded$mu
pred$var - pred_loaded$var
## End(No test)



cleanEx()
nameEx("saveRDS.gpb.Booster")
### * saveRDS.gpb.Booster

flush(stderr()); flush(stdout())

### Name: saveRDS.gpb.Booster
### Title: saveRDS for 'gpb.Booster' models
### Aliases: saveRDS.gpb.Booster

### ** Examples

## No test: 
library(gpboost)
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
data(agaricus.test, package = "gpboost")
test <- agaricus.test
dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
params <- list(objective = "regression", metric = "l2")
valids <- list(test = dtest)
model <- gpb.train(
    params = params
    , data = dtrain
    , nrounds = 10L
    , valids = valids
    , min_data = 1L
    , learning_rate = 1.0
    , early_stopping_rounds = 5L
)
model_file <- tempfile(fileext = ".rds")
saveRDS.gpb.Booster(model, model_file)
## End(No test)



cleanEx()
nameEx("set_optim_params.GPModel")
### * set_optim_params.GPModel

flush(stderr()); flush(stdout())

### Name: set_optim_params.GPModel
### Title: Set parameters for estimation of the covariance parameters
### Aliases: set_optim_params.GPModel

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
gp_model <- GPModel(group_data = group_data, likelihood="gaussian")
set_optim_params(gp_model, params=list(optimizer_cov="nelder_mead"))
## End(No test)



cleanEx()
nameEx("set_optim_params")
### * set_optim_params

flush(stderr()); flush(stdout())

### Name: set_optim_params
### Title: Set parameters for estimation of the covariance parameters
### Aliases: set_optim_params

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
gp_model <- GPModel(group_data = group_data, likelihood="gaussian")
set_optim_params(gp_model, params=list(optimizer_cov="nelder_mead"))
## End(No test)




cleanEx()
nameEx("set_prediction_data.GPModel")
### * set_prediction_data.GPModel

flush(stderr()); flush(stdout())

### Name: set_prediction_data.GPModel
### Title: Set prediction data for a 'GPModel'
### Aliases: set_prediction_data.GPModel

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
set.seed(1)
train_ind <- sample.int(length(y),size=250)
gp_model <- GPModel(group_data = group_data[train_ind,1], likelihood="gaussian")
set_prediction_data(gp_model, group_data_pred = group_data[-train_ind,1])
## End(No test)



cleanEx()
nameEx("set_prediction_data")
### * set_prediction_data

flush(stderr()); flush(stdout())

### Name: set_prediction_data
### Title: Set prediction data for a 'GPModel'
### Aliases: set_prediction_data

### ** Examples

## No test: 
data(GPBoost_data, package = "gpboost")
set.seed(1)
train_ind <- sample.int(length(y),size=250)
gp_model <- GPModel(group_data = group_data[train_ind,1], likelihood="gaussian")
set_prediction_data(gp_model, group_data_pred = group_data[-train_ind,1])
## End(No test)




cleanEx()
nameEx("setinfo")
### * setinfo

flush(stderr()); flush(stdout())

### Name: setinfo
### Title: Set information of an 'gpb.Dataset' object
### Aliases: setinfo setinfo.gpb.Dataset

### ** Examples

## No test: 
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
gpb.Dataset.construct(dtrain)

labels <- gpboost::getinfo(dtrain, "label")
gpboost::setinfo(dtrain, "label", 1 - labels)

labels2 <- gpboost::getinfo(dtrain, "label")
stopifnot(all.equal(labels2, 1 - labels))
## End(No test)



cleanEx()
nameEx("slice")
### * slice

flush(stderr()); flush(stdout())

### Name: slice
### Title: Slice a dataset
### Aliases: slice slice.gpb.Dataset

### ** Examples

## No test: 
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)

dsub <- gpboost::slice(dtrain, seq_len(42L))
gpb.Dataset.construct(dsub)
labels <- gpboost::getinfo(dsub, "label")
## End(No test)



cleanEx()
nameEx("summary.GPModel")
### * summary.GPModel

flush(stderr()); flush(stdout())

### Name: summary.GPModel
### Title: Summary for a 'GPModel'
### Aliases: summary.GPModel

### ** Examples

# See https://github.com/fabsig/GPBoost/tree/master/R-package for more examples

data(GPBoost_data, package = "gpboost")
# Add intercept column
X1 <- cbind(rep(1,dim(X)[1]),X)
X_test1 <- cbind(rep(1,dim(X_test)[1]),X_test)

#--------------------Grouped random effects model: single-level random effect----------------
gp_model <- fitGPModel(group_data = group_data[,1], y = y, X = X1,
                       likelihood="gaussian")
summary(gp_model)


## No test: 
#--------------------Gaussian process model----------------
gp_model <- fitGPModel(gp_coords = coords, cov_function = "matern", cov_fct_shape = 1.5,
                       likelihood="gaussian", y = y, X = X1)
summary(gp_model)
## End(No test)




### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
