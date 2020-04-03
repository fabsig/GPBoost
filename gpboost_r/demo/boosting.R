require(gpboost)

#--------------------Simulate data----------------
# Non-linear function for simulation
f1d <- function(x) 1.7*(1/(1+exp(-(x-0.5)*20))+0.75*x)
x <- seq(from=0,to=1,length.out=200)
plot(x,f1d(x),type="l",lwd=2,col="red",main="Mean function")
# Function that simulates data. Two covariates of which only one has an effect
sim_data <- function(n){
  X=matrix(runif(2*n),ncol=2)
  # mean function plus noise
  y=f1d(X[,1])+rnorm(n,sd=0.1)
  return(list(X=X,y=y))
}
# Simulate data
n <- 1000
set.seed(1)
data <- sim_data(2 * n)
Xtrain <- data$X[1:n,]
ytrain <- data$y[1:n]
Xtest <- data$X[1:n + n,]
ytest <- data$y[1:n + n]


#--------------------Basic training using gpboost----------------
print("Train boosting model")
bst <- gpboost(data = Xtrain,
               label = ytrain,
               nrounds = 40,
               learning_rate = 0.1,
               max_depth = 6,
               min_data_in_leaf = 5,
               objective = "regression_l2",
               verbose = 0)

# You can also use an gpb.Dataset object, which stores label, data and other meta datas needed for advanced features
print("Training with gpb.Dataset")
dtrain <- gpb.Dataset(data = Xtrain, label = ytrain)
bst <- gpboost(data = dtrain,
               nrounds = 40,
               learning_rate = 0.1,
               max_depth = 6,
               min_data_in_leaf = 5,
               objective = "regression_l2",
               verbose = 0)

# Same thing using the gpb.train function
print("Training with gpb.train")
dtrain <- gpb.Dataset(data = Xtrain, label = ytrain)
bst <- gpb.train(data = dtrain,
                 nrounds = 40,
                 learning_rate = 0.1,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 verbose = 0)

# Verbose = 1, more output
print("Train with verbose 1, print evaluation metric")
bst <- gpboost(data = dtrain,
               nrounds = 40,
               learning_rate = 0.1,
               max_depth = 6,
               min_data_in_leaf = 5,
               objective = "regression_l2",
               verbose = 1)


#--------------------Basic prediction using gpboost--------------
pred <- predict(bst, data = Xtest)
err <- mean((ytest-pred)^2)
print(paste("test-RMSE =", err))

# Compare fit to truth
x <- seq(from=0,to=1,length.out=200)
Xtest_plot <- cbind(x,rep(0,length(x)))
pred_plot <- predict(bst, data = Xtest_plot)
plot(x,f1d(x),type="l",ylim = c(-0.25,3.25), col = "red", lwd = 2,
     main = "Comparison of true and fitted value")
lines(x,pred_plot, col = "blue", lwd = 2)
legend("bottomright", legend = c("truth", "fitted"),
       lwd=2, col = c("red", "blue"), bty = "n")


#--------------------Using validation set-------------------------
# valids is a list of gpb.Dataset, each of them is tagged with a name
dtrain <- gpb.Dataset(data = Xtrain, label = ytrain)
dtest <- gpb.Dataset.create.valid(dtrain, data = Xtest, label = ytest)
valids <- list(test = dtest)

# To train with valids, use gpb.train, which contains more advanced features
# valids allows us to monitor the evaluation result on all data in the list
print("Training using gpb.train with validation data ")
bst <- gpb.train(data = dtrain,
                 nrounds = 100,
                 learning_rate = 0.1,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 verbose = 1,
                 valids = valids,
                 early_stopping_rounds = 5)
print(paste0("Optimal number of iterations: ", bst$best_iter))

# We can change evaluation metrics, or use multiple evaluation metrics
print("Train using gpb.train with multiple validation metrics")
bst <- gpb.train(data = dtrain,
                 nrounds = 100,
                 learning_rate = 0.1,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 verbose = 1,
                 valids = valids,
                 eval = c("l2","l1"),
                 early_stopping_rounds = 5)
print(paste0("Optimal number of iterations: ", bst$best_iter))


#--------------------Nesterov accelerated boosting-------------------------
dtrain <- gpb.Dataset(data = Xtrain, label = ytrain)
dtest <- gpb.Dataset.create.valid(dtrain, data = Xtest, label = ytest)
valids <- list(test = dtest)
print("Training using gpb.train with Nesterov acceleration")
bst <- gpb.train(data = dtrain,
                 nrounds = 100,
                 learning_rate = 0.01,
                 max_depth = 6,
                 min_data_in_leaf = 5,
                 objective = "regression_l2",
                 verbose = 1,
                 valids = valids,
                 early_stopping_rounds = 5,
                 use_nesterov_acc = TRUE)
# Compare fit to truth
x <- seq(from=0,to=1,length.out=200)
Xtest_plot <- cbind(x,rep(0,length(x)))
pred_plot <- predict(bst, data = Xtest_plot)
plot(x,f1d(x),type="l",ylim = c(-0.25,3.25), col = "red", lwd = 2,
     main = "Comparison of true and fitted value")
lines(x,pred_plot, col = "blue", lwd = 2)
legend("bottomright", legend = c("truth", "fitted"),
       lwd=2, col = c("red", "blue"), bty = "n")
