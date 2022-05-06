context("Predictor")

test_that("Predictor$finalize() should not fail", {
    X <- as.matrix(as.integer(iris[, "Species"]), ncol = 1L)
    y <- iris[["Sepal.Length"]]
    dtrain <- gpb.Dataset(X, label = y)
    bst <- gpb.train(
        data = dtrain
        , objective = "regression"
        , verbose = -1L
        , nrounds = 3L
    )
    model_file <- tempfile(fileext = ".model")
    bst$save_model(filename = model_file)
    predictor <- gpboost:::Predictor$new(modelfile = model_file)
    
    expect_false(gpboost:::gpb.is.null.handle(predictor$.__enclos_env__$private$handle))
    
    predictor$finalize()
    expect_true(gpboost:::gpb.is.null.handle(predictor$.__enclos_env__$private$handle))
    
    # calling finalize() a second time shouldn't cause any issues
    predictor$finalize()
    expect_true(gpboost:::gpb.is.null.handle(predictor$.__enclos_env__$private$handle))
})

test_that("predictions do not fail for integer input", {
    X <- as.matrix(as.integer(iris[, "Species"]), ncol = 1L)
    y <- iris[["Sepal.Length"]]
    dtrain <- gpb.Dataset(X, label = y)
    fit <- gpb.train(
        data = dtrain
        , objective = "regression"
        , verbose = -1L
        , nrounds = 3L
    )
    X_double <- X[c(1L, 51L, 101L), , drop = FALSE]
    X_integer <- X_double
    storage.mode(X_double) <- "double"
    pred_integer <- predict(fit, X_integer)
    pred_double <- predict(fit, X_double)
    expect_equal(pred_integer, pred_double)
})

test_that("start_iteration works correctly", {
    set.seed(708L)
    data(agaricus.train, package = "gpboost")
    data(agaricus.test, package = "gpboost")
    train <- agaricus.train
    test <- agaricus.test
    dtrain <- gpb.Dataset(
        agaricus.train$data
        , label = agaricus.train$label
    )
    dtest <- gpb.Dataset.create.valid(
        dtrain
        , agaricus.test$data
        , label = agaricus.test$label
    )
    bst <- gpboost(
        data = as.matrix(train$data)
        , label = train$label
        , num_leaves = 4L
        , learning_rate = 0.6
        , nrounds = 50L
        , objective = "binary"
        , valids = list("test" = dtest)
        , early_stopping_rounds = 2L
        , verbose = 0
    )
    expect_true(gpboost:::gpb.is.Booster(bst))
    pred1 <- predict(bst, data = test$data, pred_latent = TRUE)
    pred_contrib1 <- predict(bst, test$data, predcontrib = TRUE)
    pred2 <- rep(0.0, length(pred1))
    pred_contrib2 <- rep(0.0, length(pred2))
    step <- 11L
    end_iter <- 49L
    if (bst$best_iter != -1L) {
        end_iter <- bst$best_iter - 1L
    }
    start_iters <- seq(0L, end_iter, by = step)
    for (start_iter in start_iters) {
        n_iter <- min(c(end_iter - start_iter + 1L, step))
        inc_pred <- predict(bst, test$data
            , start_iteration = start_iter
            , num_iteration = n_iter
            , pred_latent = TRUE
        )
        inc_pred_contrib <- bst$predict(test$data
            , start_iteration = start_iter
            , num_iteration = n_iter
            , predcontrib = TRUE
        )
        pred2 <- pred2 + inc_pred
        pred_contrib2 <- pred_contrib2 + inc_pred_contrib
    }
    expect_equal(pred2, pred1)
    expect_equal(pred_contrib2, pred_contrib1)

    pred_leaf1 <- predict(bst, test$data, predleaf = TRUE)
    pred_leaf2 <- predict(bst, test$data, start_iteration = 0L, num_iteration = end_iter + 1L, predleaf = TRUE)
    expect_equal(pred_leaf1, pred_leaf2)
})
