context("gpb.unloader")

test_that("gpb.unloader works as expected", {
    data(agaricus.train, package = "gpboost")
    train <- agaricus.train
    dtrain <- gpb.Dataset(train$data, label = train$label)
    bst <- gpb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
        )
        , data = dtrain
        , nrounds = 1L
        , min_data = 1L
        , learning_rate = 1.0
    )
    expect_true(exists("bst"))
    result <- gpb.unloader(restore = TRUE, wipe = TRUE, envir = environment())
    expect_false(exists("bst"))
    expect_null(result)
})

test_that("gpb.unloader finds all boosters and removes them", {
    data(agaricus.train, package = "gpboost")
    train <- agaricus.train
    dtrain <- gpb.Dataset(train$data, label = train$label)
    bst1 <- gpb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
        )
        , data = dtrain
        , nrounds = 1L
        , min_data = 1L
        , learning_rate = 1.0
    )
    bst2 <- gpb.train(
        params = list(
            objective = "regression"
            , metric = "l2"
        )
        , data = dtrain
        , nrounds = 1L
        , min_data = 1L
        , learning_rate = 1.0
    )
    expect_true(exists("bst1"))
    expect_true(exists("bst2"))
    gpb.unloader(restore = TRUE, wipe = TRUE, envir = environment())
    expect_false(exists("bst1"))
    expect_false(exists("bst2"))
})
