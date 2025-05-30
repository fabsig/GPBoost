context("Booster")

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  test_that("Booster$finalize() should not fail", {
    X <- as.matrix(as.integer(iris[, "Species"]), ncol = 1L)
    y <- iris[["Sepal.Length"]]
    dtrain <- gpb.Dataset(X, label = y)
    bst <- gpb.train(
      data = dtrain
      , objective = "regression"
      , verbose = -1L
      , nrounds = 3L
    )
    expect_true(gpboost:::gpb.is.Booster(bst))
    
    expect_false(gpboost:::gpb.is.null.handle(bst$.__enclos_env__$private$handle))
    
    bst$.__enclos_env__$private$finalize()
    expect_true(gpboost:::gpb.is.null.handle(bst$.__enclos_env__$private$handle))
    
    # calling finalize() a second time shouldn't cause any issues
    bst$.__enclos_env__$private$finalize()
    expect_true(gpboost:::gpb.is.null.handle(bst$.__enclos_env__$private$handle))
  })
  
  context("gpb.get.eval.result")
  
  test_that("gpb.get.eval.result() should throw an informative error if booster is not an gpb.Booster", {
    bad_inputs <- list(
      matrix(1.0:10.0, 2L, 5L)
      , TRUE
      , c("a", "b")
      , NA
      , 10L
      , gpb.Dataset(
        data = matrix(1.0:10.0, 2L, 5L)
        , params = list()
      )
    )
    for (bad_input in bad_inputs) {
      expect_error({
        gpb.get.eval.result(
          booster = bad_input
          , data_name = "test"
          , eval_name = "l2"
        )
      }, regexp = "Can only use", fixed = TRUE)
    }
  })
  
  test_that("gpb.get.eval.result() should throw an informative error for incorrect data_name", {
    data(agaricus.train, package = "gpboost")
    data(agaricus.test, package = "gpboost")
    dtrain <- gpb.Dataset(
      agaricus.train$data
      , label = agaricus.train$label
    )
    model <- gpb.train(
      params = list(
        objective = "regression"
        , metric = "l2"
      )
      , data = dtrain
      , nrounds = 5L
      , valids = list(
        "test" = gpb.Dataset.create.valid(
          dtrain
          , agaricus.test$data
          , label = agaricus.test$label
        )
      )
      , min_data = 1L
      , learning_rate = 1.0
      , verbose = 0
    )
    expect_error({
      eval_results <- gpb.get.eval.result(
        booster = model
        , data_name = "testing"
        , eval_name = "l2"
      )
    }, regexp = "Only the following datasets exist in record evals: [test]", fixed = TRUE)
  })
  
  test_that("gpb.get.eval.result() should throw an informative error for incorrect eval_name", {
    data(agaricus.train, package = "gpboost")
    data(agaricus.test, package = "gpboost")
    dtrain <- gpb.Dataset(
      agaricus.train$data
      , label = agaricus.train$label
    )
    model <- gpb.train(
      params = list(
        objective = "regression"
        , metric = "l2"
      )
      , data = dtrain
      , nrounds = 5L
      , valids = list(
        "test" = gpb.Dataset.create.valid(
          dtrain
          , agaricus.test$data
          , label = agaricus.test$label
        )
      )
      , min_data = 1L
      , learning_rate = 1.0
      , verbose = 0
    )
    expect_error({
      eval_results <- gpb.get.eval.result(
        booster = model
        , data_name = "test"
        , eval_name = "l1"
      )
    }, regexp = "Only the following eval_names exist for dataset.*\\: \\[l2\\]", fixed = FALSE)
  })
  
  context("gpb.load()")
  
  test_that("gpb.load() gives the expected error messages given different incorrect inputs", {
    set.seed(708L)
    data(agaricus.train, package = "gpboost")
    data(agaricus.test, package = "gpboost")
    train <- agaricus.train
    test <- agaricus.test
    bst <- gpboost(
      data = as.matrix(train$data)
      , label = train$label
      , num_leaves = 4L
      , learning_rate = 1.0
      , nrounds = 2L
      , objective = "binary"
      , verbose = 0
    )
    
    # you have to give model_str or filename
    expect_error({
      gpb.load()
    }, regexp = "either filename or model_str must be given")
    expect_error({
      gpb.load(filename = NULL, model_str = NULL)
    }, regexp = "either filename or model_str must be given")
    
    # if given, filename should be a string that points to an existing file
    model_file <- tempfile(fileext = ".model")
    expect_error({
      gpb.load(filename = list(model_file))
    }, regexp = "filename should be character")
    file_to_check <- paste0("a.model")
    while (file.exists(file_to_check)) {
      file_to_check <- paste0("a", file_to_check)
    }
    expect_error({
      gpb.load(filename = file_to_check)
    }, regexp = "passed to filename does not exist")
    
    # if given, model_str should be a string
    expect_error({
      gpb.load(model_str = c(4.0, 5.0, 6.0))
    }, regexp = "model_str should be character")
    
  })
  
  test_that("Loading a Booster from a text file works", {
    set.seed(708L)
    data(agaricus.train, package = "gpboost")
    data(agaricus.test, package = "gpboost")
    train <- agaricus.train
    test <- agaricus.test
    bst <- gpboost(
      data = as.matrix(train$data)
      , label = train$label
      , num_leaves = 4L
      , learning_rate = 1.0
      , nrounds = 2L
      , objective = "binary"
      , verbose = 0
    )
    expect_true(gpboost:::gpb.is.Booster(bst))
    
    pred <- predict(bst, test$data)
    model_file <- tempfile(fileext = ".model")
    gpb.save(bst, model_file)
    
    # finalize the booster and destroy it so you know we aren't cheating
    bst$.__enclos_env__$private$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)
    
    bst2 <- gpb.load(
      filename = model_file
    )
    pred2 <- predict(bst2, test$data)
    expect_identical(pred, pred2)
  })
  
  test_that("boosters with linear models at leaves can be written to text file and re-loaded successfully", {
    X <- matrix(rnorm(100L), ncol = 1L)
    labels <- 2L * X + runif(nrow(X), 0L, 0.1)
    dtrain <- gpb.Dataset(
      data = X
      , label = labels
    )
    
    params <- list(
      objective = "regression"
      , verbose = -1L
      , metric = "mse"
      , seed = 0L
      , num_leaves = 2L
    )
    
    bst <- gpb.train(
      data = dtrain
      , nrounds = 10L
      , params = params
      , verbose = 0
    )
    expect_true(gpboost:::gpb.is.Booster(bst))
    
    # save predictions, then write the model to a file and destroy it in R
    preds <- predict(bst, X)
    model_file <- tempfile(fileext = ".model")
    gpb.save(bst, model_file)
    bst$.__enclos_env__$private$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)
    
    # load the booster and make predictions...should be the same
    bst2 <- gpb.load(
      filename = model_file
    )
    preds2 <- predict(bst2, X)
    expect_identical(preds, preds2)
  })
  
  
  test_that("Loading a Booster from a string works", {
    set.seed(708L)
    data(agaricus.train, package = "gpboost")
    data(agaricus.test, package = "gpboost")
    train <- agaricus.train
    test <- agaricus.test
    bst <- gpboost(
      data = as.matrix(train$data)
      , label = train$label
      , num_leaves = 4L
      , learning_rate = 1.0
      , nrounds = 2L
      , objective = "binary"
      , verbose = 0
    )
    expect_true(gpboost:::gpb.is.Booster(bst))
    
    pred <- predict(bst, test$data)
    model_string <- bst$save_model_to_string()
    
    # finalize the booster and destroy it so you know we aren't cheating
    bst$.__enclos_env__$private$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)
    
    bst2 <- gpb.load(
      model_str = model_string
    )
    pred2 <- predict(bst2, test$data)
    expect_identical(pred, pred2)
  })
  
  test_that("If a string and a file are both passed to gpb.load() the file is used model_str is totally ignored", {
    set.seed(708L)
    data(agaricus.train, package = "gpboost")
    data(agaricus.test, package = "gpboost")
    train <- agaricus.train
    test <- agaricus.test
    bst <- gpboost(
      data = as.matrix(train$data)
      , label = train$label
      , num_leaves = 4L
      , learning_rate = 1.0
      , nrounds = 2L
      , objective = "binary"
      , verbose = 0
    )
    expect_true(gpboost:::gpb.is.Booster(bst))
    
    pred <- predict(bst, test$data)
    model_file <- tempfile(fileext = ".model")
    gpb.save(bst, model_file)
    
    # finalize the booster and destroy it so you know we aren't cheating
    bst$.__enclos_env__$private$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)
    
    bst2 <- gpb.load(
      filename = model_file
      , model_str = 4.0
    )
    pred2 <- predict(bst2, test$data)
    expect_identical(pred, pred2)
  })
  
  context("Booster")
  
  test_that("Creating a Booster from a Dataset should work", {
    set.seed(708L)
    data(agaricus.train, package = "gpboost")
    data(agaricus.test, package = "gpboost")
    dtrain <- gpb.Dataset(
      agaricus.train$data
      , label = agaricus.train$label
    )
    bst <- gpboost:::Booster$new(
      params = list(
        objective = "binary"
      ),
      train_set = dtrain
      , verbose = 0
    )
    expect_true(gpboost:::gpb.is.Booster(bst))
    expect_equal(bst$current_iter(), 0L)
    expect_true(is.na(bst$best_score))
    expect_true(all(bst$predict(agaricus.train$data) == 0.5))
  })
  
  test_that("Creating a Booster from a Dataset with an existing predictor should work", {
    set.seed(708L)
    data(agaricus.train, package = "gpboost")
    nrounds <- 2L
    bst <- gpboost(
      data = as.matrix(agaricus.train$data)
      , label = agaricus.train$label
      , num_leaves = 4L
      , learning_rate = 1.0
      , nrounds = nrounds
      , objective = "binary"
      , verbose = 0
    )
    data(agaricus.test, package = "gpboost")
    dtest <- gpboost:::Dataset$new(
      data = agaricus.test$data
      , label = agaricus.test$label
      , predictor = bst$to_predictor()
    )
    capture.output( 
      bst_from_ds <- gpboost:::Booster$new(
        train_set = dtest
      )
      , file='NUL')
    expect_true(gpboost:::gpb.is.Booster(bst))
    expect_equal(bst$current_iter(), nrounds)
    expect_equal(bst$eval_train()[[1L]][["value"]], 0.1115352)
    expect_equal(bst_from_ds$current_iter(), nrounds)
    dumped_model <- RJSONIO::fromJSON(bst$dump_model())
    expect_identical(bst_from_ds$eval_train(), list())
    expect_equal(bst_from_ds$current_iter(), nrounds)
  })
  
  test_that("Booster$rollback_one_iter() should work as expected", {
    set.seed(708L)
    data(agaricus.train, package = "gpboost")
    data(agaricus.test, package = "gpboost")
    train <- agaricus.train
    test <- agaricus.test
    nrounds <- 5L
    bst <- gpboost(
      data = as.matrix(train$data)
      , label = train$label
      , num_leaves = 4L
      , learning_rate = 1.0
      , nrounds = nrounds
      , objective = "binary"
      , verbose = 0
    )
    expect_equal(bst$current_iter(), nrounds)
    expect_true(gpboost:::gpb.is.Booster(bst))
    logloss <- bst$eval_train()[[1L]][["value"]]
    expect_equal(logloss, 0.01991487)
    
    x <- bst$rollback_one_iter()
    
    # rollback_one_iter() should return a booster and modify the original
    # booster in place
    expect_true(gpboost:::gpb.is.Booster(x))
    expect_equal(bst$current_iter(), nrounds - 1L)
    
    # score should now come from the model as of 4 iterations
    logloss <- bst$eval_train()[[1L]][["value"]]
    expect_equal(logloss, 0.03150228)
  })
  
  test_that("Booster$update() passing a train_set works as expected", {
    set.seed(708L)
    data(agaricus.train, package = "gpboost")
    nrounds <- 2L
    
    # train with 2 rounds and then update
    bst <- gpboost(
      data = as.matrix(agaricus.train$data)
      , label = agaricus.train$label
      , num_leaves = 4L
      , learning_rate = 1.0
      , nrounds = nrounds
      , objective = "binary"
      , verbose = 0
    )
    expect_true(gpboost:::gpb.is.Booster(bst))
    expect_equal(bst$current_iter(), nrounds)
    capture.output( 
      bst$update(
        train_set = gpboost:::Dataset$new(
          data = agaricus.train$data
          , label = agaricus.train$label
        )
      )
      , file='NUL')
    expect_true(gpboost:::gpb.is.Booster(bst))
    expect_equal(bst$current_iter(), nrounds + 1L)
    
    # train with 3 rounds directly
    bst2 <- gpboost(
      data = as.matrix(agaricus.train$data)
      , label = agaricus.train$label
      , num_leaves = 4L
      , learning_rate = 1.0
      , nrounds = nrounds +  1L
      , objective = "binary"
      , verbose = 0
    )
    expect_true(gpboost:::gpb.is.Booster(bst2))
    expect_equal(bst2$current_iter(), nrounds +  1L)
    
    # model with 2 rounds + 1 update should be identical to 3 rounds
    expect_equal(bst2$eval_train()[[1L]][["value"]], 0.04806585)
    expect_equal(bst$eval_train()[[1L]][["value"]], bst2$eval_train()[[1L]][["value"]])
  })
  
  test_that("Booster$update() throws an informative error if you provide a non-Dataset to update()", {
    set.seed(708L)
    data(agaricus.train, package = "gpboost")
    nrounds <- 2L
    
    # train with 2 rounds and then update
    bst <- gpboost(
      data = as.matrix(agaricus.train$data)
      , label = agaricus.train$label
      , num_leaves = 4L
      , learning_rate = 1.0
      , nrounds = nrounds
      , objective = "binary"
      , verbose = 0
    )
    expect_error({
      bst$update(
        train_set = data.frame(x = rnorm(10L))
      )
    }, regexp = "gpb.Booster.update: Only can use gpb.Dataset", fixed = TRUE)
  })
  
  test_that("Booster should store parameters and Booster$reset_parameter() should update them", {
    data(agaricus.train, package = "gpboost")
    dtrain <- gpb.Dataset(
      agaricus.train$data
      , label = agaricus.train$label
    )
    # testing that this works for some cases that could break it:
    #    - multiple metrics
    #    - using "metric", "boosting", "num_class" in params
    params <- list(
      objective = "multiclass"
      , max_depth = 4L
      , bagging_fraction = 0.8
      , metric = c("multi_logloss", "multi_error")
      , boosting = "gbdt"
      , num_class = 5L
    )
    capture.output( 
      bst <- gpboost:::Booster$new(
        params = params
        , train_set = dtrain
      )
      , file='NUL')
    expect_identical(bst$params, params)
    
    params[["bagging_fraction"]] <- 0.9
    ret_bst <- bst$reset_parameter(params = params)
    expect_identical(ret_bst$params, params)
    expect_identical(bst$params, params)
  })
  
  test_that("Booster$params should include dataset params, before and after Booster$reset_parameter()", {
    data(agaricus.train, package = "gpboost")
    dtrain <- gpb.Dataset(
      agaricus.train$data
      , label = agaricus.train$label
      , params = list(
        max_bin = 17L
      )
    )
    params <- list(
      objective = "binary"
      , max_depth = 4L
      , bagging_fraction = 0.8
    )
    capture.output( 
      bst <- gpboost:::Booster$new(
        params = params
        , train_set = dtrain
      )
      , file='NUL')
    expect_identical(
      bst$params
      , list(
        objective = "binary"
        , max_depth = 4L
        , bagging_fraction = 0.8
        , max_bin = 17L
      )
    )
    
    params[["bagging_fraction"]] <- 0.9
    capture.output(
      ret_bst <- bst$reset_parameter(params = params)
      , file='NUL')
    expected_params <- list(
      objective = "binary"
      , max_depth = 4L
      , bagging_fraction = 0.9
      , max_bin = 17L
    )
    expect_identical(ret_bst$params, expected_params)
    expect_identical(bst$params, expected_params)
  })
  
  context("save_model")
  
  if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
    test_that("Saving a model with different feature importance types works", {
      set.seed(708L)
      data(agaricus.train, package = "gpboost")
      train <- agaricus.train
      bst <- gpboost(
        data = as.matrix(train$data)
        , label = train$label
        , num_leaves = 4L
        , learning_rate = 1.0
        , nrounds = 2L
        , objective = "binary"
        , verbose = 0
      )
      expect_true(gpboost:::gpb.is.Booster(bst))
      
      .feat_importance_from_string <- function(model_string) {
        file_lines <- strsplit(model_string, "\n")[[1L]]
        start_indx <- which(grepl("^feature_importances\\:$", file_lines)) + 1L
        blank_line_indices <- which(file_lines == "")
        end_indx <- blank_line_indices[blank_line_indices > start_indx][1L] - 1L
        importances <- file_lines[start_indx: end_indx]
        return(importances)
      }
      
      GAIN_IMPORTANCE <- 1L
      model_string <- bst$save_model_to_string(feature_importance_type = GAIN_IMPORTANCE)
      expect_equal(
        .feat_importance_from_string(model_string)
        , c(
          "odor=none=4010"
          , "stalk-root=club=1163"
          , "stalk-root=rooted=573"
          , "stalk-surface-above-ring=silky=450"
          , "spore-print-color=green=397"
          , "gill-color=buff=281"
        )
      )
      
      SPLIT_IMPORTANCE <- 0L
      model_string <- bst$save_model_to_string(feature_importance_type = SPLIT_IMPORTANCE)
      expect_equal(
        .feat_importance_from_string(model_string)
        , c(
          "odor=none=1"
          , "gill-color=buff=1"
          , "stalk-root=club=1"
          , "stalk-root=rooted=1"
          , "stalk-surface-above-ring=silky=1"
          , "spore-print-color=green=1"
        )
      )
    })
  }
  
  # test_that("Saving a model with unknown importance type fails", {
  #     testthat::skip("Skipping this test because it causes issues for valgrind")
  #     set.seed(708L)
  #     data(agaricus.train, package = "gpboost")
  #     train <- agaricus.train
  #     bst <- gpboost(
  #         data = as.matrix(train$data)
  #         , label = train$label
  #         , num_leaves = 4L
  #         , learning_rate = 1.0
  #         , nrounds = 2L
  #         , objective = "binary"
  #     )
  #     expect_true(gpboost:::gpb.is.Booster(bst))
  # 
  #     UNSUPPORTED_IMPORTANCE <- 2L
  #     expect_error({
  #         model_string <- bst$save_model_to_string(feature_importance_type = UNSUPPORTED_IMPORTANCE)
  #     }, "Unknown importance type")
  # })
  
  
  .params_from_model_string <- function(model_str) {
    file_lines <- strsplit(model_str, "\n")[[1L]]
    start_indx <- which(grepl("^parameters\\:$", file_lines)) + 1L
    blank_line_indices <- which(file_lines == "")
    end_indx <- blank_line_indices[blank_line_indices > start_indx][1L] - 1L
    params <- file_lines[start_indx: end_indx]
    return(params)
  }
  
  test_that("all parameters are stored correctly with save_model_to_string()", {
    dtrain <- gpb.Dataset(
      data = matrix(rnorm(500L), nrow = 100L)
      , label = rnorm(100L)
    )
    nrounds <- 4L
    bst <- gpb.train(
      params = list(
        objective = "regression"
        , metric = "l2"
      )
      , data = dtrain
      , nrounds = nrounds
      , verbose = 0L
    )
    
    model_str <- bst$save_model_to_string()
    params_in_file <- .params_from_model_string(model_str = model_str)
    
    # parameters should match what was passed from the R package
    expect_equal(sum(grepl(pattern = "^\\[metric\\:", x = params_in_file)), 1L)
    expect_equal(sum(params_in_file == "[metric: l2]"), 1L)
    
    expect_equal(sum(grepl(pattern = "^\\[num_iterations\\:", x = params_in_file)), 1L)
    expect_equal(sum(params_in_file == "[num_iterations: 4]"), 1L)
    
    expect_equal(sum(grepl(pattern = "^\\[objective\\:", x = params_in_file)), 1L)
    expect_equal(sum(params_in_file == "[objective: regression]"), 1L)
    
    expect_equal(sum(grepl(pattern = "^\\[verbosity\\:", x = params_in_file)), 1L)
    expect_equal(sum(params_in_file == "[verbosity: 0]"), 1L)
    
    # early stopping should be off by default
    expect_equal(sum(grepl(pattern = "^\\[early_stopping_round\\:", x = params_in_file)), 1L)
    expect_equal(sum(params_in_file == "[early_stopping_round: 0]"), 1L)
  })
  
  test_that("early_stopping, num_iterations are stored correctly in model string even with aliases", {
    dtrain <- gpb.Dataset(
      data = matrix(rnorm(500L), nrow = 100L)
      , label = rnorm(100L)
    )
    dvalid <- gpb.Dataset(
      data = matrix(rnorm(500L), nrow = 100L)
      , label = rnorm(100L)
    )
    
    # num_iterations values (all different)
    num_iterations <- 4L
    num_boost_round <- 2L
    n_iter <- 3L
    nrounds_kwarg <- 6L
    
    # early_stopping_round values (all different)
    early_stopping_round <- 2L
    early_stopping_round_kwarg <- 3L
    n_iter_no_change <- 4L
    
    params <- list(
      objective = "regression"
      , metric = "l2"
      , num_boost_round = num_boost_round
      , num_iterations = num_iterations
      , n_iter = n_iter
      , early_stopping_round = early_stopping_round
      , n_iter_no_change = n_iter_no_change
    )
    
    bst <- gpb.train(
      params = params
      , data = dtrain
      , nrounds = nrounds_kwarg
      , early_stopping_rounds = early_stopping_round_kwarg
      , valids = list(
        "random_valid" = dvalid
      )
      , verbose = 0L
    )
    
    model_str <- bst$save_model_to_string()
    params_in_file <- .params_from_model_string(model_str = model_str)
    
    # parameters should match what was passed from the R package, and the "main" (non-alias)
    # params values in `params` should be preferred to keyword argumentts or aliases
    expect_equal(sum(grepl(pattern = "^\\[num_iterations\\:", x = params_in_file)), 1L)
    expect_equal(sum(params_in_file == sprintf("[num_iterations: %s]", num_iterations)), 1L)
    expect_equal(sum(grepl(pattern = "^\\[early_stopping_round\\:", x = params_in_file)), 1L)
    expect_equal(sum(params_in_file == sprintf("[early_stopping_round: %s]", early_stopping_round)), 1L)
    
    # none of the aliases shouold have been written to the model file
    expect_equal(sum(grepl(pattern = "^\\[num_boost_round\\:", x = params_in_file)), 0L)
    expect_equal(sum(grepl(pattern = "^\\[n_iter\\:", x = params_in_file)), 0L)
    expect_equal(sum(grepl(pattern = "^\\[n_iter_no_change\\:", x = params_in_file)), 0L)
    
  })
  
  # this is almost identical to the test above it, but for gpb.cv(). A lot of code
  # is duplicated between gpb.train() and gpb.cv(), and this will catch cases where
  # one is updated and the other isn't
  test_that("gpb.cv() correctly handles passing through params to the model file", {
    dtrain <- gpb.Dataset(
      data = matrix(rnorm(500L), nrow = 100L)
      , label = rnorm(100L)
    )
    
    # num_iterations values (all different)
    num_iterations <- 4L
    num_boost_round <- 2L
    n_iter <- 3L
    nrounds_kwarg <- 6L
    
    # early_stopping_round values (all different)
    early_stopping_round <- 2L
    early_stopping_round_kwarg <- 3L
    n_iter_no_change <- 4L
    
    params <- list(
      objective = "regression"
      , metric = "l2"
      , num_boost_round = num_boost_round
      , num_iterations = num_iterations
      , n_iter = n_iter
      , early_stopping_round = early_stopping_round
      , n_iter_no_change = n_iter_no_change
    )
    
    cv_bst <- gpb.cv(
      params = params
      , data = dtrain
      , nrounds = nrounds_kwarg
      , early_stopping_rounds = early_stopping_round_kwarg
      , nfold = 3L
      , verbose = 0L
    )
    
    for (bst in cv_bst$boosters) {
      model_str <- bst[["booster"]]$save_model_to_string()
      params_in_file <- .params_from_model_string(model_str = model_str)
      
      # parameters should match what was passed from the R package, and the "main" (non-alias)
      # params values in `params` should be preferred to keyword argumentts or aliases
      expect_equal(sum(grepl(pattern = "^\\[num_iterations\\:", x = params_in_file)), 1L)
      expect_equal(sum(params_in_file == sprintf("[num_iterations: %s]", num_iterations)), 1L)
      expect_equal(sum(grepl(pattern = "^\\[early_stopping_round\\:", x = params_in_file)), 1L)
      expect_equal(sum(params_in_file == sprintf("[early_stopping_round: %s]", early_stopping_round)), 1L)
      
      # none of the aliases shouold have been written to the model file
      expect_equal(sum(grepl(pattern = "^\\[num_boost_round\\:", x = params_in_file)), 0L)
      expect_equal(sum(grepl(pattern = "^\\[n_iter\\:", x = params_in_file)), 0L)
      expect_equal(sum(grepl(pattern = "^\\[n_iter_no_change\\:", x = params_in_file)), 0L)
    }
    
  })
  
  context("saveRDS.gpb.Booster() and readRDS.gpb.Booster()")
  
  test_that("params (including dataset params) should be stored in .rds file for Booster", {
    data(agaricus.train, package = "gpboost")
    dtrain <- gpb.Dataset(
      agaricus.train$data
      , label = agaricus.train$label
      , params = list(
        max_bin = 17L
      )
    )
    params <- list(
      objective = "binary"
      , max_depth = 4L
      , bagging_fraction = 0.8
    )
    capture.output( 
      bst <- gpboost:::Booster$new(
        params = params
        , train_set = dtrain
      )
      , file='NUL')
    bst_file <- tempfile(fileext = ".rds")
    saveRDS.gpb.Booster(bst, file = bst_file)
    
    bst_from_file <- readRDS.gpb.Booster(file = bst_file)
    expect_identical(
      bst_from_file$params
      , list(
        objective = "binary"
        , max_depth = 4L
        , bagging_fraction = 0.8
        , max_bin = 17L
      )
    )
  })
  
  test_that("boosters with linear models at leaves can be written to RDS and re-loaded successfully", {
    X <- matrix(rnorm(100L), ncol = 1L)
    labels <- 2L * X + runif(nrow(X), 0L, 0.1)
    dtrain <- gpb.Dataset(
      data = X
      , label = labels
    )
    
    params <- list(
      objective = "regression"
      , verbose = -1L
      , metric = "mse"
      , seed = 0L
      , num_leaves = 2L
    )
    
    bst <- gpb.train(
      data = dtrain
      , nrounds = 10L
      , params = params
      , verbose = 0
    )
    expect_true(gpboost:::gpb.is.Booster(bst))
    
    # save predictions, then write the model to a file and destroy it in R
    preds <- predict(bst, X)
    model_file <- tempfile(fileext = ".rds")
    saveRDS.gpb.Booster(bst, file = model_file)
    bst$.__enclos_env__$private$finalize()
    expect_null(bst$.__enclos_env__$private$handle)
    rm(bst)
    
    # load the booster and make predictions...should be the same
    bst2 <- readRDS.gpb.Booster(file = model_file)
    preds2 <- predict(bst2, X)
    expect_identical(preds, preds2)
  })
  
} else {
  Sys.sleep(30)
}
