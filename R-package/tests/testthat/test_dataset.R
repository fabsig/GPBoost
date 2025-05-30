
context("testing gpb.Dataset functionality")

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  data(agaricus.test, package = "gpboost")
  test_data <- agaricus.test$data[1L:100L, ]
  test_label <- agaricus.test$label[1L:100L]
  
  test_that("gpb.Dataset: basic construction, saving, loading", {
    # from sparse matrix
    dtest1 <- gpb.Dataset(test_data, label = test_label)
    # from dense matrix
    dtest2 <- gpb.Dataset(as.matrix(test_data), label = test_label)
    expect_equal(getinfo(dtest1, "label"), getinfo(dtest2, "label"))
    
    # save to a local file
    tmp_file <- tempfile("gpb.Dataset_")
    capture.output( 
      gpb.Dataset.save(dtest1, tmp_file)
      , file='NUL')
    # read from a local file
    dtest3 <- gpb.Dataset(tmp_file)
    capture.output( 
      gpb.Dataset.construct(dtest3)
      , file='NUL')
    unlink(tmp_file)
    expect_equal(getinfo(dtest1, "label"), getinfo(dtest3, "label"))
  })
  
  test_that("gpb.Dataset: getinfo & setinfo", {
    dtest <- gpb.Dataset(test_data)
    dtest$construct()
    
    setinfo(dtest, "label", test_label)
    labels <- getinfo(dtest, "label")
    expect_equal(test_label, getinfo(dtest, "label"))
    
    expect_true(length(getinfo(dtest, "weight")) == 0L)
    expect_true(length(getinfo(dtest, "init_score")) == 0L)
    
    # any other label should error
    expect_error(setinfo(dtest, "asdf", test_label))
  })
  
  test_that("gpb.Dataset: slice, dim", {
    dtest <- gpb.Dataset(test_data, label = test_label)
    gpb.Dataset.construct(dtest)
    expect_equal(dim(dtest), dim(test_data))
    dsub1 <- gpboost::slice(dtest, seq_len(42L))
    gpb.Dataset.construct(dsub1)
    expect_equal(nrow(dsub1), 42L)
    expect_equal(ncol(dsub1), ncol(test_data))
  })
  
  test_that("gpb.Dataset: colnames", {
    dtest <- gpb.Dataset(test_data, label = test_label)
    expect_equal(colnames(dtest), colnames(test_data))
    gpb.Dataset.construct(dtest)
    expect_equal(colnames(dtest), colnames(test_data))
    expect_error({
      colnames(dtest) <- "asdf"
    })
    new_names <- make.names(seq_len(ncol(test_data)))
    expect_silent(colnames(dtest) <- new_names)
    expect_equal(colnames(dtest), new_names)
  })
  
  test_that("gpb.Dataset: nrow is correct for a very sparse matrix", {
    nr <- 1000L
    x <- Matrix::rsparsematrix(nr, 100L, density = 0.0005)
    # we want it very sparse, so that last rows are empty
    expect_lt(max(x@i), nr)
    dtest <- gpb.Dataset(x)
    expect_equal(dim(dtest), dim(x))
  })
  
  test_that("gpb.Dataset: Dataset should be able to construct from matrix and return non-null handle", {
    rawData <- matrix(runif(1000L), ncol = 10L)
    ref_handle <- NULL
    handle <- .Call(
      gpboost:::LGBM_DatasetCreateFromMat_R
      , rawData
      , nrow(rawData)
      , ncol(rawData)
      , gpboost:::gpb.params2str(params = list())
      , ref_handle
    )
    expect_is(handle, "externalptr")
    expect_false(is.null(handle))
    .Call(gpboost:::LGBM_DatasetFree_R, handle)
    handle <- NULL
  })
  
  test_that("cpp errors should be raised as proper R errors", {
    data(agaricus.train, package = "gpboost")
    train <- agaricus.train
    dtrain <- gpb.Dataset(
      train$data
      , label = train$label
      , init_score = seq_len(10L)
    )
    expect_error({
      dtrain$construct()
    }, regexp = "Initial score size doesn't match data size")
  })
  
  test_that("gpb.Dataset$setinfo() should convert 'group' to integer", {
    ds <- gpb.Dataset(
      data = matrix(rnorm(100L), nrow = 50L, ncol = 2L)
      , label = sample(c(0L, 1L), size = 50L, replace = TRUE)
    )
    ds$construct()
    current_group <- ds$getinfo("group")
    expect_null(current_group)
    group_as_numeric <- rep(25.0, 2L)
    ds$setinfo("group", group_as_numeric)
    expect_identical(ds$getinfo("group"), as.integer(group_as_numeric))
  })
  
  test_that("gpb.Dataset should throw an error if 'reference' is provided but of the wrong format", {
    data(agaricus.test, package = "gpboost")
    test_data <- agaricus.test$data[1L:100L, ]
    test_label <- agaricus.test$label[1L:100L]
    # Try to trick gpb.Dataset() into accepting bad input
    expect_error({
      dtest <- gpb.Dataset(
        data = test_data
        , label = test_label
        , reference = data.frame(x = seq_len(10L), y = seq_len(10L))
      )
    }, regexp = "reference must be a")
  })
  
  test_that("Dataset$get_params() successfully returns parameters if you passed them", {
    # note that this list uses one "main" parameter (feature_pre_filter) and one that
    # is an alias (is_sparse), to check that aliases are handled correctly
    params <- list(
      "feature_pre_filter" = TRUE
      , "is_sparse" = FALSE
    )
    ds <- gpb.Dataset(
      test_data
      , label = test_label
      , params = params
    )
    returned_params <- ds$get_params()
    expect_identical(class(returned_params), "list")
    expect_identical(length(params), length(returned_params))
    expect_identical(sort(names(params)), sort(names(returned_params)))
    for (param_name in names(params)) {
      expect_identical(params[[param_name]], returned_params[[param_name]])
    }
  })
  
  test_that("Dataset$get_params() ignores irrelevant parameters", {
    params <- list(
      "feature_pre_filter" = TRUE
      , "is_sparse" = FALSE
      , "nonsense_parameter" = c(1.0, 2.0, 5.0)
    )
    ds <- gpb.Dataset(
      test_data
      , label = test_label
      , params = params
    )
    returned_params <- ds$get_params()
    expect_false("nonsense_parameter" %in% names(returned_params))
  })
  
  test_that("Dataset$update_parameters() does nothing for empty inputs", {
    ds <- gpb.Dataset(
      test_data
      , label = test_label
    )
    initial_params <- ds$get_params()
    expect_identical(initial_params, list())
    
    # update_params() should return "self" so it can be chained
    res <- ds$update_params(
      params = list()
    )
    expect_true(gpboost:::gpb.is.Dataset(res))
    
    new_params <- ds$get_params()
    expect_identical(new_params, initial_params)
  })
  
  test_that("Dataset$update_params() works correctly for recognized Dataset parameters", {
    ds <- gpb.Dataset(
      test_data
      , label = test_label
    )
    initial_params <- ds$get_params()
    expect_identical(initial_params, list())
    
    new_params <- list(
      "data_random_seed" = 708L
      , "enable_bundle" = FALSE
    )
    res <- ds$update_params(
      params = new_params
    )
    expect_true(gpboost:::gpb.is.Dataset(res))
    
    updated_params <- ds$get_params()
    for (param_name in names(new_params)) {
      expect_identical(new_params[[param_name]], updated_params[[param_name]])
    }
  })
  
  test_that("Dataset$finalize() should not fail on an already-finalized Dataset", {
    dtest <- gpb.Dataset(
      data = test_data
      , label = test_label
    )
    expect_true(gpboost:::gpb.is.null.handle(dtest$.__enclos_env__$private$handle))
    
    dtest$construct()
    expect_false(gpboost:::gpb.is.null.handle(dtest$.__enclos_env__$private$handle))
    
    dtest$.__enclos_env__$private$finalize()
    expect_true(gpboost:::gpb.is.null.handle(dtest$.__enclos_env__$private$handle))
    
    # calling finalize() a second time shouldn't cause any issues
    dtest$.__enclos_env__$private$finalize()
    expect_true(gpboost:::gpb.is.null.handle(dtest$.__enclos_env__$private$handle))
  })
  
  test_that("gpb.Dataset: should be able to run gpb.train() immediately after using gpb.Dataset() on a file", {
    dtest <- gpb.Dataset(
      data = test_data
      , label = test_label
    )
    tmp_file <- tempfile(pattern = "gpb.Dataset_")
    capture.output( 
      gpb.Dataset.save(
        dataset = dtest
        , fname = tmp_file
      )
      , file='NUL')
    
    # read from a local file
    dtest_read_in <- gpb.Dataset(data = tmp_file)
    
    param <- list(
      objective = "binary"
      , metric = "binary_logloss"
      , num_leaves = 5L
      , learning_rate = 1.0
    )
    
    # should be able to train right away
    bst <- gpb.train(
      params = param
      , data = dtest_read_in
      , verbose = 0
    )
    
    expect_true(gpboost:::gpb.is.Booster(x = bst))
  })
  
  test_that("gpb.Dataset: should be able to run gpb.cv() immediately after using gpb.Dataset() on a file", {
    dtest <- gpb.Dataset(
      data = test_data
      , label = test_label
    )
    tmp_file <- tempfile(pattern = "gpb.Dataset_")
    capture.output( 
      gpb.Dataset.save(
        dataset = dtest
        , fname = tmp_file
      )
      , file='NUL')
    
    # read from a local file
    dtest_read_in <- gpb.Dataset(data = tmp_file)
    
    param <- list(
      objective = "binary"
      , metric = "binary_logloss"
      , num_leaves = 5L
      , learning_rate = 1.0
    )
    
    # should be able to train right away
    bst <- gpb.cv(
      params = param
      , data = dtest_read_in
      , verbose = 0
    )
    
    expect_is(bst, "gpb.CVBooster")
  })
  
}
