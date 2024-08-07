context("gpb.plot.importance()")

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  test_that("gpb.plot.importance() should run without error for well-formed inputs", {
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
    model <- gpb.train(params, dtrain, 3L, verbose = 0)
    tree_imp <- gpb.importance(model, percentage = TRUE)
    
    # # Check that there are no plots present before plotting
    # expect_null(dev.list())
    
    args_no_cex <- list(
      "tree_imp" = tree_imp
      , top_n = 10L
      , measure = "Gain"
    )
    args_cex <- args_no_cex
    args_cex[["cex"]] <- 0.75
    
    for (arg_list in list(args_no_cex, args_cex)) {
      
      resDT <- do.call(
        what = gpb.plot.importance
        , args = arg_list
      )
      
      # Check that gpb.plot.importance() returns the data.table of the plotted data
      expect_true(data.table::is.data.table(resDT))
      expect_named(resDT, c("Feature", "Gain", "Cover", "Frequency"))
      
      # Check that a plot was produced
      expect_false(is.null(dev.list()))
      
      # remove all plots
      dev.off()
      expect_null(dev.list())
    }
  })
  
}
