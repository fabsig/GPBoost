context("num_parallel_threads")

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){

  test_that("'num_parallel_threads' of a model does not change the number of threads of other models ", {

    num_threads_before <- gpb.get.num.threads()
    expect_gte(num_threads_before, 1L)

    n <- 100
    group <- rep(1:10, each = n / 10)
    y <- rep(c(-1, 1), n / 2) + 0.1 * (1:n)
    # A model that uses only one thread must not change the number of threads used afterwards
    capture.output( gp_model <- fitGPModel(group_data = group, y = y, params = list(maxit = 5),
                                           num_parallel_threads = 1L) , file = 'NUL')
    expect_equal(gpb.get.num.threads(), num_threads_before)
    # ... also not while the model still exists and is used
    capture.output( pred <- predict(gp_model, group_data_pred = group[1:5], predict_var = TRUE) , file = 'NUL')
    expect_equal(gpb.get.num.threads(), num_threads_before)
    # ... and a model created afterwards without 'num_parallel_threads' uses the default number of threads
    capture.output( gp_model2 <- fitGPModel(group_data = group, y = y, params = list(maxit = 5)) , file = 'NUL')
    expect_equal(gpb.get.num.threads(), num_threads_before)
    # Both models give the same results (the number of threads only affects the order of summation)
    expect_lt(abs(gp_model$get_cov_pars(std_err = FALSE)[1] - gp_model2$get_cov_pars(std_err = FALSE)[1]), 1E-6)

    # Setting the number of threads explicitly
    gpb.set.num.threads(2L)
    expect_equal(gpb.get.num.threads(), 2L)
    # A model with its own number of threads does not change this
    capture.output( gp_model3 <- fitGPModel(group_data = group, y = y, params = list(maxit = 5),
                                            num_parallel_threads = 1L) , file = 'NUL')
    expect_equal(gpb.get.num.threads(), 2L)
    # Non-positive values reset to the default number of threads
    gpb.set.num.threads(-1L)
    expect_equal(gpb.get.num.threads(), num_threads_before)
    expect_error(gpb.set.num.threads("two"), "num_threads needs to be an integer of length one", fixed = TRUE)

  })

}
