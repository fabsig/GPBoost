context("gpb.importance")

# Avoid that long tests get executed on CRAN
if(Sys.getenv("GPBOOST_ALL_TESTS") == "GPBOOST_ALL_TESTS"){
  
  test_that("gpb.importance() should reject bad inputs", {
    bad_inputs <- list(
      .Machine$integer.max
      , Inf
      , -Inf
      , NA
      , NA_real_
      , -10L:10L
      , list(c("a", "b", "c"))
      , data.frame(
        x = rnorm(20L)
        , y = sample(
          x = c(1L, 2L)
          , size = 20L
          , replace = TRUE
        )
      )
      , data.table::data.table(
        x = rnorm(20L)
        , y = sample(
          x = c(1L, 2L)
          , size = 20L
          , replace = TRUE
        )
      )
      , gpb.Dataset(
        data = matrix(rnorm(100L), ncol = 2L)
        , label = matrix(sample(c(0L, 1L), 50L, replace = TRUE))
      )
      , "gpboost.model"
    )
    for (input in bad_inputs) {
      expect_error({
        gpb.importance(input)
      }, regexp = "'model' has to be an object of class gpb\\.Booster")
    }
  })
  
}
