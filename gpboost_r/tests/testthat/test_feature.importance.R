context("feature.importance")

test_that("feature.importance() should reject bad inputs", {
    bad_inputs <- list(
        .Machine$integer.max
        , Inf
        , -Inf
        , NA
        , NA_real_
        , -10L:10L
        , list(c("a", "b", "c"))
        , data.frame(
            x = rnorm(20)
            , y = sample(
                x = c(1, 2)
                , size = 20
                , replace = TRUE
            )
        )
        , data.table::data.table(
            x = rnorm(20)
            , y = sample(
                x = c(1, 2)
                , size = 20
                , replace = TRUE
            )
        )
        , gpb.Dataset(
            data = matrix(rnorm(100), ncol = 2)
            , label = matrix(sample(c(0, 1), 50, replace = TRUE))
        )
        , "gpboost.model"
    )
    for (input in bad_inputs){
        expect_error({
            feature.importance(input)
        }, regexp = "'model' has to be an object of class gpb\\.Booster")
    }
})
