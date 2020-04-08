require(gpboost)
require(Matrix)

context("testing gpb.Dataset functionality")

if(R.Version()$arch != "i386"){##32-bit version is not supported
  data(agaricus.test, package='gpboost')
  test_data <- agaricus.test$data[1:100,]
  test_label <- agaricus.test$label[1:100]
  
  test_that("gpb.Dataset: basic construction, saving, loading", {
    # from sparse matrix
    dtest1 <- gpb.Dataset(test_data, label=test_label)
    # from dense matrix
    dtest2 <- gpb.Dataset(as.matrix(test_data), label=test_label)
    expect_equal(getinfo(dtest1, 'label'), getinfo(dtest2, 'label'))
    
    # save to a local file
    tmp_file <- tempfile('gpb.Dataset_')
    gpb.Dataset.save(dtest1, tmp_file)
    # read from a local file
    dtest3 <- gpb.Dataset(tmp_file)
    gpb.Dataset.construct(dtest3)
    unlink(tmp_file)
    expect_equal(getinfo(dtest1, 'label'), getinfo(dtest3, 'label'))
  })
  
  test_that("gpb.Dataset: getinfo & setinfo", {
    dtest <- gpb.Dataset(test_data)
    dtest$construct()
    
    setinfo(dtest, 'label', test_label)
    labels <- getinfo(dtest, 'label')
    expect_equal(test_label, getinfo(dtest, 'label'))
    
    expect_true(length(getinfo(dtest, 'weight')) == 0)
    expect_true(length(getinfo(dtest, 'init_score')) == 0)
    
    # any other label should error
    expect_error(setinfo(dtest, 'asdf', test_label))
  })
  
  test_that("gpb.Dataset: slice, dim", {
    dtest <- gpb.Dataset(test_data, label=test_label)
    gpb.Dataset.construct(dtest)
    expect_equal(dim(dtest), dim(test_data))
    dsub1 <- slice(dtest, 1:42)
    gpb.Dataset.construct(dsub1)
    expect_equal(nrow(dsub1), 42)
    expect_equal(ncol(dsub1), ncol(test_data))
  })
  
  test_that("gpb.Dataset: colnames", {
    dtest <- gpb.Dataset(test_data, label=test_label)
    expect_equal(colnames(dtest), colnames(test_data))
    gpb.Dataset.construct(dtest)
    expect_equal(colnames(dtest), colnames(test_data))
    expect_error( colnames(dtest) <- 'asdf')
    new_names <- make.names(1:ncol(test_data))
    expect_silent(colnames(dtest) <- new_names)
    expect_equal(colnames(dtest), new_names)
  })
  
  test_that("gpb.Dataset: nrow is correct for a very sparse matrix", {
    nr <- 1000
    x <- Matrix::rsparsematrix(nr, 100, density=0.0005)
    # we want it very sparse, so that last rows are empty
    expect_lt(max(x@i), nr)
    dtest <- gpb.Dataset(x)
    expect_equal(dim(dtest), dim(x))
  })
  
  test_that("gpb.Dataset: Dataset should be able to construct from matrix and return non-null handle", {
    rawData <- matrix(runif(1000),ncol=10)
    handle <- NA_real_
    ref_handle <- NULL
    handle <- gpboost:::gpb.call("LGBM_DatasetCreateFromMat_R"
                                 , ret = handle
                                 , rawData
                                 , nrow(rawData)
                                 , ncol(rawData)
                                 , gpboost:::gpb.params2str(params=list())
                                 , ref_handle)
    expect_false(is.na(handle))
  })
}
