/*!
* Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#include <LightGBM/metric.h>

#include "binary_metric.hpp"
#include "map_metric.hpp"
#include "multiclass_metric.hpp"
#include "rank_metric.hpp"
#include "regression_metric.hpp"
#include "xentropy_metric.hpp"
#include "random_effects_metric.hpp"

namespace LightGBM {

Metric* Metric::CreateMetric(const std::string& type, const Config& config) {
  if (type == std::string("l2")) {
    return new L2Metric(config);
  } else if (type == std::string("rmse")) {
    return new RMSEMetric(config);
  } else if (type == std::string("l1")) {
    return new L1Metric(config);
  } else if (type == std::string("quantile")) {
    return new QuantileMetric(config);
  } else if (type == std::string("huber")) {
    return new HuberLossMetric(config);
  } else if (type == std::string("fair")) {
    return new FairLossMetric(config);
  } else if (type == std::string("poisson")) {
    return new PoissonMetric(config);
  } else if (type == std::string("binary_logloss")) {
    return new BinaryLoglossMetric(config);
  } else if (type == std::string("binary_error")) {
    return new BinaryErrorMetric(config);
  } else if (type == std::string("auc")) {
    return new AUCMetric(config);
  } else if (type == std::string("average_precision")) {
    return new AveragePrecisionMetric(config);
  } else if (type == std::string("auc_mu")) {
    return new AucMuMetric(config);
  } else if (type == std::string("ndcg")) {
    return new NDCGMetric(config);
  } else if (type == std::string("map")) {
    return new MapMetric(config);
  } else if (type == std::string("multi_logloss")) {
    return new MultiSoftmaxLoglossMetric(config);
  } else if (type == std::string("multi_error")) {
    return new MultiErrorMetric(config);
  } else if (type == std::string("cross_entropy")) {
    return new CrossEntropyMetric(config);
  } else if (type == std::string("cross_entropy_lambda")) {
    return new CrossEntropyLambdaMetric(config);
  } else if (type == std::string("kullback_leibler")) {
    return new KullbackLeiblerDivergence(config);
  } else if (type == std::string("mape")) {
    return new MAPEMetric(config);
  } else if (type == std::string("gamma")) {
    return new GammaMetric(config);
  } else if (type == std::string("gamma_deviance")) {
    return new GammaDevianceMetric(config);
  } else if (type == std::string("tweedie")) {
    return new TweedieMetric(config);
  } else if (type == std::string("approx_neg_marginal_log_likelihood")) {
      return new LatenGaussianLaplace(config);
  } else if (type == std::string("neg_log_likelihood")) {
      return new NegLogLikelihood(config);
  } else if (type == std::string("gaussian_neg_log_likelihood")) {
      Log::Fatal("The metric 'gaussian_neg_log_likelihood' is no longer supported. "
          "Please use the equivalent metric 'test_neg_log_likelihood' instead ");
  } else if (type == std::string("test_neg_log_likelihood")) {
      return new TestNegLogLikelihood(config);
  } else if (type == std::string("crps_gaussian")) {
      return new CRPSGaussian(config);
  }
  return nullptr;
}

}  // namespace LightGBM
