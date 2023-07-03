/*!
* Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#include <LightGBM/objective_function.h>

#include "binary_objective.hpp"
#include "multiclass_objective.hpp"
#include "rank_objective.hpp"
#include "regression_objective.hpp"
#include "xentropy_objective.hpp"

namespace LightGBM {

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction(const std::string& type, const Config& config) {
  if (type == std::string("regression")) {
    return new RegressionL2loss(config);
  } else if (type == std::string("regression_l1")) {
    return new RegressionL1loss(config);
  } else if (type == std::string("quantile")) {
    return new RegressionQuantileloss(config);
  } else if (type == std::string("huber")) {
    return new RegressionHuberLoss(config);
  } else if (type == std::string("fair")) {
    return new RegressionFairLoss(config);
  } else if (type == std::string("poisson")) {
    return new RegressionPoissonLoss(config);
  } else if (type == std::string("bernoulli_logit") || type == std::string("binary")) {
    return new BinaryLogloss(config);
  } else if (type == std::string("lambdarank")) {
    return new LambdarankNDCG(config);
  } else if (type == std::string("rank_xendcg")) {
    return new RankXENDCG(config);
  } else if (type == std::string("multiclass")) {
    return new MulticlassSoftmax(config);
  } else if (type == std::string("multiclassova")) {
    return new MulticlassOVA(config);
  } else if (type == std::string("cross_entropy")) {
    return new CrossEntropy(config);
  } else if (type == std::string("cross_entropy_lambda")) {
    return new CrossEntropyLambda(config);
  } else if (type == std::string("mape")) {
    return new RegressionMAPELOSS(config);
  } else if (type == std::string("gamma")) {
    return new RegressionGammaLoss(config);
  } else if (type == std::string("tweedie")) {
    return new RegressionTweedieLoss(config);
  } else if (type == std::string("tobit")) {
      return new TobitLoss(config);
  } else if (type == std::string("custom")) {
    return nullptr;
  }
  Log::Fatal("Unknown objective type name: %s", type.c_str());
  return nullptr;
}

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction(const std::string& str) {
  auto strs = Common::Split(str.c_str(), ' ');
  auto type = strs[0];
  if (type == std::string("regression") || type == std::string("gaussian")) {
    return new RegressionL2loss(strs);
  } else if (type == std::string("regression_l1")) {
    return new RegressionL1loss(strs);
  } else if (type == std::string("quantile")) {
    return new RegressionQuantileloss(strs);
  } else if (type == std::string("huber")) {
    return new RegressionHuberLoss(strs);
  } else if (type == std::string("fair")) {
    return new RegressionFairLoss(strs);
  } else if (type == std::string("poisson")) {
    return new RegressionPoissonLoss(strs);
  } else if (type == std::string("bernoulli_logit") || type == std::string("binary")) {
    return new BinaryLogloss(strs);
  } else if (type == std::string("lambdarank")) {
    return new LambdarankNDCG(strs);
  } else if (type == std::string("rank_xendcg")) {
    return new RankXENDCG(strs);
  } else if (type == std::string("multiclass")) {
    return new MulticlassSoftmax(strs);
  } else if (type == std::string("multiclassova")) {
    return new MulticlassOVA(strs);
  } else if (type == std::string("cross_entropy")) {
    return new CrossEntropy(strs);
  } else if (type == std::string("cross_entropy_lambda")) {
    return new CrossEntropyLambda(strs);
  } else if (type == std::string("mape")) {
    return new RegressionMAPELOSS(strs);
  } else if (type == std::string("gamma")) {
    return new RegressionGammaLoss(strs);
  } else if (type == std::string("tweedie")) {
    return new RegressionTweedieLoss(strs);
  } else if (type == std::string("tobit")) {
      return new TobitLoss(strs);
  } else if (type == std::string("custom")) {
    return nullptr;
  }
  Log::Fatal("Unknown objective type name: %s", type.c_str());
  return nullptr;
}

void ObjectiveFunction::InitGPModel(REModel* re_model,
    bool train_gp_model_cov_pars,
    bool use_gp_model_for_validation,
    const label_t* label) {
    CHECK(re_model != nullptr);
    re_model_ = re_model;
    if (train_gp_model_cov_pars) {
        re_model_->ResetCovPars();
    }
    has_gp_model_ = true;
    train_gp_model_cov_pars_ = train_gp_model_cov_pars;
    use_gp_model_for_validation_ = use_gp_model_for_validation;
    if (!(re_model_->GaussLikelihood())) {
        re_model_->SetY(label);
        re_model_->InitializeCovParsIfNotDefined(nullptr);
        likelihood_type_ = re_model_->GetLikelihood();
    }
}

bool ObjectiveFunction::HasGPModel() const {
    return(has_gp_model_);
}

bool ObjectiveFunction::UseGPModelForValidation() const {
    return(use_gp_model_for_validation_);
}

REModel* ObjectiveFunction::GetGPModel() const {
    return(re_model_);
}

void ObjectiveFunction::NewtonUpdateLeafValues(const int* data_leaf_index,
    const int num_leaves, double* leaf_values) const {//used only for "regression" loss
    if (has_gp_model_) {
        re_model_->NewtonUpdateLeafValues(data_leaf_index, num_leaves, leaf_values);
    }
}

}  // namespace LightGBM
