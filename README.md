<img src="https://github.com/fabsig/GPBoost/blob/master/docs/logo/gpboost_logo.png?raw=true"
     alt="GPBoost icon"
     align = "right"
     width="30%" />
     
GPBoost: Combining Tree-Boosting with Gaussian Process and Mixed Effects Models
===============================================================================
     
### Table of Contents
1. [Introduction](#introduction)
2. [Modeling background](#modeling-background)
3. [News](#news)
4. [Open issues - contribute](#open-issues---contribute)
5. [References](#references)
6. [License](#license)

## Introduction
**GPBoost is a software library for combining tree-boosting with Gaussian process and grouped random effects models (aka mixed effects models or latent Gaussian models).** It also allows for independently applying tree-boosting as well as Gaussian process and (generalized) linear mixed effects models (LMMs and GLMMs). The GPBoost library is predominantly written in C++, it has a C interface, and there exist both a [**Python package**](https://github.com/fabsig/GPBoost/tree/master/python-package) and an [**R package**](https://github.com/fabsig/GPBoost/tree/master/R-package).

For more information, you may want to have a look at:

* The [**Python package**](https://github.com/fabsig/GPBoost/tree/master/python-package) and [**R package**](https://github.com/fabsig/GPBoost/tree/master/R-package) including installation instructions
* The companion articles [**Sigrist (2022, JMLR)**](https://www.jmlr.org/papers/v23/20-322.html) and [**Sigrist (2023, TPAMI)**](https://ieeexplore.ieee.org/document/9759834) for background on the methodology
* Detailed [**Python examples**](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide) and [**R examples**](https://github.com/fabsig/GPBoost/tree/master/R-package/demo)
* [**Main parameters**](https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst): the most important parameters / settings for the GPBoost library
<!-- * [Detailed tree-boosting parameters](https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst): a comprehensive list of all tree-boosting (i.e., not random effects) related parameters -->
* The following **blog posts**:
   * [Combine tree-boosting with grouped random effects models in Python](https://towardsdatascience.com/tree-boosted-mixed-effects-models-4df610b624cb?sk=2b43bb9c6188c20ce699e42a46a2ced5)
   * [GPBoost for High-Cardinality Categorical Variables in Python & R](https://towardsdatascience.com/mixed-effects-machine-learning-for-high-cardinality-categorical-variables-part-ii-gpboost-3bdd9ef74492?sk=c0090356b176ab9b3e704e4392727123)
   * [GPBoost for grouped and areal spatial econometric data in Python & R](https://towardsdatascience.com/mixed-effects-machine-learning-with-gpboost-for-grouped-and-areal-spatial-econometric-data-b26f8bddd385?sk=87fd6bbb817f4163f22d9a4860ff2b5b)
   * [Combine tree-boosting with Gaussian processes for spatial data in Python & R](https://towardsdatascience.com/tree-boosting-for-spatial-data-789145d6d97d?sk=4f9924d378dbb517e883fc9c612c34f1)
   * [GPBoost for Longitudinal & Panel Data in Python & R](https://towardsdatascience.com/mixed-effects-machine-learning-for-longitudinal-panel-data-with-gpboost-part-iii-523bb38effc?sk=491ff65929c3fbdc508211fe4a8c05f4)
   * [Generalized Linear Mixed Effects Models (GLMMs) in R and Python with GPBoost](https://towardsdatascience.com/generalized-linear-mixed-effects-models-in-r-and-python-with-gpboost-89297622820c?sk=2a4b12edb915d3ff8c86cc01175eea97)
   * [Demo](https://htmlpreview.github.io/?https://github.com/fabsig/GPBoost/blob/master/examples/GPBoost_demo.html) on how GPBoost can be used in R and Python

* The [CLI installation guide](https://github.com/fabsig/GPBoost/blob/master/docs/Installation_guide.rst) explaining how to install the command line interface (CLI) version
* Comments on [computational efficiency and large data](https://github.com/fabsig/GPBoost/blob/master/docs/Computational_efficiency.rst)
* The documentation at [https://gpboost.readthedocs.io](https://gpboost.readthedocs.io/en/latest/)


## Modeling background
The GPBoost algorithm combines tree-boosting with latent Gaussian models such as Gaussian process (GP) and grouped random effects models. This allows to leverage advantages and remedy drawbacks of both tree-boosting and latent Gaussian models; see [below](#strength-and-weaknesses-of-tree-boosting-and-linear-mixed-effects-and-gp-models) for a list of strength and weaknesses of these two modeling approaches. The GPBoost algorithm can be seen as a generalization of both traditional (generalized) linear mixed effects and Gaussian process models and classical independent tree-boosting (which often has the highest prediction for tabular data). 


### Advantages of the GPBoost algorithm

Compared to (generalized) linear mixed effects and Gaussian process models, the GPBoost algorithm allows for 

* modeling the fixed effects function in a non-parametric and non-linear manner which can result in more realistic models which, consequently, have higher prediction accuracy

Compared to classical independent boosting, the GPBoost algorithm allows for  

* more efficient learning of predictor functions which, among other things, can translate into increased prediction accuracy
* efficient modeling of high-cardinality categorical variables
* modeling spatial or spatio-temporal data when, e.g., spatial predictions should vary continuously , or smoothly, over space

### Modeling details

**For Gaussian likelihoods (GPBoost algorithm)**, it is assumed that the response variable (aka label) $y$ is the sum of a potentially non-linear mean function $F(X)$ and random effects $Zb$:

$$y = F(X) + Zb + x_i$$

where $F(X)$ is a sum (="ensemble") of trees, $x_i$ is an independent error term, and $X$ are predictor variables (aka covariates or features). The random effects $Zb$ can currently consist of:

- Gaussian processes (including random coefficient processes)
- Grouped random effects (including nested, crossed, and random coefficient effects)
- Combinations of the above

**For non-Gaussian likelihoods (LaGaBoost algorithm)**, it is assumed that the response variable y follows a distribution $p(y|m)$ and that a (potentially multivariate) parameter $m$ of this distribution is related to a non-linear function $F(X)$ and random effects $Zb$:

$$
\begin{equation} \begin{split}
y & \sim p(y | m) \\
m & = G(F(X) + Zb) \\
\end{split} \end{equation}
$$

where $G()$ is a so-called link function. See [here](https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst) for a list of [currently supported likelihoods](https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst) $p(y|m)$.

**Estimating or training** the above-mentioned models means learning both the covariance parameters (aka hyperparameters) of the random effects and the predictor function $F(X)$. Both the GPBoost and the LaGaBoost algorithms iteratively learn the covariance parameters and add a tree to the ensemble of trees $F(X)$ using a [functional gradient and/or a Newton boosting step](https://www.sciencedirect.com/science/article/abs/pii/S0957417420308381). See [Sigrist (2022, JMLR)](https://www.jmlr.org/papers/v23/20-322.html) and [Sigrist (2023, TPAMI)](https://ieeexplore.ieee.org/document/9759834) for more details.

### Strength and weaknesses of tree-boosting and linear mixed effects and GP models

#### Classical independent tree-boosting

| Strengths | Weaknesses |
|:--- |:--- |
| - State-of-the-art prediction accuracy | - Assumes conditional independence of samples |
| - Automatic modeling of non-linearities, discontinuities, and complex high-order interactions | - Produces discontinuous predictions for, e.g., spatial data |
| - Robust to outliers in and multicollinearity among predictor variables | - Can have difficulty with high-cardinality categorical variables |
| - Scale-invariant to monotone transformations of predictor variables |  |
| - Automatic handling of missing values in predictor variables |  |

#### Linear mixed effects and Gaussian process (GPs) models (aka latent Gaussian models)

| Strengths | Weaknesses |
|:--- |:--- |
| - Probabilistic predictions which allows for uncertainty quantification | - Zero or a linear prior mean (predictor, fixed effects) function |
| - Incorporation of reasonable prior knowledge. E.g. for spatial data: "close samples are more similar to each other than distant samples" and a function should vary continuously / smoothly over space |  |
| - Modeling of dependency which, among other things, can allow for more efficient learning of the fixed effects (predictor) function |  |
| - Grouped random effects can be used for modeling high-cardinality categorical variables |  |

## News

* See the [GitHub releases](https://github.com/fabsig/GPBoost/releases) page
* October 2022: Glad to announce that the two companion articles are published in the [Journal of Machine Learning Research (JMLR)](https://www.jmlr.org/papers/v23/20-322.html) and [IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)](https://ieeexplore.ieee.org/document/9759834)
* 04/06/2020 : First release of GPBoost

## Open issues - contribute

- See the open issues on GitHub with an *enhancement* label

#### Software issues
- Add [Python tests](https://github.com/fabsig/GPBoost/tree/master/tests) (see corresponding [R tests](https://github.com/fabsig/GPBoost/tree/master/R-package/tests))
- Setting up a CI environment 
- Support conversion of GPBoost models to [ONNX model format](https://onnx.ai/)

#### Computational issues
- Add GPU support for Gaussian processes
- Add [CHOLMOD](https://github.com/DrTimothyAldenDavis/SuiteSparse) support

#### Methodological issues
- Add multivariate models, e.g., using coregionalization
- Add spatio-temporal Gaussian process models
- Add areal models for spatial data such as CAR and SAR models
- Add possibility to predict separate latent Gaussian processes and random effects (e.g., random coefficients)
- Implement more approaches such that computations scale well (memory and time) for Gaussian process models and mixed effects models with more than one grouping variable for non-Gaussian data
- Support sample weights
- Support other distances besides the Euclidean distance (e.g., great circle distance) for Gaussian processes

## References

- Sigrist Fabio. "[Gaussian Process Boosting](https://www.jmlr.org/papers/v23/20-322.html)". *Journal of Machine Learning Research* (2022).
- Sigrist Fabio. "[Latent Gaussian Model Boosting](https://ieeexplore.ieee.org/document/9759834)". *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2023).
- Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)". *Advances in Neural Information Processing Systems* 30 (2017).
- Williams, Christopher KI, and Carl Edward Rasmussen. *Gaussian processes for machine learning*. MIT press, 2006.
- Pinheiro, Jose, and Douglas Bates. *Mixed-effects models in S and S-PLUS*. Springer science & business media, 2006.

## License

This project is licensed under the terms of the Apache License 2.0. See [LICENSE](https://github.com/fabsig/GPBoost/blob/master/LICENSE) for more information.
