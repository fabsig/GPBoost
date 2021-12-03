<img src="https://github.com/fabsig/GPBoost/blob/master/gpboost_sticker.jpg?raw=true"
     alt="GPBoost icon"
     align = "right"
     width="40%" />
     
GPBoost: Combining Tree-Boosting with Gaussian Process and Mixed Effects Models
===============================================================================
     
### Table of Contents
1. [Get Started](#get-started)
2. [Modeling background](#modeling-background)
3. [News](#news)
4. [Open issues - contribute](#open-issues---contribute)
5. [References](#references)
6. [License](#license)

## Get started
**GPBoost is a software library for combining tree-boosting with Gaussian process and grouped random effects models (aka mixed effects models or latent Gaussian models).** It also allows for independently applying tree-boosting as well as Gaussian process and (generalized) linear mixed effects models (LMMs and GLMMs). The GPBoost library is predominantly written in C++, it has a C interface, and there exist both a [**Python package**](https://github.com/fabsig/GPBoost/tree/master/python-package) and an [**R package**](https://github.com/fabsig/GPBoost/tree/master/R-package).

**For more information**, you may want to have a look at:

* The [**Python package**](https://github.com/fabsig/GPBoost/tree/master/python-package) and [**R package**](https://github.com/fabsig/GPBoost/tree/master/R-package) including installation instructions
* The companion articles [**Sigrist (2020)**](http://arxiv.org/abs/2004.02653) and [**Sigrist (2021)**](https://arxiv.org/abs/2105.08966)
* These **blog posts** on how to 
   * [Combine tree-boosting with grouped random effects models](https://towardsdatascience.com/tree-boosted-mixed-effects-models-4df610b624cb) 
   * [Combine tree-boosting with Gaussian processes for spatial data](https://towardsdatascience.com/tree-boosting-for-spatial-data-789145d6d97d)
   * [Use GPBoost for generalized linear mixed effects models (GLMMs)](https://towardsdatascience.com/generalized-linear-mixed-effects-models-in-r-and-python-with-gpboost-89297622820c) 
* [This demo](https://htmlpreview.github.io/?https://github.com/fabsig/GPBoost/blob/master/examples/GPBoost_demo.html) on how GPBoost can be used in R and Python
* Detailed [**Python examples**](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide) and [**R examples**](https://github.com/fabsig/GPBoost/tree/master/R-package/demo)
* [**Main parameters**](https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst) presenting the most important parameters / settings for the GPBoost library
* [**Parameters**](https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst) an exhaustive list of all possible parametes and customizations for the tree-boosting part
* The [**CLI installation guide**](https://github.com/fabsig/GPBoost/blob/master/docs/Installation_guide.rst) explaining how to install the command line interface (CLI) version
* Comments on [**computational efficiency and large data**](https://github.com/fabsig/GPBoost/blob/master/docs/Computational_efficiency.md)


## Modeling background
The GPBoost library allows for combining tree-boosting with Gaussian process (GP) and grouped random effects models (aka mixed effects models or latent Gaussian models) in order to leverage advantages and remedy drawbacks of these two approaches (see [below](#background-on-gaussian-process-and-grouped-random-effects-models) for a list with advantages and disadvantages of these modeling techniques). In particular, the GPBoost / LaGaBoost algorithms are generalizations of classical boosting algorithms which assume (conditional) independence across samples. Advantages include that (i) this can allow for more efficient learning of predictor functions which, among other things, can translate into increased prediction accuracy, (ii) it can be used as a solution for high-cardinality categorical variables in tree-boosting, and (iii) it can be used for modeling spatial or spatio-temporal data when, e.g., spatial predictions should vary continuously , or smoothly, over space. Further, the GPBoost / LaGaBoost algorithms are non-linear extensions of classical mixed effects or latent Gaussian models, where the linear predictor function is replaced by a non-linear function which is learned using tree-boosting (=arguably, often the "best" approach for tabular data).

### GPBoost and LaGaBoost algorithms

The GPBoost library implements two algorithms for combining tree-boosting with Gaussian process and grouped random effects models: 

* The **GPBoost algorithm** [(Sigrist, 2020)](http://arxiv.org/abs/2004.02653) for data with a Gaussian likelihood (conditional distribution of data)
* The **LaGaBoost algorithm** [(Sigrist, 2021)](https://arxiv.org/abs/2105.08966) for data with non-Gaussian likelihoods

**For Gaussian likelihoods (GPBoost algorithm)**, it is assumed that the response variable (aka label) y is the sum of a potentially non-linear mean function F(X) and random effects Zb:
```
y = F(X) + Zb + xi
```
where xi is an independent error term and X are predictor variables (aka covariates or features).

**For non-Gaussian likelihoods (LaGaBoost algorithm)**, it is assumed that the response variable y follows some distribution p(y|m) and that a (potentially multivariate) parameter m of this distribution is related to a non-linear function F(X) and random effects Zb:
```
y ~ p(y|m)
m = G(F(X) + Zb)
```
where G() is a so-called link function.

In the GPBoost library, the **random effects** can consist of

- Gaussian processes (including random coefficient processes)
- Grouped random effects (including nested, crossed, and random coefficient effects)
- Combinations of the above

Learning the above-mentioned models means **learning both the covariance parameters** (aka hyperparameters) of the random effects and the **predictor function F(X)**. Both the GPBoost and the LaGaBoost algorithms iteratively learn the covariance parameters and add a tree to the ensemble of trees F(X) using a [gradient and/or a Newton boosting](https://www.sciencedirect.com/science/article/abs/pii/S0957417420308381) step. In the GPBoost library, covariance parameters can (currently) be learned using (Nesterov accelerated) gradient descent, Fisher scoring (aka natural gradient descent), and Nelder-Mead. Further, trees are learned using the [LightGBM](https://github.com/microsoft/LightGBM/) library. 

See [Sigrist (2020)](http://arxiv.org/abs/2004.02653) and [Sigrist (2021)](https://arxiv.org/abs/2105.08966) for more details.

### Background on Gaussian process and grouped random effects models

**Tree-boosting** has the following **advantages and disadvantages**: 

| Advantages of tree-boosting | Disadvantages of tree-boosting |
|:--- |:--- |
| - State-of-the-art prediction accuracy | - Assumes conditional independence of samples |
| - Automatic modeling of non-linearities, discontinuities, and complex high-order interactions | - Produces discontinuous predictions for, e.g., spatial data |
| - Robust to outliers in and multicollinearity among predictor variables | - Can have difficulty with high-cardinality categorical variables |
| - Scale-invariant to monotone transformations of predictor variables |  |
| - Automatic handling of missing values in predictor variables |  |

**Gaussian process (GPs) and grouped random effects models** (aka mixed effects models or latent Gaussian models) have the following **advantages and disadvantages**:

| Advantages of GPs / random effects models | Disadvantages of GPs / random effects models |
|:--- |:--- |
| - Probabilistic predictions which allows for uncertainty quantification | - Zero or a linear prior mean (predictor, fixed effects) function |
| - Incorporation of reasonable prior knowledge. E.g. for spatial data: "close samples are more similar to each other than distant samples" and a function should vary continuously / smoothly over space |  |
| - Modeling of dependency which, among other things, can allow for more efficient learning of the fixed effects (predictor) function |  |
| - Grouped random effects can be used for modeling high-cardinality categorical variables |  |

## News

* See the [GitHub releases](https://github.com/fabsig/GPBoost/releases) page
* 04/06/2020 : First release of GPBoost

## Open issues - contribute

#### Software issues
- Add [Python tests](https://github.com/fabsig/GPBoost/tree/master/tests) (see corresponding [R tests](https://github.com/fabsig/GPBoost/tree/master/R-package/tests))
- Setting up a CI environment 

#### Computational issues
- Add GPU support for Gaussian processes
- Add [CHOLMOD](https://github.com/DrTimothyAldenDavis/SuiteSparse) support

#### Methodological issues
- Add multivariate models, e.g., using coregionalization
- Add spatio-temporal Gaussian process models
- Add possibility to predict latent Gaussian processes and random effects (e.g., random coefficients)
- Implement more approaches such that computations scale well (memory and time) for Gaussian process models and mixed effects models with more than one grouping variable for non-Gaussian data
- Support sample weights

## References

- Sigrist Fabio. "[Gaussian Process Boosting](http://arxiv.org/abs/2004.02653)". Preprint (2020).
- Sigrist Fabio. "[Latent Gaussian Model Boosting](https://arxiv.org/abs/2105.08966)". Preprint (2021).
- Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)". Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.

## License

This project is licensed under the terms of the Apache License 2.0. See [LICENSE](https://github.com/fabsig/GPBoost/blob/master/LICENSE) for more information.
