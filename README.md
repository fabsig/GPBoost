GPBoost: Combining Tree-Boosting with Gaussian Process and Mixed Effects Models
===============================================================================

# Table of Contents
1. [Get Started](#get-started)
2. [Modeling Background](#modeling-background)
3. [News](#news)
4. [Open Issues - Contribute](#open-issues---contribute)
5. [References](#references)
6. [License](#license)

## Get started
**GPBoost is a software library for combining tree-boosting with Gaussian process and mixed effects models.** 

The GPBoost library is written in C++ and it has a C API. There exist both a [**Python package**](https://github.com/fabsig/GPBoost/tree/master/python-package) and an [**R package**](https://github.com/fabsig/GPBoost/tree/master/R-package).

For more information, you may want to have a look at:

* The [**GPBoost R and Python demo**](https://htmlpreview.github.io/?https://github.com/fabsig/GPBoost/blob/master/examples/GPBoost_demo.html) illustrating how GPBoost can be used in R and Python
* The [**Python package**](https://github.com/fabsig/GPBoost/tree/master/python-package) and [**R package**](https://github.com/fabsig/GPBoost/tree/master/R-package) with installation instructions for the Python and R packages
* Additional [**Python examples**](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide) and [**R examples**](https://github.com/fabsig/GPBoost/tree/master/R-package/demo)
* [**Main parameters**](https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst) presenting the most important parameters / settings for using the GPBoost library
* [**Parameters**](https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst) an exhaustive list of all possible parametes and customizations for the tree-boosting part
* The [**CLI installation guide**](https://github.com/fabsig/GPBoost/blob/master/docs/Installation_guide.rst) explaining how to install the command line interface (CLI) version


## Modeling Background
Both tree-boosting and Gaussian processes are techniques that achieve **state-of-the-art predictive accuracy**. Besides this, **tree-boosting** has the following **advantages**: 

* Automatic modeling of non-linearities, discontinuities, and complex high-order interactions
* Robust to outliers in and multicollinearity among predictor variables
* Scale-invariant to monotone transformations of the predictor variables
* Automatic handling of missing values in predictor variables

**Gaussian process** models have the following **advantage**:

* Probabilistic predictions which allows for uncertainty quantification

For the GPBoost algorithm, it is assumed that the response variable (label) is the sum of a non-linear mean function and so-called random effects. The **random effects** can consists of

- Gaussian processes (including random coefficient processes)
- Grouped random effects (including nested, crossed, and random coefficient effects)
- A sum of the above

The **model is trained using the GPBoost algorithm**, where trainings means estimating the **covariance parameters** of the random effects and the **mean function F(X) using a tree ensemble**. In brief, the GPBoost algorithm is a boosting algorithm that iteratively learns the covariance parameters and adds a tree to the ensemble of trees using a gradient and/or a Newton boosting step. In the GPBoost library, covariance parameters can be learned using (accelerated) gradient descent or Fisher scoring. Further, trees are learned using the [LightGBM](https://github.com/microsoft/LightGBM/) library. See the [reference paper](http://arxiv.org/abs/2004.02653) for more details.

## News

04/06/2020 : First release of GPBoost

## Open Issues - Contribute

#### Software issues
- Add tests: [Python tests](https://github.com/fabsig/GPBoost/tree/master/tests) and [R tests](https://github.com/fabsig/GPBoost/tree/master/R-package/tests) such that the coverage is higher
- Setting up Travis CI for GPBoost 

#### Computational issues
- Add GPU support for Gaussian processes

#### Methodological issues
- Add a spatio-temporal Gaussian process model (e.g. a separable one)
- Add possibility to predict latent Gaussian processes and random effects (e.g. random coefficients)

## References

Sigrist Fabio. "[Gaussian Process Boosting](http://arxiv.org/abs/2004.02653)". Preprint (2020).

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)". Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.

## License

This project is licensed under the terms of the Apache License 2.0. See [LICENSE](https://github.com/fabsig/GPBoost/blob/master/LICENSE) for additional details.
