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

**GPBoost is a machine learning software library that combines tree-boosting with Gaussian process and mixed effects models.**

The GPBoost library is mainly written in C++ and it has a C API. There exist both a [**Python package**](https://github.com/fabsig/GPBoost/tree/master/python-package) and an [**R package**](https://github.com/fabsig/GPBoost/tree/master/R-package). The [**GPBoost R and Python demo**](https://htmlpreview.github.io/?https://github.com/fabsig/GPBoost/blob/master/examples/GPBoost_demo.html) explains how the GPBoost library can be used from R and Python.

For more information, you may want to have a look at:

* The [**GPBoost R and Python demo**](https://htmlpreview.github.io/?https://github.com/fabsig/GPBoost/blob/master/examples/GPBoost_demo.html) illustrates how GPBoost can be used in R and Python
* Additional [**Python examples**](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide) and [**R examples**](https://github.com/fabsig/GPBoost/tree/master/R-package/demo)
* [**Main parameters**](https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst) presents the most important parameters / settings for using the GPBoost library
* [**Parameters**](https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst) is an exhaustive list of all possible parametes and customizations for the tree-boosting part
* See [**Python package**](https://github.com/fabsig/GPBoost/tree/master/python-package) and [**R package**](https://github.com/fabsig/GPBoost/tree/master/R-package) for installation instructions for the Python and R packages
* The [**CLI installation guide**](https://github.com/fabsig/GPBoost/blob/master/docs/Installation_guide.rst) explains how to install the command line interface (CLI) version


## Modeling Background
It is assumed that the data is the sum of a non-linear mean function and so-called random effects. The **random effects** can consists of

- Gaussian processes (including random coefficient processes)
- Grouped random effects (including nested, crossed, and random coefficient effects)
- A sum of the above

The **model is trained using the GPBoost algorithm**, where trainings means estimating the variance and **covariance parameters** of the random effects and the **mean function F(X) using a tree ensemble**. In brief, the GPBoost algorithm iteratively estimates the covariance parameters and adds a tree to the ensemble using boosting steps. Trees are learned using the [LightGBM](https://github.com/microsoft/LightGBM/) library. See the [reference paper](#references) for more details.

## News

04/01/2020 : First release of GPBoost

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

Sigrist Fabio. "[Gaussian Process Boosting](https://arxiv.org/abs/XXX)". Preprint (2020).

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)". Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.

## License

This project is licensed under the terms of the Apache License 2.0. See [LICENSE](https://github.com/fabsig/GPBoost/blob/master/LICENSE) for additional details.
