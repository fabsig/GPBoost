<img src="https://github.com/fabsig/GPBoost/blob/master/gpboost_sticker.jpg?raw=true"
     alt="GPBoost icon"
     align = "right"
     width="40%" />
     
GPBoost: Combining Tree-Boosting with Gaussian Process and Mixed Effects Models
===============================================================================
     
### Table of Contents
1. [Get Started](#get-started)
2. [Modeling Background](#modeling-background)
3. [News](#news)
4. [Open Issues - Contribute](#open-issues---contribute)
5. [References](#references)
6. [License](#license)

## Get started
**GPBoost is a software library for combining tree-boosting with Gaussian process and mixed effects models.** It also allows for independently doing tree-boosting as well as inference and prediction for Gaussian process and mixed effects models. The GPBoost library is predominantly written in C++, and there exist both a [**Python package**](https://github.com/fabsig/GPBoost/tree/master/python-package) and an [**R package**](https://github.com/fabsig/GPBoost/tree/master/R-package).

**For more information**, you may want to have a look at:

* The [**GPBoost R and Python demo**](https://htmlpreview.github.io/?https://github.com/fabsig/GPBoost/blob/master/examples/GPBoost_demo.html) illustrating how GPBoost can be used in R and Python
* The [**Python package**](https://github.com/fabsig/GPBoost/tree/master/python-package) and [**R package**](https://github.com/fabsig/GPBoost/tree/master/R-package) with installation instructions for the Python and R packages
* The companion article [**Sigrist (2020)**](http://arxiv.org/abs/2004.02653) or this [**blog post**](https://towardsdatascience.com/tree-boosted-mixed-effects-models-4df610b624cb) on how to combine tree-boosting with mixed effects models
* Detailed [**Python examples**](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide) and [**R examples**](https://github.com/fabsig/GPBoost/tree/master/R-package/demo)
* [**Main parameters**](https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst) presenting the most important parameters / settings for the GPBoost library
* [**Parameters**](https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst) an exhaustive list of all possible parametes and customizations for the tree-boosting part
* The [**CLI installation guide**](https://github.com/fabsig/GPBoost/blob/master/docs/Installation_guide.rst) explaining how to install the command line interface (CLI) version
* Comments on [**computational efficiency and large data**](https://github.com/fabsig/GPBoost/blob/master/docs/Computational_efficiency.md)


## Modeling Background
Both tree-boosting and Gaussian processes are techniques that achieve **state-of-the-art predictive accuracy**. Besides this, **tree-boosting** has the following advantages: 

* Automatic modeling of non-linearities, discontinuities, and complex high-order interactions
* Robust to outliers in and multicollinearity among predictor variables
* Scale-invariance to monotone transformations of the predictor variables
* Automatic handling of missing values in predictor variables

**Gaussian process** and **mixed effects** models have the following advantages:

* Probabilistic predictions which allows for uncertainty quantification
* Modeling of dependency which, among other things, can allow for more efficient learning of the fixed effects / regression function

For the GPBoost algorithm, it is assumed that the **response variable (aka label) y is the sum of a potentially non-linear mean function F(X) and random effects Zb**:
```
y = F(X) + Zb + xi
```
where where xi is an independent error term and X are predictor variables (aka covariates or features).


The **random effects** can consists of

- Gaussian processes (including random coefficient processes)
- Grouped random effects (including nested, crossed, and random coefficient effects)
- A sum of the above

The model is trained using the **GPBoost algorithm, where training means learning the covariance parameters** (aka hyperparameters) of the random effects and the **predictor function F(X)** using a tree ensemble. In brief, the GPBoost algorithm is a boosting algorithm that iteratively learns the covariance parameters and adds a tree to the ensemble of trees using a [gradient and/or a Newton boosting](https://www.sciencedirect.com/science/article/abs/pii/S0957417420308381) step. In the GPBoost library, covariance parameters can be learned using (Nesterov accelerated) gradient descent or Fisher scoring (aka natural gradient descent). Further, trees are learned using the [LightGBM](https://github.com/microsoft/LightGBM/) library. See [Sigrist (2020)](http://arxiv.org/abs/2004.02653) for more details.

## News

* See the [GitHub releases](https://github.com/fabsig/GPBoost/releases) page
* 04/06/2020 : First release of GPBoost

## Open Issues - Contribute

#### Software issues
- Add [Python tests](https://github.com/fabsig/GPBoost/tree/master/tests) (see corresponding [R tests](https://github.com/fabsig/GPBoost/tree/master/R-package/tests))
- Setting up Travis CI for GPBoost 

#### Computational issues
- Add GPU support for Gaussian processes
- Add [CHOLMOD](https://github.com/DrTimothyAldenDavis/SuiteSparse) support

#### Methodological issues
- Implement an approach such that computations scale well (memory and time) for Gaussian process models for non-Gaussian data
- Add a spatio-temporal Gaussian process model (e.g. a separable one)
- Add possibility to predict latent Gaussian processes and random effects (e.g. random coefficients)

## References

Sigrist Fabio. "[Gaussian Process Boosting](http://arxiv.org/abs/2004.02653)". Preprint (2020).

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)". Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.

## License

This project is licensed under the terms of the Apache License 2.0. See [LICENSE](https://github.com/fabsig/GPBoost/blob/master/LICENSE) for additional details.
