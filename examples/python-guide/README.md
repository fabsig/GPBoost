Python Package Examples
=======================

Here are examples on how to use the gpboost Python package. You should install the GPBoost [Python package](https://github.com/fabsig/GPBoost/tree/master/python-package) first. You also need scikit-learn and matplotlib (for plots) to run the examples, but they are not required for the package itself. You can install these packages with pip:

```
pip install scikit-learn matplotlib -U
```

It is recommended that you **run the examples in interactive mode in e.g. Spyder or PyCharm**. Alternatively, you can run the examples from the command line in this folder, for example:

```
python boosting_example.py
```

Examples include:

  * [GPBoost algorithm](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/GPBoost_algorithm.py) for combining tree-boosting with Gaussian process and random effects models
  * [GPBoost algorithm for binary classification and other non-Gaussian data](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/classification_non_Gaussian_data.py) (Poisson regression, etc.)
  * [Cross validation](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/cross_validation.py) for parameter tuning
  * [Linear Gaussian process and mixed effects model examples](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/linear_Gaussian_process_mixed_effects_models.py)
  * [Generalized linear Gaussian process and mixed effects model examples](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/generalized_linear_Gaussian_process_mixed_effects_models.py)
  * [Standard boosting functionality (without Gaussian process or random  effects)](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/boosting.py)
