Python Package Examples
=======================

Here are examples on how to use the gpboost Python package. You should install the GPBoost [Python package](https://github.com/fabsig/GPBoost/tree/master/python-package) first. You also need scikit-learn and matplotlib (for plots) to run the examples, but they are not required for the package itself. You can install these packages with pip:

```
pip install scikit-learn matplotlib -U
```

It is recommended that you **run the examples in interactive mode using, e.g., Spyder or PyCharm**. Alternatively, you can run the examples from the command line in this folder, for example:

```
python boosting_example.py
```

Examples include:

  * [GPBoost and LaGaboost algorithms](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/GPBoost_algorithm.py) for Gaussian data ("regression") and non-Gaussian data ("classification", etc.) combining tree-boosting with Gaussian process and random effects models
  * [Parameter tuning](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/parameter_tuning.py) using deterministic or random grid search
  * [Generalized linear Gaussian process and mixed effects model examples](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/generalized_linear_Gaussian_process_mixed_effects_models.py)
  * [GPBoost algorithm applied to panel data](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/panel_data_example.py)
