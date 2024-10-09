Python Package
==============

Below is the official documentation of the Python package. See also the `Python package GitHub page <https://github.com/fabsig/GPBoost/tree/master/python-package>`__ for more information on the Python package (e.g., installation and examples).

.. currentmodule:: gpboost

GPModel, booster, and data structure
------------------------------------

.. autosummary::
    :toctree: pythonapi/

    GPModel
    Booster
    Dataset


GPBoost Algorithm Training and Choosing Tuning Parameters
---------------------------------------------------------

.. autosummary::
    :toctree: pythonapi/

    train
    tune_pars_TPE_algorithm_optuna
    grid_search_tune_parameters
    cv

Scikit-learn API
----------------

.. autosummary::
    :toctree: pythonapi/

    GPBoostModel
    GPBoostClassifier
    GPBoostRegressor
    GPBoostRanker

Various
-------

Callbacks

.. autosummary::
    :toctree: pythonapi/

    early_stopping
    print_evaluation
    record_evaluation
    reset_parameter

Plotting

.. autosummary::
    :toctree: pythonapi/

    plot_importance
    plot_split_value_histogram
    plot_metric
    plot_tree
    create_tree_digraph

Utilities

.. autosummary::
    :toctree: pythonapi/

    register_logger
    get_nested_categories
    CVBooster
