Python API
==========

.. currentmodule:: gpboost

Data Structure API
------------------

.. autosummary::
    :toctree: pythonapi/

    Dataset
    Booster
    GPModel
    CVBooster

Training API
------------

.. autosummary::
    :toctree: pythonapi/

    train
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

Callbacks
---------

.. autosummary::
    :toctree: pythonapi/

    early_stopping
    print_evaluation
    record_evaluation
    reset_parameter

Plotting
--------

.. autosummary::
    :toctree: pythonapi/

    plot_importance
    plot_split_value_histogram
    plot_metric
    plot_tree
    create_tree_digraph

Utilities
---------

.. autosummary::
    :toctree: pythonapi/

    register_logger
