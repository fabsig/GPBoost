# coding: utf-8
""" GPBoost
 Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
"""
from .basic import Booster, Dataset, register_logger, GPModel, get_nested_categories
from .callback import (early_stopping, print_evaluation, record_evaluation,
                       reset_parameter)
from .engine import cv, train, CVBooster, grid_search_tune_parameters

import os

try:
    from .sklearn import GPBoostModel, GPBoostRegressor, GPBoostClassifier, GPBoostRanker
except ImportError:
    pass
try:
    from .plotting import (plot_importance, plot_split_value_histogram, plot_metric,
                           plot_tree, create_tree_digraph)
except ImportError:
    pass


dir_path = os.path.dirname(os.path.realpath(__file__))

if os.path.isfile(os.path.join(dir_path, 'VERSION.txt')):
    with open(os.path.join(dir_path, 'VERSION.txt')) as version_file:
        __version__ = version_file.read().strip()

__all__ = ['Dataset', 'Booster', 'CVBooster',
           'register_logger',
           'train', 'cv', 'grid_search_tune_parameters',
           'GPBoostModel', 'GPBoostRegressor', 'GPBoostClassifier', 'GPBoostRanker',
           'print_evaluation', 'record_evaluation', 'reset_parameter', 'early_stopping',
           'plot_importance', 'plot_split_value_histogram', 'plot_metric', 'plot_tree', 'create_tree_digraph']
