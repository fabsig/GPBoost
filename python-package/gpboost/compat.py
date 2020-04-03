# coding: utf-8
"""Compatibility library."""
from __future__ import absolute_import

import inspect
import sys

import numpy as np

is_py3 = (sys.version_info[0] == 3)

"""Compatibility between Python2 and Python3"""
if is_py3:
    zip_ = zip
    string_type = str
    numeric_types = (int, float, bool)
    integer_types = (int, )
    range_ = range

    def argc_(func):
        """Count the number of arguments of a function."""
        return len(inspect.signature(func).parameters)

    def decode_string(bytestring):
        """Decode C bytestring to ordinary string."""
        return bytestring.decode('utf-8')
else:
    from itertools import izip as zip_
    string_type = basestring
    numeric_types = (int, long, float, bool)
    integer_types = (int, long)
    range_ = xrange

    def argc_(func):
        """Count the number of arguments of a function."""
        return len(inspect.getargspec(func).args)

    def decode_string(bytestring):
        """Decode C bytestring to ordinary string."""
        return bytestring

"""json"""
try:
    import simplejson as json
except (ImportError, SyntaxError):
    # simplejson does not support Python 3.2, it throws a SyntaxError
    # because of u'...' Unicode literals.
    import json


def json_default_with_numpy(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


"""pandas"""
try:
    from pandas import Series, DataFrame
    from pandas.api.types import is_sparse as is_dtype_sparse
    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False

    class Series(object):
        """Dummy class for pandas.Series."""

        pass

    class DataFrame(object):
        """Dummy class for pandas.DataFrame."""

        pass

    is_dtype_sparse = None

"""matplotlib"""
try:
    import matplotlib
    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

"""graphviz"""
try:
    import graphviz
    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False

"""datatable"""
try:
    import datatable
    if hasattr(datatable, "Frame"):
        DataTable = datatable.Frame
    else:
        DataTable = datatable.DataTable
    DATATABLE_INSTALLED = True
except ImportError:
    DATATABLE_INSTALLED = False

    class DataTable(object):
        """Dummy class for DataTable."""

        pass


"""sklearn"""
try:
    from sklearn.base import BaseEstimator
    from sklearn.base import RegressorMixin, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.utils.multiclass import check_classification_targets
    from sklearn.utils.validation import (assert_all_finite, check_X_y,
                                          check_array, check_consistent_length)
    try:
        from sklearn.model_selection import StratifiedKFold, GroupKFold
        from sklearn.exceptions import NotFittedError
    except ImportError:
        from sklearn.cross_validation import StratifiedKFold, GroupKFold
        from sklearn.utils.validation import NotFittedError
    SKLEARN_INSTALLED = True
    from sklearn import __version__ as SKLEARN_VERSION
    _GPBoostModelBase = BaseEstimator
    _GPBoostRegressorBase = RegressorMixin
    _GPBoostClassifierBase = ClassifierMixin
    _GPBoostLabelEncoder = LabelEncoder
    GPBoostNotFittedError = NotFittedError
    _GPBoostStratifiedKFold = StratifiedKFold
    _GPBoostGroupKFold = GroupKFold
    _GPBoostCheckXY = check_X_y
    _GPBoostCheckArray = check_array
    _GPBoostCheckConsistentLength = check_consistent_length
    _GPBoostAssertAllFinite = assert_all_finite
    _GPBoostCheckClassificationTargets = check_classification_targets
    _GPBoostComputeSampleWeight = compute_sample_weight
except ImportError:
    SKLEARN_INSTALLED = False
    SKLEARN_VERSION = '0.0.0'
    _GPBoostModelBase = object
    _GPBoostClassifierBase = object
    _GPBoostRegressorBase = object
    _GPBoostLabelEncoder = None
    GPBoostNotFittedError = ValueError
    _GPBoostStratifiedKFold = None
    _GPBoostGroupKFold = None
    _GPBoostCheckXY = None
    _GPBoostCheckArray = None
    _GPBoostCheckConsistentLength = None
    _GPBoostAssertAllFinite = None
    _GPBoostCheckClassificationTargets = None
    _GPBoostComputeSampleWeight = None


# DeprecationWarning is not shown by default, so let's create our own with higher level
class GPBoostDeprecationWarning(UserWarning):
    """Custom deprecation warning."""

    pass
