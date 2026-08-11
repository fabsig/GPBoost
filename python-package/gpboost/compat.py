# coding: utf-8
"""Compatibility library."""

"""pandas"""
try:
    from pandas import concat
    from pandas import Series as pd_Series
    from pandas import DataFrame as pd_DataFrame
    try:
        from pandas import SparseDtype as _pd_SparseDtype

        def is_dtype_sparse(dtype):
            """Check whether a pandas dtype is a sparse dtype."""
            return isinstance(dtype, _pd_SparseDtype)
    except ImportError:  # pandas < 1.0
        from pandas.api.types import is_sparse as is_dtype_sparse
    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False

    class pd_Series:
        """Dummy class for pandas.Series."""

        pass

    class pd_DataFrame:
        """Dummy class for pandas.DataFrame."""

        pass

    concat = None
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
        dt_DataTable = datatable.Frame
    else:
        dt_DataTable = datatable.DataTable
    DATATABLE_INSTALLED = True
except ImportError:
    DATATABLE_INSTALLED = False

    class dt_DataTable:
        """Dummy class for datatable.DataTable."""

        pass


"""sklearn"""
try:
    from sklearn.base import BaseEstimator
    from sklearn.base import RegressorMixin, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.utils.multiclass import check_classification_targets
    from sklearn.utils.validation import assert_all_finite, check_X_y, check_array
    try:
        from sklearn.model_selection import StratifiedKFold, GroupKFold
        from sklearn.exceptions import NotFittedError
    except ImportError:
        from sklearn.cross_validation import StratifiedKFold, GroupKFold
        from sklearn.utils.validation import NotFittedError
    try:
        from sklearn.utils.validation import _check_sample_weight
    except ImportError:
        from sklearn.utils.validation import check_consistent_length

        # dummy function to support older version of scikit-learn
        def _check_sample_weight(sample_weight, X, dtype=None):
            check_consistent_length(sample_weight, X)
            return sample_weight

    # scikit-learn renamed 'force_all_finite' to 'ensure_all_finite' in 1.6
    # and removed the old name in 1.8
    from inspect import signature as _signature
    if 'ensure_all_finite' in _signature(check_array).parameters:
        def _rename_all_finite_kwarg(kwargs):
            if 'force_all_finite' in kwargs:
                kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
            return kwargs
    else:
        def _rename_all_finite_kwarg(kwargs):
            if 'ensure_all_finite' in kwargs:
                kwargs['force_all_finite'] = kwargs.pop('ensure_all_finite')
            return kwargs

    def _GPBoostCheckXY(X, y, **kwargs):
        """Call sklearn's check_X_y independently of the scikit-learn version."""
        return check_X_y(X, y, **_rename_all_finite_kwarg(kwargs))

    def _GPBoostCheckArray(array, **kwargs):
        """Call sklearn's check_array independently of the scikit-learn version."""
        return check_array(array, **_rename_all_finite_kwarg(kwargs))

    SKLEARN_INSTALLED = True
    _GPBoostModelBase = BaseEstimator
    _GPBoostRegressorBase = RegressorMixin
    _GPBoostClassifierBase = ClassifierMixin
    _GPBoostLabelEncoder = LabelEncoder
    GPBoostNotFittedError = NotFittedError
    _GPBoostStratifiedKFold = StratifiedKFold
    _GPBoostGroupKFold = GroupKFold
    _GPBoostCheckSampleWeight = _check_sample_weight
    _GPBoostAssertAllFinite = assert_all_finite
    _GPBoostCheckClassificationTargets = check_classification_targets
    _GPBoostComputeSampleWeight = compute_sample_weight
except ImportError:
    SKLEARN_INSTALLED = False
    _GPBoostModelBase = object
    _GPBoostClassifierBase = object
    _GPBoostRegressorBase = object
    _GPBoostLabelEncoder = None
    GPBoostNotFittedError = ValueError
    _GPBoostStratifiedKFold = None
    _GPBoostGroupKFold = None
    _GPBoostCheckXY = None
    _GPBoostCheckArray = None
    _GPBoostCheckSampleWeight = None
    _GPBoostAssertAllFinite = None
    _GPBoostCheckClassificationTargets = None
    _GPBoostComputeSampleWeight = None