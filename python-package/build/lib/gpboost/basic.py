# coding: utf-8
# pylint: disable = invalid-name, C0111, C0301
# pylint: disable = R0912, R0913, R0914, W0105, W0201, W0212
"""Wrapper for C API of GPBoost."""
from __future__ import absolute_import

import copy
import ctypes
import os
import warnings
from tempfile import NamedTemporaryFile

import numpy as np
import scipy.sparse

from .compat import (PANDAS_INSTALLED, DataFrame, Series, is_dtype_sparse,
                     DataTable,
                     decode_string, string_type,
                     integer_types, numeric_types,
                     json, json_default_with_numpy,
                     range_, zip_)
from .libpath import find_lib_path


def _load_lib():
    """Load GPBoost library."""
    lib_path = find_lib_path()
    if len(lib_path) == 0:
        return None
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    lib.LGBM_GetLastError.restype = ctypes.c_char_p
    return lib


_LIB = _load_lib()


def _safe_call(ret):
    """Check the return value from C API call.

    Parameters
    ----------
    ret : int
        The return value from C API calls.
    """
    if ret != 0:
        raise GPBoostError(decode_string(_LIB.LGBM_GetLastError()))


def is_numeric(obj):
    """Check whether object is a number or not, include numpy number, etc."""
    try:
        float(obj)
        return True
    except (TypeError, ValueError):
        # TypeError: obj is not a string or a number
        # ValueError: invalid literal
        return False


def is_numpy_1d_array(data):
    """Check whether data is a numpy 1-D array."""
    return isinstance(data, np.ndarray) and len(data.shape) == 1


def is_1d_list(data):
    """Check whether data is a 1-D list."""
    return isinstance(data, list) and (not data or is_numeric(data[0]))


def list_to_1d_numpy(data, dtype=np.float32, name='list'):
    """Convert data to numpy 1-D array."""
    if is_numpy_1d_array(data):
        if data.dtype == dtype:
            return data
        else:
            return data.astype(dtype=dtype, copy=False)
    elif is_1d_list(data):
        return np.array(data, dtype=dtype, copy=False)
    elif isinstance(data, Series):
        if _get_bad_pandas_dtypes([data.dtypes]):
            raise ValueError('Series.dtypes must be int, float or bool')
        return np.array(data, dtype=dtype, copy=False)  # SparseArray should be supported as well
    else:
        raise TypeError("Wrong type({0}) for {1}.\n"
                        "It should be list, numpy 1-D array or pandas Series".format(type(data).__name__, name))


def cfloat32_array_to_numpy(cptr, length):
    """Convert a ctypes float pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        return np.fromiter(cptr, dtype=np.float32, count=length)
    else:
        raise RuntimeError('Expected float pointer')


def cfloat64_array_to_numpy(cptr, length):
    """Convert a ctypes double pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_double)):
        return np.fromiter(cptr, dtype=np.float64, count=length)
    else:
        raise RuntimeError('Expected double pointer')


def cint32_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int32)):
        return np.fromiter(cptr, dtype=np.int32, count=length)
    else:
        raise RuntimeError('Expected int pointer')


def cint8_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int8)):
        return np.fromiter(cptr, dtype=np.int8, count=length)
    else:
        raise RuntimeError('Expected int pointer')


def c_str(string):
    """Convert a Python string to C string."""
    return ctypes.c_char_p(string.encode('utf-8'))


def string_array_c_str(string_array):
    """Convert a list/array of Python strings to a contiguous (in memory) sequence of C strings."""
    return ctypes.c_char_p("\0".join(string_array).encode('utf-8'))


def c_array(ctype, values):
    """Convert a Python array to C array."""
    return (ctype * len(values))(*values)


def param_dict_to_str(data):
    """Convert Python dictionary to string, which is passed to C API."""
    if data is None or not data:
        return ""
    pairs = []
    for key, val in data.items():
        if isinstance(val, (list, tuple, set)) or is_numpy_1d_array(val):
            pairs.append(str(key) + '=' + ','.join(map(str, val)))
        elif isinstance(val, string_type) or isinstance(val, numeric_types) or is_numeric(val):
            pairs.append(str(key) + '=' + str(val))
        elif val is not None:
            raise TypeError('Unknown type of parameter:%s, got:%s'
                            % (key, type(val).__name__))
    return ' '.join(pairs)


class _TempFile(object):
    def __enter__(self):
        with NamedTemporaryFile(prefix="gpboost_tmp_", delete=True) as f:
            self.name = f.name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.isfile(self.name):
            os.remove(self.name)

    def readlines(self):
        with open(self.name, "r+") as f:
            ret = f.readlines()
        return ret

    def writelines(self, lines):
        with open(self.name, "w+") as f:
            f.writelines(lines)


class GPBoostError(Exception):
    """Error thrown by GPBoost."""

    pass


class _ConfigAliases(object):
    aliases = {"boosting": {"boosting",
                            "boosting_type",
                            "boost"},
               "categorical_feature": {"categorical_feature",
                                       "cat_feature",
                                       "categorical_column",
                                       "cat_column"},
               "early_stopping_round": {"early_stopping_round",
                                        "early_stopping_rounds",
                                        "early_stopping",
                                        "n_iter_no_change"},
               "eval_at": {"eval_at",
                           "ndcg_eval_at",
                           "ndcg_at",
                           "map_eval_at",
                           "map_at"},
               "header": {"header",
                          "has_header"},
               "machines": {"machines",
                            "workers",
                            "nodes"},
               "metric": {"metric",
                          "metrics",
                          "metric_types"},
               "num_class": {"num_class",
                             "num_classes"},
               "num_iterations": {"num_iterations",
                                  "num_iteration",
                                  "n_iter",
                                  "num_tree",
                                  "num_trees",
                                  "num_round",
                                  "num_rounds",
                                  "num_boost_round",
                                  "n_estimators"},
               "objective": {"objective",
                             "objective_type",
                             "app",
                             "application"},
               "verbosity": {"verbosity",
                             "verbose"}}

    @classmethod
    def get(cls, *args):
        ret = set()
        for i in args:
            ret |= cls.aliases.get(i, set())
        return ret


MAX_INT32 = (1 << 31) - 1

"""Macro definition of data type in C API of GPBoost"""
C_API_DTYPE_FLOAT32 = 0
C_API_DTYPE_FLOAT64 = 1
C_API_DTYPE_INT32 = 2
C_API_DTYPE_INT64 = 3
C_API_DTYPE_INT8 = 4

"""Matrix is row major in Python"""
C_API_IS_ROW_MAJOR = 1

"""Macro definition of prediction type in C API of GPBoost"""
C_API_PREDICT_NORMAL = 0
C_API_PREDICT_RAW_SCORE = 1
C_API_PREDICT_LEAF_INDEX = 2
C_API_PREDICT_CONTRIB = 3

"""Data type of data field"""
FIELD_TYPE_MAPPER = {"label": C_API_DTYPE_FLOAT32,
                     "weight": C_API_DTYPE_FLOAT32,
                     "init_score": C_API_DTYPE_FLOAT64,
                     "group": C_API_DTYPE_INT32,
                     "feature_penalty": C_API_DTYPE_FLOAT64,
                     "monotone_constraints": C_API_DTYPE_INT8}


def convert_from_sliced_object(data):
    """Fix the memory of multi-dimensional sliced object."""
    if isinstance(data, np.ndarray) and isinstance(data.base, np.ndarray):
        if not data.flags.c_contiguous:
            warnings.warn("Usage of np.ndarray subset (sliced data) is not recommended "
                          "due to it will double the peak memory cost in GPBoost.")
            return np.copy(data)
    return data


def c_float_array(data):
    """Get pointer of float numpy array / list."""
    if is_1d_list(data):
        data = np.array(data, copy=False)
    if is_numpy_1d_array(data):
        data = convert_from_sliced_object(data)
        assert data.flags.c_contiguous
        if data.dtype == np.float32:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            type_data = C_API_DTYPE_FLOAT32
        elif data.dtype == np.float64:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            type_data = C_API_DTYPE_FLOAT64
        else:
            raise TypeError("Expected np.float32 or np.float64, met type({})"
                            .format(data.dtype))
    else:
        raise TypeError("Unknown type({})".format(type(data).__name__))
    return (ptr_data, type_data, data)  # return `data` to avoid the temporary copy is freed


def c_int_array(data):
    """Get pointer of int numpy array / list."""
    if is_1d_list(data):
        data = np.array(data, copy=False)
    if is_numpy_1d_array(data):
        data = convert_from_sliced_object(data)
        assert data.flags.c_contiguous
        if data.dtype == np.int32:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            type_data = C_API_DTYPE_INT32
        elif data.dtype == np.int64:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            type_data = C_API_DTYPE_INT64
        else:
            raise TypeError("Expected np.int32 or np.int64, met type({})"
                            .format(data.dtype))
    else:
        raise TypeError("Unknown type({})".format(type(data).__name__))
    return (ptr_data, type_data, data)  # return `data` to avoid the temporary copy is freed


def _get_bad_pandas_dtypes(dtypes):
    pandas_dtype_mapper = {'int8': 'int', 'int16': 'int', 'int32': 'int',
                           'int64': 'int', 'uint8': 'int', 'uint16': 'int',
                           'uint32': 'int', 'uint64': 'int', 'bool': 'int',
                           'float16': 'float', 'float32': 'float', 'float64': 'float'}
    bad_indices = [i for i, dtype in enumerate(dtypes) if (dtype.name not in pandas_dtype_mapper
                                                           and (not is_dtype_sparse(dtype)
                                                                or dtype.subtype.name not in pandas_dtype_mapper))]
    return bad_indices


def _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical):
    if isinstance(data, DataFrame):
        if len(data.shape) != 2 or data.shape[0] < 1:
            raise ValueError('Input data must be 2 dimensional and non empty.')
        if feature_name == 'auto' or feature_name is None:
            data = data.rename(columns=str)
        cat_cols = list(data.select_dtypes(include=['category']).columns)
        cat_cols_not_ordered = [col for col in cat_cols if not data[col].cat.ordered]
        if pandas_categorical is None:  # train dataset
            pandas_categorical = [list(data[col].cat.categories) for col in cat_cols]
        else:
            if len(cat_cols) != len(pandas_categorical):
                raise ValueError('train and valid dataset categorical_feature do not match.')
            for col, category in zip_(cat_cols, pandas_categorical):
                if list(data[col].cat.categories) != list(category):
                    data[col] = data[col].cat.set_categories(category)
        if len(cat_cols):  # cat_cols is list
            data = data.copy()  # not alter origin DataFrame
            data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes).replace({-1: np.nan})
        if categorical_feature is not None:
            if feature_name is None:
                feature_name = list(data.columns)
            if categorical_feature == 'auto':  # use cat cols from DataFrame
                categorical_feature = cat_cols_not_ordered
            else:  # use cat cols specified by user
                categorical_feature = list(categorical_feature)
        if feature_name == 'auto':
            feature_name = list(data.columns)
        bad_indices = _get_bad_pandas_dtypes(data.dtypes)
        if bad_indices:
            raise ValueError("DataFrame.dtypes for data must be int, float or bool.\n"
                             "Did not expect the data types in the following fields: "
                             + ', '.join(data.columns[bad_indices]))
        data = data.values
        if data.dtype != np.float32 and data.dtype != np.float64:
            data = data.astype(np.float32)
    else:
        if feature_name == 'auto':
            feature_name = None
        if categorical_feature == 'auto':
            categorical_feature = None
    return data, feature_name, categorical_feature, pandas_categorical


def _label_from_pandas(label):
    if isinstance(label, DataFrame):
        if len(label.columns) > 1:
            raise ValueError('DataFrame for label cannot have multiple columns')
        if _get_bad_pandas_dtypes(label.dtypes):
            raise ValueError('DataFrame.dtypes for label must be int, float or bool')
        label = np.ravel(label.values.astype(np.float32, copy=False))
    return label


def _dump_pandas_categorical(pandas_categorical, file_name=None):
    pandas_str = ('\npandas_categorical:'
                  + json.dumps(pandas_categorical, default=json_default_with_numpy)
                  + '\n')
    if file_name is not None:
        with open(file_name, 'a') as f:
            f.write(pandas_str)
    return pandas_str


def _load_pandas_categorical(file_name=None, model_str=None):
    pandas_key = 'pandas_categorical:'
    offset = -len(pandas_key)
    if file_name is not None:
        max_offset = -os.path.getsize(file_name)
        with open(file_name, 'rb') as f:
            while True:
                if offset < max_offset:
                    offset = max_offset
                f.seek(offset, os.SEEK_END)
                lines = f.readlines()
                if len(lines) >= 2:
                    break
                offset *= 2
        last_line = decode_string(lines[-1]).strip()
        if not last_line.startswith(pandas_key):
            last_line = decode_string(lines[-2]).strip()
    elif model_str is not None:
        idx = model_str.rfind('\n', 0, offset)
        last_line = model_str[idx:].strip()
    if last_line.startswith(pandas_key):
        return json.loads(last_line[len(pandas_key):])
    else:
        return None


class _InnerPredictor(object):
    """_InnerPredictor.

    Not exposed to user.
    Used only for prediction, usually used for continued training.

    .. note::

        Can be converted from Booster, but cannot be converted to Booster.
    """

    def __init__(self, model_file=None, booster_handle=None, pred_parameter=None):
        """Initialize the _InnerPredictor.

        Parameters
        ----------
        model_file : string or None, optional (default=None)
            Path to the model file.
        booster_handle : object or None, optional (default=None)
            Handle of Booster.
        pred_parameter: dict or None, optional (default=None)
            Other parameters for the prediciton.
        """
        self.handle = ctypes.c_void_p()
        self.__is_manage_handle = True
        if model_file is not None:
            """Prediction task"""
            out_num_iterations = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterCreateFromModelfile(
                c_str(model_file),
                ctypes.byref(out_num_iterations),
                ctypes.byref(self.handle)))
            out_num_class = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.num_class = out_num_class.value
            self.num_total_iteration = out_num_iterations.value
            self.pandas_categorical = _load_pandas_categorical(file_name=model_file)
        elif booster_handle is not None:
            self.__is_manage_handle = False
            self.handle = booster_handle
            out_num_class = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.num_class = out_num_class.value
            out_num_iterations = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetCurrentIteration(
                self.handle,
                ctypes.byref(out_num_iterations)))
            self.num_total_iteration = out_num_iterations.value
            self.pandas_categorical = None
        else:
            raise TypeError('Need model_file or booster_handle to create a predictor')

        pred_parameter = {} if pred_parameter is None else pred_parameter
        self.pred_parameter = param_dict_to_str(pred_parameter)

    def __del__(self):
        try:
            if self.__is_manage_handle:
                _safe_call(_LIB.LGBM_BoosterFree(self.handle))
        except AttributeError:
            pass

    def __getstate__(self):
        this = self.__dict__.copy()
        this.pop('handle', None)
        return this

    def predict(self, data, num_iteration=-1,
                raw_score=False, pred_leaf=False, pred_contrib=False, data_has_header=False,
                is_reshape=True):
        """Predict logic.

        Parameters
        ----------
        data : string, numpy array, pandas DataFrame, H2O DataTable's Frame or scipy.sparse
            Data source for prediction.
            When data type is string, it represents the path of txt file.
        num_iteration : int, optional (default=-1)
            Iteration used for prediction.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.
        data_has_header : bool, optional (default=False)
            Whether data has header.
            Used only for txt data.
        is_reshape : bool, optional (default=True)
            Whether to reshape to (nrow, ncol).

        Returns
        -------
        result : numpy array
            Prediction result.
        """
        if isinstance(data, Dataset):
            raise TypeError("Cannot use Dataset instance for prediction, please use raw data instead")
        data = _data_from_pandas(data, None, None, self.pandas_categorical)[0]
        predict_type = C_API_PREDICT_NORMAL
        if raw_score:
            predict_type = C_API_PREDICT_RAW_SCORE
        if pred_leaf:
            predict_type = C_API_PREDICT_LEAF_INDEX
        if pred_contrib:
            predict_type = C_API_PREDICT_CONTRIB
        int_data_has_header = 1 if data_has_header else 0
        if num_iteration > self.num_total_iteration:
            num_iteration = self.num_total_iteration

        if isinstance(data, string_type):
            with _TempFile() as f:
                _safe_call(_LIB.LGBM_BoosterPredictForFile(
                    self.handle,
                    c_str(data),
                    ctypes.c_int(int_data_has_header),
                    ctypes.c_int(predict_type),
                    ctypes.c_int(num_iteration),
                    c_str(self.pred_parameter),
                    c_str(f.name)))
                lines = f.readlines()
                nrow = len(lines)
                preds = [float(token) for line in lines for token in line.split('\t')]
                preds = np.array(preds, dtype=np.float64, copy=False)
        elif isinstance(data, scipy.sparse.csr_matrix):
            preds, nrow = self.__pred_for_csr(data, num_iteration, predict_type)
        elif isinstance(data, scipy.sparse.csc_matrix):
            preds, nrow = self.__pred_for_csc(data, num_iteration, predict_type)
        elif isinstance(data, np.ndarray):
            preds, nrow = self.__pred_for_np2d(data, num_iteration, predict_type)
        elif isinstance(data, list):
            try:
                data = np.array(data)
            except BaseException:
                raise ValueError('Cannot convert data list to numpy array.')
            preds, nrow = self.__pred_for_np2d(data, num_iteration, predict_type)
        elif isinstance(data, DataTable):
            preds, nrow = self.__pred_for_np2d(data.to_numpy(), num_iteration, predict_type)
        else:
            try:
                warnings.warn('Converting data to scipy sparse matrix.')
                csr = scipy.sparse.csr_matrix(data)
            except BaseException:
                raise TypeError('Cannot predict data for type {}'.format(type(data).__name__))
            preds, nrow = self.__pred_for_csr(csr, num_iteration, predict_type)
        if pred_leaf:
            preds = preds.astype(np.int32)
        if is_reshape and preds.size != nrow:
            if preds.size % nrow == 0:
                preds = preds.reshape(nrow, -1)
            else:
                raise ValueError('Length of predict result (%d) cannot be divide nrow (%d)'
                                 % (preds.size, nrow))
        return preds

    def __get_num_preds(self, num_iteration, nrow, predict_type):
        """Get size of prediction result."""
        if nrow > MAX_INT32:
            raise GPBoostError('GPBoost cannot perform prediction for data'
                               'with number of rows greater than MAX_INT32 (%d).\n'
                               'You can split your data into chunks'
                               'and then concatenate predictions for them' % MAX_INT32)
        n_preds = ctypes.c_int64(0)
        _safe_call(_LIB.LGBM_BoosterCalcNumPredict(
            self.handle,
            ctypes.c_int(nrow),
            ctypes.c_int(predict_type),
            ctypes.c_int(num_iteration),
            ctypes.byref(n_preds)))
        return n_preds.value

    def __pred_for_np2d(self, mat, num_iteration, predict_type):
        """Predict for a 2-D numpy matrix."""
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray or list must be 2 dimensional')

        def inner_predict(mat, num_iteration, predict_type, preds=None):
            if mat.dtype == np.float32 or mat.dtype == np.float64:
                data = np.array(mat.reshape(mat.size), dtype=mat.dtype, copy=False)
            else:  # change non-float data to float data, need to copy
                data = np.array(mat.reshape(mat.size), dtype=np.float32)
            ptr_data, type_ptr_data, _ = c_float_array(data)
            n_preds = self.__get_num_preds(num_iteration, mat.shape[0], predict_type)
            if preds is None:
                preds = np.zeros(n_preds, dtype=np.float64)
            elif len(preds.shape) != 1 or len(preds) != n_preds:
                raise ValueError("Wrong length of pre-allocated predict array")
            out_num_preds = ctypes.c_int64(0)
            _safe_call(_LIB.LGBM_BoosterPredictForMat(
                self.handle,
                ptr_data,
                ctypes.c_int(type_ptr_data),
                ctypes.c_int(mat.shape[0]),
                ctypes.c_int(mat.shape[1]),
                ctypes.c_int(C_API_IS_ROW_MAJOR),
                ctypes.c_int(predict_type),
                ctypes.c_int(num_iteration),
                c_str(self.pred_parameter),
                ctypes.byref(out_num_preds),
                preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
            if n_preds != out_num_preds.value:
                raise ValueError("Wrong length for predict results")
            return preds, mat.shape[0]

        nrow = mat.shape[0]
        if nrow > MAX_INT32:
            sections = np.arange(start=MAX_INT32, stop=nrow, step=MAX_INT32)
            # __get_num_preds() cannot work with nrow > MAX_INT32, so calculate overall number of predictions piecemeal
            n_preds = [self.__get_num_preds(num_iteration, i, predict_type) for i in
                       np.diff([0] + list(sections) + [nrow])]
            n_preds_sections = np.array([0] + n_preds, dtype=np.intp).cumsum()
            preds = np.zeros(sum(n_preds), dtype=np.float64)
            for chunk, (start_idx_pred, end_idx_pred) in zip_(np.array_split(mat, sections),
                                                              zip_(n_preds_sections, n_preds_sections[1:])):
                # avoid memory consumption by arrays concatenation operations
                inner_predict(chunk, num_iteration, predict_type, preds[start_idx_pred:end_idx_pred])
            return preds, nrow
        else:
            return inner_predict(mat, num_iteration, predict_type)

    def __pred_for_csr(self, csr, num_iteration, predict_type):
        """Predict for a CSR data."""

        def inner_predict(csr, num_iteration, predict_type, preds=None):
            nrow = len(csr.indptr) - 1
            n_preds = self.__get_num_preds(num_iteration, nrow, predict_type)
            if preds is None:
                preds = np.zeros(n_preds, dtype=np.float64)
            elif len(preds.shape) != 1 or len(preds) != n_preds:
                raise ValueError("Wrong length of pre-allocated predict array")
            out_num_preds = ctypes.c_int64(0)

            ptr_indptr, type_ptr_indptr, __ = c_int_array(csr.indptr)
            ptr_data, type_ptr_data, _ = c_float_array(csr.data)

            assert csr.shape[1] <= MAX_INT32
            csr.indices = csr.indices.astype(np.int32, copy=False)

            _safe_call(_LIB.LGBM_BoosterPredictForCSR(
                self.handle,
                ptr_indptr,
                ctypes.c_int32(type_ptr_indptr),
                csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ptr_data,
                ctypes.c_int(type_ptr_data),
                ctypes.c_int64(len(csr.indptr)),
                ctypes.c_int64(len(csr.data)),
                ctypes.c_int64(csr.shape[1]),
                ctypes.c_int(predict_type),
                ctypes.c_int(num_iteration),
                c_str(self.pred_parameter),
                ctypes.byref(out_num_preds),
                preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
            if n_preds != out_num_preds.value:
                raise ValueError("Wrong length for predict results")
            return preds, nrow

        nrow = len(csr.indptr) - 1
        if nrow > MAX_INT32:
            sections = [0] + list(np.arange(start=MAX_INT32, stop=nrow, step=MAX_INT32)) + [nrow]
            # __get_num_preds() cannot work with nrow > MAX_INT32, so calculate overall number of predictions piecemeal
            n_preds = [self.__get_num_preds(num_iteration, i, predict_type) for i in np.diff(sections)]
            n_preds_sections = np.array([0] + n_preds, dtype=np.intp).cumsum()
            preds = np.zeros(sum(n_preds), dtype=np.float64)
            for (start_idx, end_idx), (start_idx_pred, end_idx_pred) in zip_(zip_(sections, sections[1:]),
                                                                             zip_(n_preds_sections,
                                                                                  n_preds_sections[1:])):
                # avoid memory consumption by arrays concatenation operations
                inner_predict(csr[start_idx:end_idx], num_iteration, predict_type, preds[start_idx_pred:end_idx_pred])
            return preds, nrow
        else:
            return inner_predict(csr, num_iteration, predict_type)

    def __pred_for_csc(self, csc, num_iteration, predict_type):
        """Predict for a CSC data."""
        nrow = csc.shape[0]
        if nrow > MAX_INT32:
            return self.__pred_for_csr(csc.tocsr(), num_iteration, predict_type)
        n_preds = self.__get_num_preds(num_iteration, nrow, predict_type)
        preds = np.zeros(n_preds, dtype=np.float64)
        out_num_preds = ctypes.c_int64(0)

        ptr_indptr, type_ptr_indptr, __ = c_int_array(csc.indptr)
        ptr_data, type_ptr_data, _ = c_float_array(csc.data)

        assert csc.shape[0] <= MAX_INT32
        csc.indices = csc.indices.astype(np.int32, copy=False)

        _safe_call(_LIB.LGBM_BoosterPredictForCSC(
            self.handle,
            ptr_indptr,
            ctypes.c_int32(type_ptr_indptr),
            csc.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ptr_data,
            ctypes.c_int(type_ptr_data),
            ctypes.c_int64(len(csc.indptr)),
            ctypes.c_int64(len(csc.data)),
            ctypes.c_int64(csc.shape[0]),
            ctypes.c_int(predict_type),
            ctypes.c_int(num_iteration),
            c_str(self.pred_parameter),
            ctypes.byref(out_num_preds),
            preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if n_preds != out_num_preds.value:
            raise ValueError("Wrong length for predict results")
        return preds, nrow


class Dataset(object):
    """Dataset in GPBoost."""

    def __init__(self, data, label=None, reference=None,
                 weight=None, group=None, init_score=None, silent=False,
                 feature_name='auto', categorical_feature='auto', params=None,
                 free_raw_data=False):
        """Initialize Dataset.

        Parameters
        ----------
        data : string, numpy array, pandas DataFrame, H2O DataTable's Frame, scipy.sparse or list of numpy arrays
            Data source of Dataset.
            If string, it represents the path to txt file.
        label : list, numpy 1-D array, pandas Series / one-column DataFrame or None, optional (default=None)
            Label of the data.
        reference : Dataset or None, optional (default=None)
            If this is Dataset for validation, training data should be used as reference.
        weight : list, numpy 1-D array, pandas Series or None, optional (default=None)
            Weight for each instance.
        group : list, numpy 1-D array, pandas Series or None, optional (default=None)
            Group/query size for Dataset.
        init_score : list, numpy 1-D array, pandas Series or None, optional (default=None)
            Init score for Dataset.
        silent : bool, optional (default=False)
            Whether to print messages during construction.
        feature_name : list of strings or 'auto', optional (default="auto")
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of strings or int, or 'auto', optional (default="auto")
            Categorical features.
            If list of int, interpreted as indices.
            If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
        params : dict or None, optional (default=None)
            Other parameters for Dataset.
        free_raw_data : bool, optional (default=False)
            If True, raw data is freed after constructing inner Dataset.
        """
        self.handle = None
        self.data = data
        self.label = label
        self.reference = reference
        self.weight = weight
        self.group = group
        self.init_score = init_score
        self.silent = silent
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.params = copy.deepcopy(params)
        self.free_raw_data = free_raw_data
        self.used_indices = None
        self.need_slice = True
        self._predictor = None
        self.pandas_categorical = None
        self.params_back_up = None
        self.feature_penalty = None
        self.monotone_constraints = None

    def __del__(self):
        try:
            self._free_handle()
        except AttributeError:
            pass

    def _free_handle(self):
        if self.handle is not None:
            _safe_call(_LIB.LGBM_DatasetFree(self.handle))
            self.handle = None
        self.need_slice = True
        if self.used_indices is not None:
            self.data = None
        return self

    def _set_init_score_by_predictor(self, predictor, data, used_indices=None):
        data_has_header = False
        if isinstance(data, string_type):
            # check data has header or not
            data_has_header = any(self.params.get(alias, False) for alias in _ConfigAliases.get("header"))
        init_score = predictor.predict(data,
                                       raw_score=True,
                                       data_has_header=data_has_header,
                                       is_reshape=False)
        num_data = self.num_data()
        if used_indices is not None:
            assert not self.need_slice
            if isinstance(data, string_type):
                sub_init_score = np.zeros(num_data * predictor.num_class, dtype=np.float32)
                assert num_data == len(used_indices)
                for i in range_(len(used_indices)):
                    for j in range_(predictor.num_class):
                        sub_init_score[i * predictor.num_class + j] = init_score[
                            used_indices[i] * predictor.num_class + j]
                init_score = sub_init_score
        if predictor.num_class > 1:
            # need to regroup init_score
            new_init_score = np.zeros(init_score.size, dtype=np.float32)
            for i in range_(num_data):
                for j in range_(predictor.num_class):
                    new_init_score[j * num_data + i] = init_score[i * predictor.num_class + j]
            init_score = new_init_score
        self.set_init_score(init_score)

    def _lazy_init(self, data, label=None, reference=None,
                   weight=None, group=None, init_score=None, predictor=None,
                   silent=False, feature_name='auto',
                   categorical_feature='auto', params=None):
        if data is None:
            self.handle = None
            return self
        if reference is not None:
            self.pandas_categorical = reference.pandas_categorical
            categorical_feature = reference.categorical_feature
        data, feature_name, categorical_feature, self.pandas_categorical = _data_from_pandas(data,
                                                                                             feature_name,
                                                                                             categorical_feature,
                                                                                             self.pandas_categorical)
        label = _label_from_pandas(label)

        # process for args
        params = {} if params is None else params
        args_names = (getattr(self.__class__, '_lazy_init')
                          .__code__
                          .co_varnames[:getattr(self.__class__, '_lazy_init').__code__.co_argcount])
        for key, _ in params.items():
            if key in args_names:
                warnings.warn('{0} keyword has been found in `params` and will be ignored.\n'
                              'Please use {0} argument of the Dataset constructor to pass this parameter.'
                              .format(key))
        # user can set verbose with params, it has higher priority
        if not any(verbose_alias in params for verbose_alias in _ConfigAliases.get("verbosity")) and silent:
            params["verbose"] = -1
        # get categorical features
        if categorical_feature is not None:
            categorical_indices = set()
            feature_dict = {}
            if feature_name is not None:
                feature_dict = {name: i for i, name in enumerate(feature_name)}
            for name in categorical_feature:
                if isinstance(name, string_type) and name in feature_dict:
                    categorical_indices.add(feature_dict[name])
                elif isinstance(name, integer_types):
                    categorical_indices.add(name)
                else:
                    raise TypeError("Wrong type({}) or unknown name({}) in categorical_feature"
                                    .format(type(name).__name__, name))
            if categorical_indices:
                for cat_alias in _ConfigAliases.get("categorical_feature"):
                    if cat_alias in params:
                        warnings.warn('{} in param dict is overridden.'.format(cat_alias))
                        params.pop(cat_alias, None)
                params['categorical_column'] = sorted(categorical_indices)

        params_str = param_dict_to_str(params)
        # process for reference dataset
        ref_dataset = None
        if isinstance(reference, Dataset):
            ref_dataset = reference.construct().handle
        elif reference is not None:
            raise TypeError('Reference dataset should be None or dataset instance')
        # start construct data
        if isinstance(data, string_type):
            self.handle = ctypes.c_void_p()
            _safe_call(_LIB.LGBM_DatasetCreateFromFile(
                c_str(data),
                c_str(params_str),
                ref_dataset,
                ctypes.byref(self.handle)))
        elif isinstance(data, scipy.sparse.csr_matrix):
            self.__init_from_csr(data, params_str, ref_dataset)
        elif isinstance(data, scipy.sparse.csc_matrix):
            self.__init_from_csc(data, params_str, ref_dataset)
        elif isinstance(data, np.ndarray):
            self.__init_from_np2d(data, params_str, ref_dataset)
        elif isinstance(data, list) and len(data) > 0 and all(isinstance(x, np.ndarray) for x in data):
            self.__init_from_list_np2d(data, params_str, ref_dataset)
        elif isinstance(data, DataTable):
            self.__init_from_np2d(data.to_numpy(), params_str, ref_dataset)
        else:
            try:
                csr = scipy.sparse.csr_matrix(data)
                self.__init_from_csr(csr, params_str, ref_dataset)
            except BaseException:
                raise TypeError('Cannot initialize Dataset from {}'.format(type(data).__name__))
        if label is not None:
            self.set_label(label)
        if self.get_label() is None:
            raise ValueError("Label should not be None")
        if weight is not None:
            self.set_weight(weight)
        if group is not None:
            self.set_group(group)
        if isinstance(predictor, _InnerPredictor):
            if self._predictor is None and init_score is not None:
                warnings.warn("The init_score will be overridden by the prediction of init_model.")
            self._set_init_score_by_predictor(predictor, data)
        elif init_score is not None:
            self.set_init_score(init_score)
        elif predictor is not None:
            raise TypeError('Wrong predictor type {}'.format(type(predictor).__name__))
        # set feature names
        return self.set_feature_name(feature_name)

    def __init_from_np2d(self, mat, params_str, ref_dataset):
        """Initialize data from a 2-D numpy matrix."""
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray must be 2 dimensional')

        self.handle = ctypes.c_void_p()
        if mat.dtype == np.float32 or mat.dtype == np.float64:
            data = np.array(mat.reshape(mat.size), dtype=mat.dtype, copy=False)
        else:  # change non-float data to float data, need to copy
            data = np.array(mat.reshape(mat.size), dtype=np.float32)

        ptr_data, type_ptr_data, _ = c_float_array(data)
        _safe_call(_LIB.LGBM_DatasetCreateFromMat(
            ptr_data,
            ctypes.c_int(type_ptr_data),
            ctypes.c_int(mat.shape[0]),
            ctypes.c_int(mat.shape[1]),
            ctypes.c_int(C_API_IS_ROW_MAJOR),
            c_str(params_str),
            ref_dataset,
            ctypes.byref(self.handle)))
        return self

    def __init_from_list_np2d(self, mats, params_str, ref_dataset):
        """Initialize data from a list of 2-D numpy matrices."""
        ncol = mats[0].shape[1]
        nrow = np.zeros((len(mats),), np.int32)
        if mats[0].dtype == np.float64:
            ptr_data = (ctypes.POINTER(ctypes.c_double) * len(mats))()
        else:
            ptr_data = (ctypes.POINTER(ctypes.c_float) * len(mats))()

        holders = []
        type_ptr_data = None

        for i, mat in enumerate(mats):
            if len(mat.shape) != 2:
                raise ValueError('Input numpy.ndarray must be 2 dimensional')

            if mat.shape[1] != ncol:
                raise ValueError('Input arrays must have same number of columns')

            nrow[i] = mat.shape[0]

            if mat.dtype == np.float32 or mat.dtype == np.float64:
                mats[i] = np.array(mat.reshape(mat.size), dtype=mat.dtype, copy=False)
            else:  # change non-float data to float data, need to copy
                mats[i] = np.array(mat.reshape(mat.size), dtype=np.float32)

            chunk_ptr_data, chunk_type_ptr_data, holder = c_float_array(mats[i])
            if type_ptr_data is not None and chunk_type_ptr_data != type_ptr_data:
                raise ValueError('Input chunks must have same type')
            ptr_data[i] = chunk_ptr_data
            type_ptr_data = chunk_type_ptr_data
            holders.append(holder)

        self.handle = ctypes.c_void_p()
        _safe_call(_LIB.LGBM_DatasetCreateFromMats(
            ctypes.c_int(len(mats)),
            ctypes.cast(ptr_data, ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
            ctypes.c_int(type_ptr_data),
            nrow.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(ncol),
            ctypes.c_int(C_API_IS_ROW_MAJOR),
            c_str(params_str),
            ref_dataset,
            ctypes.byref(self.handle)))
        return self

    def __init_from_csr(self, csr, params_str, ref_dataset):
        """Initialize data from a CSR matrix."""
        if len(csr.indices) != len(csr.data):
            raise ValueError('Length mismatch: {} vs {}'.format(len(csr.indices), len(csr.data)))
        self.handle = ctypes.c_void_p()

        ptr_indptr, type_ptr_indptr, __ = c_int_array(csr.indptr)
        ptr_data, type_ptr_data, _ = c_float_array(csr.data)

        assert csr.shape[1] <= MAX_INT32
        csr.indices = csr.indices.astype(np.int32, copy=False)

        _safe_call(_LIB.LGBM_DatasetCreateFromCSR(
            ptr_indptr,
            ctypes.c_int(type_ptr_indptr),
            csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ptr_data,
            ctypes.c_int(type_ptr_data),
            ctypes.c_int64(len(csr.indptr)),
            ctypes.c_int64(len(csr.data)),
            ctypes.c_int64(csr.shape[1]),
            c_str(params_str),
            ref_dataset,
            ctypes.byref(self.handle)))
        return self

    def __init_from_csc(self, csc, params_str, ref_dataset):
        """Initialize data from a CSC matrix."""
        if len(csc.indices) != len(csc.data):
            raise ValueError('Length mismatch: {} vs {}'.format(len(csc.indices), len(csc.data)))
        self.handle = ctypes.c_void_p()

        ptr_indptr, type_ptr_indptr, __ = c_int_array(csc.indptr)
        ptr_data, type_ptr_data, _ = c_float_array(csc.data)

        assert csc.shape[0] <= MAX_INT32
        csc.indices = csc.indices.astype(np.int32, copy=False)

        _safe_call(_LIB.LGBM_DatasetCreateFromCSC(
            ptr_indptr,
            ctypes.c_int(type_ptr_indptr),
            csc.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ptr_data,
            ctypes.c_int(type_ptr_data),
            ctypes.c_int64(len(csc.indptr)),
            ctypes.c_int64(len(csc.data)),
            ctypes.c_int64(csc.shape[0]),
            c_str(params_str),
            ref_dataset,
            ctypes.byref(self.handle)))
        return self

    def construct(self):
        """Lazy init.

        Returns
        -------
        self : Dataset
            Constructed Dataset object.
        """
        if self.handle is None:
            if self.reference is not None:
                if self.used_indices is None:
                    # create valid
                    self._lazy_init(self.data, label=self.label, reference=self.reference,
                                    weight=self.weight, group=self.group,
                                    init_score=self.init_score, predictor=self._predictor,
                                    silent=self.silent, feature_name=self.feature_name, params=self.params)
                else:
                    # construct subset
                    used_indices = list_to_1d_numpy(self.used_indices, np.int32, name='used_indices')
                    assert used_indices.flags.c_contiguous
                    if self.reference.group is not None:
                        group_info = np.array(self.reference.group).astype(np.int32, copy=False)
                        _, self.group = np.unique(
                            np.repeat(range_(len(group_info)), repeats=group_info)[self.used_indices],
                            return_counts=True)
                    self.handle = ctypes.c_void_p()
                    params_str = param_dict_to_str(self.params)
                    _safe_call(_LIB.LGBM_DatasetGetSubset(
                        self.reference.construct().handle,
                        used_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        ctypes.c_int(used_indices.shape[0]),
                        c_str(params_str),
                        ctypes.byref(self.handle)))
                    if not self.free_raw_data:
                        self.get_data()
                    if self.group is not None:
                        self.set_group(self.group)
                    if self.get_label() is None:
                        raise ValueError("Label should not be None.")
                    if isinstance(self._predictor,
                                  _InnerPredictor) and self._predictor is not self.reference._predictor:
                        self.get_data()
                        self._set_init_score_by_predictor(self._predictor, self.data, used_indices)
            else:
                # create train
                self._lazy_init(self.data, label=self.label,
                                weight=self.weight, group=self.group,
                                init_score=self.init_score, predictor=self._predictor,
                                silent=self.silent, feature_name=self.feature_name,
                                categorical_feature=self.categorical_feature, params=self.params)
            if self.free_raw_data:
                self.data = None
        return self

    def create_valid(self, data, label=None, weight=None, group=None,
                     init_score=None, silent=False, params=None):
        """Create validation data align with current Dataset.

        Parameters
        ----------
        data : string, numpy array, pandas DataFrame, H2O DataTable's Frame, scipy.sparse or list of numpy arrays
            Data source of Dataset.
            If string, it represents the path to txt file.
        label : list, numpy 1-D array, pandas Series / one-column DataFrame or None, optional (default=None)
            Label of the data.
        weight : list, numpy 1-D array, pandas Series or None, optional (default=None)
            Weight for each instance.
        group : list, numpy 1-D array, pandas Series or None, optional (default=None)
            Group/query size for Dataset.
        init_score : list, numpy 1-D array, pandas Series or None, optional (default=None)
            Init score for Dataset.
        silent : bool, optional (default=False)
            Whether to print messages during construction.
        params : dict or None, optional (default=None)
            Other parameters for validation Dataset.

        Returns
        -------
        valid : Dataset
            Validation Dataset with reference to self.
        """
        ret = Dataset(data, label=label, reference=self,
                      weight=weight, group=group, init_score=init_score,
                      silent=silent, params=params, free_raw_data=self.free_raw_data)
        ret._predictor = self._predictor
        ret.pandas_categorical = self.pandas_categorical
        return ret

    def subset(self, used_indices, params=None):
        """Get subset of current Dataset.

        Parameters
        ----------
        used_indices : list of int
            Indices used to create the subset.
        params : dict or None, optional (default=None)
            These parameters will be passed to Dataset constructor.

        Returns
        -------
        subset : Dataset
            Subset of the current Dataset.
        """
        if params is None:
            params = self.params
        ret = Dataset(None, reference=self, feature_name=self.feature_name,
                      categorical_feature=self.categorical_feature, params=params,
                      free_raw_data=self.free_raw_data)
        ret._predictor = self._predictor
        ret.pandas_categorical = self.pandas_categorical
        ret.used_indices = sorted(used_indices)
        return ret

    def save_binary(self, filename):
        """Save Dataset to a binary file.

        Parameters
        ----------
        filename : string
            Name of the output file.

        Returns
        -------
        self : Dataset
            Returns self.
        """
        _safe_call(_LIB.LGBM_DatasetSaveBinary(
            self.construct().handle,
            c_str(filename)))
        return self

    def _update_params(self, params):
        if self.handle is not None and params is not None:
            _safe_call(_LIB.LGBM_DatasetUpdateParam(self.handle, c_str(param_dict_to_str(params))))
        if not self.params:
            self.params = copy.deepcopy(params)
        else:
            self.params_back_up = copy.deepcopy(self.params)
            self.params.update(params)
        return self

    def _reverse_update_params(self):
        self.params = copy.deepcopy(self.params_back_up)
        self.params_back_up = None
        if self.handle is not None and self.params is not None:
            _safe_call(_LIB.LGBM_DatasetUpdateParam(self.handle, c_str(param_dict_to_str(self.params))))
        return self

    def set_field(self, field_name, data):
        """Set property into the Dataset.

        Parameters
        ----------
        field_name : string
            The field name of the information.
        data : list, numpy 1-D array, pandas Series or None
            The array of data to be set.

        Returns
        -------
        self : Dataset
            Dataset with set property.
        """
        if self.handle is None:
            raise Exception("Cannot set %s before construct dataset" % field_name)
        if data is None:
            # set to None
            _safe_call(_LIB.LGBM_DatasetSetField(
                self.handle,
                c_str(field_name),
                None,
                ctypes.c_int(0),
                ctypes.c_int(FIELD_TYPE_MAPPER[field_name])))
            return self
        dtype = np.float32
        if field_name == 'group':
            dtype = np.int32
        elif field_name == 'init_score':
            dtype = np.float64
        data = list_to_1d_numpy(data, dtype, name=field_name)
        if data.dtype == np.float32 or data.dtype == np.float64:
            ptr_data, type_data, _ = c_float_array(data)
        elif data.dtype == np.int32:
            ptr_data, type_data, _ = c_int_array(data)
        else:
            raise TypeError("Expected np.float32/64 or np.int32, met type({})".format(data.dtype))
        if type_data != FIELD_TYPE_MAPPER[field_name]:
            raise TypeError("Input type error for set_field")
        _safe_call(_LIB.LGBM_DatasetSetField(
            self.handle,
            c_str(field_name),
            ptr_data,
            ctypes.c_int(len(data)),
            ctypes.c_int(type_data)))
        return self

    def get_field(self, field_name):
        """Get property from the Dataset.

        Parameters
        ----------
        field_name : string
            The field name of the information.

        Returns
        -------
        info : numpy array
            A numpy array with information from the Dataset.
        """
        if self.handle is None:
            raise Exception("Cannot get %s before construct Dataset" % field_name)
        tmp_out_len = ctypes.c_int()
        out_type = ctypes.c_int()
        ret = ctypes.POINTER(ctypes.c_void_p)()
        _safe_call(_LIB.LGBM_DatasetGetField(
            self.handle,
            c_str(field_name),
            ctypes.byref(tmp_out_len),
            ctypes.byref(ret),
            ctypes.byref(out_type)))
        if out_type.value != FIELD_TYPE_MAPPER[field_name]:
            raise TypeError("Return type error for get_field")
        if tmp_out_len.value == 0:
            return None
        if out_type.value == C_API_DTYPE_INT32:
            return cint32_array_to_numpy(ctypes.cast(ret, ctypes.POINTER(ctypes.c_int32)), tmp_out_len.value)
        elif out_type.value == C_API_DTYPE_FLOAT32:
            return cfloat32_array_to_numpy(ctypes.cast(ret, ctypes.POINTER(ctypes.c_float)), tmp_out_len.value)
        elif out_type.value == C_API_DTYPE_FLOAT64:
            return cfloat64_array_to_numpy(ctypes.cast(ret, ctypes.POINTER(ctypes.c_double)), tmp_out_len.value)
        elif out_type.value == C_API_DTYPE_INT8:
            return cint8_array_to_numpy(ctypes.cast(ret, ctypes.POINTER(ctypes.c_int8)), tmp_out_len.value)
        else:
            raise TypeError("Unknown type")

    def set_categorical_feature(self, categorical_feature):
        """Set categorical features.

        Parameters
        ----------
        categorical_feature : list of int or strings
            Names or indices of categorical features.

        Returns
        -------
        self : Dataset
            Dataset with set categorical features.
        """
        if self.categorical_feature == categorical_feature:
            return self
        if self.data is not None:
            if self.categorical_feature is None:
                self.categorical_feature = categorical_feature
                return self._free_handle()
            elif categorical_feature == 'auto':
                warnings.warn('Using categorical_feature in Dataset.')
                return self
            else:
                warnings.warn('categorical_feature in Dataset is overridden.\n'
                              'New categorical_feature is {}'.format(sorted(list(categorical_feature))))
                self.categorical_feature = categorical_feature
                return self._free_handle()
        else:
            raise GPBoostError("Cannot set categorical feature after freed raw data, "
                               "set free_raw_data=False when construct Dataset to avoid this.")

    def _set_predictor(self, predictor):
        """Set predictor for continued training.

        It is not recommended for user to call this function.
        Please use init_model argument in engine.train() or engine.cv() instead.
        """
        if predictor is self._predictor:
            return self
        if self.data is not None or (self.used_indices is not None
                                     and self.reference is not None
                                     and self.reference.data is not None):
            self._predictor = predictor
            return self._free_handle()
        else:
            raise GPBoostError("Cannot set predictor after freed raw data, "
                               "set free_raw_data=False when construct Dataset to avoid this.")

    def set_reference(self, reference):
        """Set reference Dataset.

        Parameters
        ----------
        reference : Dataset
            Reference that is used as a template to construct the current Dataset.

        Returns
        -------
        self : Dataset
            Dataset with set reference.
        """
        self.set_categorical_feature(reference.categorical_feature) \
            .set_feature_name(reference.feature_name) \
            ._set_predictor(reference._predictor)
        # we're done if self and reference share a common upstrem reference
        if self.get_ref_chain().intersection(reference.get_ref_chain()):
            return self
        if self.data is not None:
            self.reference = reference
            return self._free_handle()
        else:
            raise GPBoostError("Cannot set reference after freed raw data, "
                               "set free_raw_data=False when construct Dataset to avoid this.")

    def set_feature_name(self, feature_name):
        """Set feature name.

        Parameters
        ----------
        feature_name : list of strings
            Feature names.

        Returns
        -------
        self : Dataset
            Dataset with set feature name.
        """
        if feature_name != 'auto':
            self.feature_name = feature_name
        if self.handle is not None and feature_name is not None and feature_name != 'auto':
            if len(feature_name) != self.num_feature():
                raise ValueError("Length of feature_name({}) and num_feature({}) don't match"
                                 .format(len(feature_name), self.num_feature()))
            c_feature_name = [c_str(name) for name in feature_name]
            _safe_call(_LIB.LGBM_DatasetSetFeatureNames(
                self.handle,
                c_array(ctypes.c_char_p, c_feature_name),
                ctypes.c_int(len(feature_name))))
        return self

    def set_label(self, label):
        """Set label of Dataset.

        Parameters
        ----------
        label : list, numpy 1-D array, pandas Series / one-column DataFrame or None
            The label information to be set into Dataset.

        Returns
        -------
        self : Dataset
            Dataset with set label.
        """
        self.label = label
        if self.handle is not None:
            label = list_to_1d_numpy(_label_from_pandas(label), name='label')
            self.set_field('label', label)
            self.label = self.get_field('label')  # original values can be modified at cpp side
        return self

    def set_weight(self, weight):
        """Set weight of each instance.

        Parameters
        ----------
        weight : list, numpy 1-D array, pandas Series or None
            Weight to be set for each data point.

        Returns
        -------
        self : Dataset
            Dataset with set weight.
        """
        if weight is not None and np.all(weight == 1):
            weight = None
        self.weight = weight
        if self.handle is not None and weight is not None:
            weight = list_to_1d_numpy(weight, name='weight')
            self.set_field('weight', weight)
            self.weight = self.get_field('weight')  # original values can be modified at cpp side
        return self

    def set_init_score(self, init_score):
        """Set init score of Booster to start from.

        Parameters
        ----------
        init_score : list, numpy 1-D array, pandas Series or None
            Init score for Booster.

        Returns
        -------
        self : Dataset
            Dataset with set init score.
        """
        self.init_score = init_score
        if self.handle is not None and init_score is not None:
            init_score = list_to_1d_numpy(init_score, np.float64, name='init_score')
            self.set_field('init_score', init_score)
            self.init_score = self.get_field('init_score')  # original values can be modified at cpp side
        return self

    def set_group(self, group):
        """Set group size of Dataset (used for ranking).

        Parameters
        ----------
        group : list, numpy 1-D array, pandas Series or None
            Group size of each group.

        Returns
        -------
        self : Dataset
            Dataset with set group.
        """
        self.group = group
        if self.handle is not None and group is not None:
            group = list_to_1d_numpy(group, np.int32, name='group')
            self.set_field('group', group)
        return self

    def get_label(self):
        """Get the label of the Dataset.

        Returns
        -------
        label : numpy array or None
            The label information from the Dataset.
        """
        if self.label is None:
            self.label = self.get_field('label')
        return self.label

    def get_weight(self):
        """Get the weight of the Dataset.

        Returns
        -------
        weight : numpy array or None
            Weight for each data point from the Dataset.
        """
        if self.weight is None:
            self.weight = self.get_field('weight')
        return self.weight

    def get_feature_penalty(self):
        """Get the feature penalty of the Dataset.

        Returns
        -------
        feature_penalty : numpy array or None
            Feature penalty for each feature in the Dataset.
        """
        if self.feature_penalty is None:
            self.feature_penalty = self.get_field('feature_penalty')
        return self.feature_penalty

    def get_monotone_constraints(self):
        """Get the monotone constraints of the Dataset.

        Returns
        -------
        monotone_constraints : numpy array or None
            Monotone constraints: -1, 0 or 1, for each feature in the Dataset.
        """
        if self.monotone_constraints is None:
            self.monotone_constraints = self.get_field('monotone_constraints')
        return self.monotone_constraints

    def get_init_score(self):
        """Get the initial score of the Dataset.

        Returns
        -------
        init_score : numpy array or None
            Init score of Booster.
        """
        if self.init_score is None:
            self.init_score = self.get_field('init_score')
        return self.init_score

    def get_data(self):
        """Get the raw data of the Dataset.

        Returns
        -------
        data : string, numpy array, pandas DataFrame, H2O DataTable's Frame, scipy.sparse, list of numpy arrays or None
            Raw data used in the Dataset construction.
        """
        if self.handle is None:
            raise Exception("Cannot get data before construct Dataset")
        if self.need_slice and self.used_indices is not None and self.reference is not None:
            self.data = self.reference.data
            if self.data is not None:
                if isinstance(self.data, np.ndarray) or scipy.sparse.issparse(self.data):
                    self.data = self.data[self.used_indices, :]
                elif isinstance(self.data, DataFrame):
                    self.data = self.data.iloc[self.used_indices].copy()
                elif isinstance(self.data, DataTable):
                    self.data = self.data[self.used_indices, :]
                else:
                    warnings.warn("Cannot subset {} type of raw data.\n"
                                  "Returning original raw data".format(type(self.data).__name__))
            self.need_slice = False
        if self.data is None:
            raise GPBoostError("Cannot call `get_data` after freed raw data, "
                               "set free_raw_data=False when construct Dataset to avoid this.")
        return self.data

    def get_group(self):
        """Get the group of the Dataset.

        Returns
        -------
        group : numpy array or None
            Group size of each group.
        """
        if self.group is None:
            self.group = self.get_field('group')
            if self.group is not None:
                # group data from GPBoost is boundaries data, need to convert to group size
                self.group = np.diff(self.group)
        return self.group

    def num_data(self):
        """Get the number of rows in the Dataset.

        Returns
        -------
        number_of_rows : int
            The number of rows in the Dataset.
        """
        if self.handle is not None:
            ret = ctypes.c_int()
            _safe_call(_LIB.LGBM_DatasetGetNumData(self.handle,
                                                   ctypes.byref(ret)))
            return ret.value
        else:
            raise GPBoostError("Cannot get num_data before construct dataset")

    def num_feature(self):
        """Get the number of columns (features) in the Dataset.

        Returns
        -------
        number_of_columns : int
            The number of columns (features) in the Dataset.
        """
        if self.handle is not None:
            ret = ctypes.c_int()
            _safe_call(_LIB.LGBM_DatasetGetNumFeature(self.handle,
                                                      ctypes.byref(ret)))
            return ret.value
        else:
            raise GPBoostError("Cannot get num_feature before construct dataset")

    def get_ref_chain(self, ref_limit=100):
        """Get a chain of Dataset objects.

        Starts with r, then goes to r.reference (if exists),
        then to r.reference.reference, etc.
        until we hit ``ref_limit`` or a reference loop.

        Parameters
        ----------
        ref_limit : int, optional (default=100)
            The limit number of references.

        Returns
        -------
        ref_chain : set of Dataset
            Chain of references of the Datasets.
        """
        head = self
        ref_chain = set()
        while len(ref_chain) < ref_limit:
            if isinstance(head, Dataset):
                ref_chain.add(head)
                if (head.reference is not None) and (head.reference not in ref_chain):
                    head = head.reference
                else:
                    break
            else:
                break
        return ref_chain

    def add_features_from(self, other):
        """Add features from other Dataset to the current Dataset.

        Both Datasets must be constructed before calling this method.

        Parameters
        ----------
        other : Dataset
            The Dataset to take features from.

        Returns
        -------
        self : Dataset
            Dataset with the new features added.
        """
        if self.handle is None or other.handle is None:
            raise ValueError('Both source and target Datasets must be constructed before adding features')
        _safe_call(_LIB.LGBM_DatasetAddFeaturesFrom(self.handle, other.handle))
        return self

    def _dump_text(self, filename):
        """Save Dataset to a text file.

        This format cannot be loaded back in by GPBoost, but is useful for debugging purposes.

        Parameters
        ----------
        filename : string
            Name of the output file.

        Returns
        -------
        self : Dataset
            Returns self.
        """
        _safe_call(_LIB.LGBM_DatasetDumpText(
            self.construct().handle,
            c_str(filename)))
        return self


class Booster(object):
    """Booster."""

    def __init__(self, params=None, train_set=None, model_file=None, model_str=None, silent=False, gp_model=None):
        """Initialize the Booster.

        Parameters
        ----------
        params : dict or None, optional (default=None)
            Parameters for Booster.
        train_set : Dataset or None, optional (default=None)
            Training dataset.
        model_file : string or None, optional (default=None)
            Path to the model file.
        model_str : string or None, optional (default=None)
            Model will be loaded from this string.
        silent : bool, optional (default=False)
            Whether to print messages during construction.
        gp_model : GPModel or None, optional (default=None)
            GPModel object for Gaussian process boosting. Can currently only be used for objective = "regression"
        """
        self.handle = None
        self.network = False
        self.__need_reload_eval_info = True
        self._train_data_name = "training"
        self.__attr = {}
        self.__set_objective_to_none = False
        self.best_iteration = -1
        self.best_score = {}
        self.has_gp_model = False
        self.gp_model = None
        params = {} if params is None else copy.deepcopy(params)
        if gp_model is not None:
            if not isinstance(gp_model, GPModel):
                raise TypeError('gp_model should be GPModel instance, met {}'
                                .format(type(gp_model).__name__))
            params['has_gp_model'] = True
        # user can set verbose with params, it has higher priority
        if not any(verbose_alias in params for verbose_alias in _ConfigAliases.get("verbosity")) and silent:
            params["verbose"] = -1
        if train_set is not None:
            # Training task
            if not isinstance(train_set, Dataset):
                raise TypeError('Training data should be Dataset instance, met {}'
                                .format(type(train_set).__name__))
            params_str = param_dict_to_str(params)
            # set network if necessary
            for alias in _ConfigAliases.get("machines"):
                if alias in params:
                    machines = params[alias]
                    if isinstance(machines, string_type):
                        num_machines = len(machines.split(','))
                    elif isinstance(machines, (list, set)):
                        num_machines = len(machines)
                        machines = ','.join(machines)
                    else:
                        raise ValueError("Invalid machines in params.")
                    self.set_network(machines,
                                     local_listen_port=params.get("local_listen_port", 12400),
                                     listen_time_out=params.get("listen_time_out", 120),
                                     num_machines=params.get("num_machines", num_machines))
                    break
            # construct booster object
            self.handle = ctypes.c_void_p()
            if gp_model is None:
                _safe_call(_LIB.LGBM_BoosterCreate(
                    train_set.construct().handle,
                    c_str(params_str),
                    ctypes.byref(self.handle)))
            else:
                train_set.construct()
                if gp_model.num_data != train_set.num_data():
                    raise ValueError("Number of data points in gp_model and train_set are not equal")
                self.has_gp_model = True
                self.gp_model = gp_model
                _safe_call(_LIB.LGBM_GPBoosterCreate(
                    train_set.construct().handle,
                    c_str(params_str),
                    gp_model.handle,
                    ctypes.byref(self.handle)))
            # save reference to data
            self.train_set = train_set
            self.valid_sets = []
            self.name_valid_sets = []
            self.__num_dataset = 1
            self.__init_predictor = train_set._predictor
            if self.__init_predictor is not None:
                _safe_call(_LIB.LGBM_BoosterMerge(
                    self.handle,
                    self.__init_predictor.handle))
            out_num_class = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.__num_class = out_num_class.value
            # buffer for inner predict
            self.__inner_predict_buffer = [None]
            self.__is_predicted_cur_iter = [False]
            self.__get_eval_info()
            self.pandas_categorical = train_set.pandas_categorical
        elif model_file is not None:
            # Prediction task
            out_num_iterations = ctypes.c_int(0)
            self.handle = ctypes.c_void_p()
            _safe_call(_LIB.LGBM_BoosterCreateFromModelfile(
                c_str(model_file),
                ctypes.byref(out_num_iterations),
                ctypes.byref(self.handle)))
            out_num_class = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.__num_class = out_num_class.value
            self.pandas_categorical = _load_pandas_categorical(file_name=model_file)
        elif model_str is not None:
            self.model_from_string(model_str, not silent)
        else:
            raise TypeError('Need at least one training dataset or model file or model string '
                            'to create Booster instance')
        self.params = params

    def __del__(self):
        try:
            if self.network:
                self.free_network()
        except AttributeError:
            pass
        try:
            if self.handle is not None:
                _safe_call(_LIB.LGBM_BoosterFree(self.handle))
        except AttributeError:
            pass

    def __copy__(self):
        return self.__deepcopy__(None)

    def __deepcopy__(self, _):
        model_str = self.model_to_string(num_iteration=-1)
        booster = Booster(model_str=model_str)
        return booster

    def __getstate__(self):
        this = self.__dict__.copy()
        handle = this['handle']
        this.pop('train_set', None)
        this.pop('valid_sets', None)
        if handle is not None:
            this["handle"] = self.model_to_string(num_iteration=-1)
        return this

    def __setstate__(self, state):
        model_str = state.get('handle', None)
        if model_str is not None:
            handle = ctypes.c_void_p()
            out_num_iterations = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterLoadModelFromString(
                c_str(model_str),
                ctypes.byref(out_num_iterations),
                ctypes.byref(handle)))
            state['handle'] = handle
        self.__dict__.update(state)

    def free_dataset(self):
        """Free Booster's Datasets.

        Returns
        -------
        self : Booster
            Booster without Datasets.
        """
        self.__dict__.pop('train_set', None)
        self.__dict__.pop('valid_sets', None)
        self.__num_dataset = 0
        return self

    def _free_buffer(self):
        self.__inner_predict_buffer = []
        self.__is_predicted_cur_iter = []
        return self

    def set_network(self, machines, local_listen_port=12400,
                    listen_time_out=120, num_machines=1):
        """Set the network configuration.

        Parameters
        ----------
        machines : list, set or string
            Names of machines.
        local_listen_port : int, optional (default=12400)
            TCP listen port for local machines.
        listen_time_out : int, optional (default=120)
            Socket time-out in minutes.
        num_machines : int, optional (default=1)
            The number of machines for parallel learning application.

        Returns
        -------
        self : Booster
            Booster with set network.
        """
        _safe_call(_LIB.LGBM_NetworkInit(c_str(machines),
                                         ctypes.c_int(local_listen_port),
                                         ctypes.c_int(listen_time_out),
                                         ctypes.c_int(num_machines)))
        self.network = True
        return self

    def free_network(self):
        """Free Booster's network.

        Returns
        -------
        self : Booster
            Booster with freed network.
        """
        _safe_call(_LIB.LGBM_NetworkFree())
        self.network = False
        return self

    def set_train_data_name(self, name):
        """Set the name to the training Dataset.

        Parameters
        ----------
        name : string
            Name for the training Dataset.

        Returns
        -------
        self : Booster
            Booster with set training Dataset name.
        """
        self._train_data_name = name
        return self

    def add_valid(self, data, name):
        """Add validation data.

        Parameters
        ----------
        data : Dataset
            Validation data.
        name : string
            Name of validation data.

        Returns
        -------
        self : Booster
            Booster with set validation data.
        """
        if not isinstance(data, Dataset):
            raise TypeError('Validation data should be Dataset instance, met {}'
                            .format(type(data).__name__))
        if data._predictor is not self.__init_predictor:
            raise GPBoostError("Add validation data failed, "
                               "you should use same predictor for these data")
        _safe_call(_LIB.LGBM_BoosterAddValidData(
            self.handle,
            data.construct().handle))
        self.valid_sets.append(data)
        self.name_valid_sets.append(name)
        self.__num_dataset += 1
        self.__inner_predict_buffer.append(None)
        self.__is_predicted_cur_iter.append(False)
        return self

    def reset_parameter(self, params):
        """Reset parameters of Booster.

        Parameters
        ----------
        params : dict
            New parameters for Booster.

        Returns
        -------
        self : Booster
            Booster with new parameters.
        """
        if any(metric_alias in params for metric_alias in _ConfigAliases.get("metric")):
            self.__need_reload_eval_info = True
        params_str = param_dict_to_str(params)
        if params_str:
            _safe_call(_LIB.LGBM_BoosterResetParameter(
                self.handle,
                c_str(params_str)))
        self.params.update(params)
        return self

    def update(self, train_set=None, fobj=None):
        """Update Booster for one iteration.

        Parameters
        ----------
        train_set : Dataset or None, optional (default=None)
            Training data.
            If None, last training data is used.
        fobj : callable or None, optional (default=None)
            Customized objective function.
            Should accept two parameters: preds, train_data,
            and return (grad, hess).

                preds : list or numpy 1-D array
                    The predicted values.
                train_data : Dataset
                    The training dataset.
                grad : list or numpy 1-D array
                    The value of the first order derivative (gradient) for each sample point.
                hess : list or numpy 1-D array
                    The value of the second order derivative (Hessian) for each sample point.

            For multi-class task, the preds is group by class_id first, then group by row_id.
            If you want to get i-th row preds in j-th class, the access way is score[j * num_data + i]
            and you should group grad and hess in this way as well.

        Returns
        -------
        is_finished : bool
            Whether the update was successfully finished.
        """
        # need reset training data
        if train_set is not None and train_set is not self.train_set:
            if not isinstance(train_set, Dataset):
                raise TypeError('Training data should be Dataset instance, met {}'
                                .format(type(train_set).__name__))
            if train_set._predictor is not self.__init_predictor:
                raise GPBoostError("Replace training data failed, "
                                   "you should use same predictor for these data")
            self.train_set = train_set
            _safe_call(_LIB.LGBM_BoosterResetTrainingData(
                self.handle,
                self.train_set.construct().handle))
            self.__inner_predict_buffer[0] = None
        is_finished = ctypes.c_int(0)
        if fobj is None:
            if self.__set_objective_to_none:
                raise GPBoostError('Cannot update due to null objective function.')
            _safe_call(_LIB.LGBM_BoosterUpdateOneIter(
                self.handle,
                ctypes.byref(is_finished)))
            self.__is_predicted_cur_iter = [False for _ in range_(self.__num_dataset)]
            return is_finished.value == 1
        else:
            if not self.__set_objective_to_none:
                self.reset_parameter({"objective": "none"}).__set_objective_to_none = True
            grad, hess = fobj(self.__inner_predict(0), self.train_set)
            return self.__boost(grad, hess)

    def __boost(self, grad, hess):
        """Boost Booster for one iteration with customized gradient statistics.

        .. note::

            For multi-class task, the score is group by class_id first, then group by row_id.
            If you want to get i-th row score in j-th class, the access way is score[j * num_data + i]
            and you should group grad and hess in this way as well.

        Parameters
        ----------
        grad : list or numpy 1-D array
            The first order derivative (gradient).
        hess : list or numpy 1-D array
            The second order derivative (Hessian).

        Returns
        -------
        is_finished : bool
            Whether the boost was successfully finished.
        """
        grad = list_to_1d_numpy(grad, name='gradient')
        hess = list_to_1d_numpy(hess, name='hessian')
        assert grad.flags.c_contiguous
        assert hess.flags.c_contiguous
        if len(grad) != len(hess):
            raise ValueError("Lengths of gradient({}) and hessian({}) don't match"
                             .format(len(grad), len(hess)))
        is_finished = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterUpdateOneIterCustom(
            self.handle,
            grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            hess.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(is_finished)))
        self.__is_predicted_cur_iter = [False for _ in range_(self.__num_dataset)]
        return is_finished.value == 1

    def rollback_one_iter(self):
        """Rollback one iteration.

        Returns
        -------
        self : Booster
            Booster with rolled back one iteration.
        """
        _safe_call(_LIB.LGBM_BoosterRollbackOneIter(
            self.handle))
        self.__is_predicted_cur_iter = [False for _ in range_(self.__num_dataset)]
        return self

    def current_iteration(self):
        """Get the index of the current iteration.

        Returns
        -------
        cur_iter : int
            The index of the current iteration.
        """
        out_cur_iter = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetCurrentIteration(
            self.handle,
            ctypes.byref(out_cur_iter)))
        return out_cur_iter.value

    def num_model_per_iteration(self):
        """Get number of models per iteration.

        Returns
        -------
        model_per_iter : int
            The number of models per iteration.
        """
        model_per_iter = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterNumModelPerIteration(
            self.handle,
            ctypes.byref(model_per_iter)))
        return model_per_iter.value

    def num_trees(self):
        """Get number of weak sub-models.

        Returns
        -------
        num_trees : int
            The number of weak sub-models.
        """
        num_trees = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterNumberOfTotalModel(
            self.handle,
            ctypes.byref(num_trees)))
        return num_trees.value

    def eval(self, data, name, feval=None):
        """Evaluate for data.

        Parameters
        ----------
        data : Dataset
            Data for the evaluating.
        name : string
            Name of the data.
        feval : callable or None, optional (default=None)
            Customized evaluation function.
            Should accept two parameters: preds, eval_data,
            and return (eval_name, eval_result, is_higher_better) or list of such tuples.

                preds : list or numpy 1-D array
                    The predicted values.
                eval_data : Dataset
                    The evaluation dataset.
                eval_name : string
                    The name of evaluation function (without whitespaces).
                eval_result : float
                    The eval result.
                is_higher_better : bool
                    Is eval result higher better, e.g. AUC is ``is_higher_better``.

            For multi-class task, the preds is group by class_id first, then group by row_id.
            If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].

        Returns
        -------
        result : list
            List with evaluation results.
        """
        if not isinstance(data, Dataset):
            raise TypeError("Can only eval for Dataset instance")
        data_idx = -1
        if data is self.train_set:
            data_idx = 0
        else:
            for i in range_(len(self.valid_sets)):
                if data is self.valid_sets[i]:
                    data_idx = i + 1
                    break
        # need to push new valid data
        if data_idx == -1:
            self.add_valid(data, name)
            data_idx = self.__num_dataset - 1

        return self.__inner_eval(name, data_idx, feval)

    def eval_train(self, feval=None):
        """Evaluate for training data.

        Parameters
        ----------
        feval : callable or None, optional (default=None)
            Customized evaluation function.
            Should accept two parameters: preds, train_data,
            and return (eval_name, eval_result, is_higher_better) or list of such tuples.

                preds : list or numpy 1-D array
                    The predicted values.
                train_data : Dataset
                    The training dataset.
                eval_name : string
                    The name of evaluation function (without whitespaces).
                eval_result : float
                    The eval result.
                is_higher_better : bool
                    Is eval result higher better, e.g. AUC is ``is_higher_better``.

            For multi-class task, the preds is group by class_id first, then group by row_id.
            If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].

        Returns
        -------
        result : list
            List with evaluation results.
        """
        return self.__inner_eval(self._train_data_name, 0, feval)

    def eval_valid(self, feval=None):
        """Evaluate for validation data.

        Parameters
        ----------
        feval : callable or None, optional (default=None)
            Customized evaluation function.
            Should accept two parameters: preds, valid_data,
            and return (eval_name, eval_result, is_higher_better) or list of such tuples.

                preds : list or numpy 1-D array
                    The predicted values.
                valid_data : Dataset
                    The validation dataset.
                eval_name : string
                    The name of evaluation function (without whitespaces).
                eval_result : float
                    The eval result.
                is_higher_better : bool
                    Is eval result higher better, e.g. AUC is ``is_higher_better``.

            For multi-class task, the preds is group by class_id first, then group by row_id.
            If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].

        Returns
        -------
        result : list
            List with evaluation results.
        """
        return [item for i in range_(1, self.__num_dataset)
                for item in self.__inner_eval(self.name_valid_sets[i - 1], i, feval)]

    def save_model(self, filename, num_iteration=None, start_iteration=0):
        """Save Booster to file.

        Parameters
        ----------
        filename : string
            Filename to save Booster.
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.

        Returns
        -------
        self : Booster
            Returns self.
        """
        if num_iteration is None:
            num_iteration = self.best_iteration
        _safe_call(_LIB.LGBM_BoosterSaveModel(
            self.handle,
            ctypes.c_int(start_iteration),
            ctypes.c_int(num_iteration),
            c_str(filename)))
        _dump_pandas_categorical(self.pandas_categorical, filename)
        return self

    def shuffle_models(self, start_iteration=0, end_iteration=-1):
        """Shuffle models.

        Parameters
        ----------
        start_iteration : int, optional (default=0)
            The first iteration that will be shuffled.
        end_iteration : int, optional (default=-1)
            The last iteration that will be shuffled.
            If <= 0, means the last available iteration.

        Returns
        -------
        self : Booster
            Booster with shuffled models.
        """
        _safe_call(_LIB.LGBM_BoosterShuffleModels(
            self.handle,
            ctypes.c_int(start_iteration),
            ctypes.c_int(end_iteration)))
        return self

    def model_from_string(self, model_str, verbose=True):
        """Load Booster from a string.

        Parameters
        ----------
        model_str : string
            Model will be loaded from this string.
        verbose : bool, optional (default=True)
            Whether to print messages while loading model.

        Returns
        -------
        self : Booster
            Loaded Booster object.
        """
        if self.handle is not None:
            _safe_call(_LIB.LGBM_BoosterFree(self.handle))
        self._free_buffer()
        self.handle = ctypes.c_void_p()
        out_num_iterations = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterLoadModelFromString(
            c_str(model_str),
            ctypes.byref(out_num_iterations),
            ctypes.byref(self.handle)))
        out_num_class = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetNumClasses(
            self.handle,
            ctypes.byref(out_num_class)))
        if verbose:
            print('Finished loading model, total used %d iterations' % int(out_num_iterations.value))
        self.__num_class = out_num_class.value
        self.pandas_categorical = _load_pandas_categorical(model_str=model_str)
        return self

    def model_to_string(self, num_iteration=None, start_iteration=0):
        """Save Booster to string.

        Parameters
        ----------
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.

        Returns
        -------
        str_repr : string
            String representation of Booster.
        """
        if num_iteration is None:
            num_iteration = self.best_iteration
        buffer_len = 1 << 20
        tmp_out_len = ctypes.c_int64(0)
        string_buffer = ctypes.create_string_buffer(buffer_len)
        ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
        _safe_call(_LIB.LGBM_BoosterSaveModelToString(
            self.handle,
            ctypes.c_int(start_iteration),
            ctypes.c_int(num_iteration),
            ctypes.c_int64(buffer_len),
            ctypes.byref(tmp_out_len),
            ptr_string_buffer))
        actual_len = tmp_out_len.value
        # if buffer length is not long enough, re-allocate a buffer
        if actual_len > buffer_len:
            string_buffer = ctypes.create_string_buffer(actual_len)
            ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
            _safe_call(_LIB.LGBM_BoosterSaveModelToString(
                self.handle,
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                ctypes.c_int64(actual_len),
                ctypes.byref(tmp_out_len),
                ptr_string_buffer))
        ret = string_buffer.value.decode()
        ret += _dump_pandas_categorical(self.pandas_categorical)
        return ret

    def dump_model(self, num_iteration=None, start_iteration=0):
        """Dump Booster to JSON format.

        Parameters
        ----------
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be dumped.
            If None, if the best iteration exists, it is dumped; otherwise, all iterations are dumped.
            If <= 0, all iterations are dumped.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be dumped.

        Returns
        -------
        json_repr : dict
            JSON format of Booster.
        """
        if num_iteration is None:
            num_iteration = self.best_iteration
        buffer_len = 1 << 20
        tmp_out_len = ctypes.c_int64(0)
        string_buffer = ctypes.create_string_buffer(buffer_len)
        ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
        _safe_call(_LIB.LGBM_BoosterDumpModel(
            self.handle,
            ctypes.c_int(start_iteration),
            ctypes.c_int(num_iteration),
            ctypes.c_int64(buffer_len),
            ctypes.byref(tmp_out_len),
            ptr_string_buffer))
        actual_len = tmp_out_len.value
        # if buffer length is not long enough, reallocate a buffer
        if actual_len > buffer_len:
            string_buffer = ctypes.create_string_buffer(actual_len)
            ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
            _safe_call(_LIB.LGBM_BoosterDumpModel(
                self.handle,
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                ctypes.c_int64(actual_len),
                ctypes.byref(tmp_out_len),
                ptr_string_buffer))
        ret = json.loads(string_buffer.value.decode())
        ret['pandas_categorical'] = json.loads(json.dumps(self.pandas_categorical,
                                                          default=json_default_with_numpy))
        return ret

    def predict(self, data, num_iteration=None,
                raw_score=False, pred_leaf=False, pred_contrib=False,
                data_has_header=False, is_reshape=True,
                group_data_pred=None, group_rand_coef_data_pred=None,
                gp_coords_pred=None, gp_rand_coef_data_pred=None,
                cluster_ids_pred=None, vecchia_pred_type=None,
                num_neighbors_pred=-1, predict_cov_mat=False, **kwargs):
        """Make a prediction.

        Parameters
        ----------
        data : string, numpy array, pandas DataFrame, H2O DataTable's Frame or scipy.sparse
            Data source for prediction.
            If string, it represents the path to txt file.
        num_iteration : int or None, optional (default=None)
            Limit number of iterations in the prediction.
            If None, if the best iteration exists, it is used; otherwise, all iterations are used.
            If <= 0, all iterations are used (no limits).
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.

            .. note::

                If you want to get more explanations for your model's predictions using SHAP values,
                like SHAP interaction values,
                you can install the shap package (https://github.com/slundberg/shap).
                Note that unlike the shap package, with ``pred_contrib`` we return a matrix with an extra
                column, where the last column is the expected value.

        data_has_header : bool, optional (default=False)
            Whether the data has header.
            Used only if data is string.
        is_reshape : bool, optional (default=True)
            If True, result is reshaped to [nrow, ncol].
        group_data_pred : numpy array with numeric or string data or None, optional (default=None)
                Labels of group levels for grouped random effects. Used only if the Booster has a GPModel
        group_rand_coef_data_pred : numpy array with numeric data or None, optional (default=None)
            Covariate data for grouped random coefficients. Used only if the Booster has a GPModel
        gp_coords_pred : numpy array with numeric data or None, optional (default=None)
            Coordinates (features) for Gaussian process. Used only if the Booster has a GPModel
        gp_rand_coef_data_pred : numpy array with numeric data or None, optional (default=None)
            Covariate data for Gaussian process random coefficients. Used only if the Booster has a GPModel
        vecchia_pred_type : string, optional (default="order_obs_first_cond_obs_only")
            Type of Vecchia approximation used for making predictions. Used only if the Booster has a GPModel.
            "order_obs_first_cond_obs_only" = observed data is ordered first and the neighbors are only observed
            points, "order_obs_first_cond_all" = observed data is ordered first and the neighbors are selected
            among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for
            making predictions, "latent_order_obs_first_cond_obs_only" = Vecchia approximation for the latent
            process and observed data is ordered first and neighbors are only observed points,
            "latent_order_obs_first_cond_all" = Vecchia approximation for the latent process and observed data is
            ordered first and neighbors are selected among all points
        num_neighbors_pred : integer or None, optional (default=None)
            Number of neighbors for the Vecchia approximation for making predictions. Used only if the Booster has a GPModel
        cluster_ids_pred : one dimensional numpy array (vector) with integer data or None, optional (default=None)
            IDs / labels indicating independent realizations of random effects / Gaussian processes
            (same values = same process realization). Used only if the Booster has a GPModel
        predict_cov_mat : bool, optional (default=False)
            If True, the (posterior / conditional) predictive covariance is calculated in addition to the
            (posterior / conditional) predictive mean. Used only if the Booster has a GPModel
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        result : either a numpy array (if there is no GPModel) or a dict with three entries each having numpy arrays
                as values (if there is a GPModel)
            If there is no GPModel: Predictions from the tree-booster.
            If there is a GPModel: Separate predictions for the tree-ensemble (=fixed_effect) and the GP / random effect
            The second entry of the dict result['fixed_effect'] are the predictions from the tree-booster.
            The second entry of the dict result['random_effect_mean'] is the predicted mean of the GP / random effect.
            The third entry result['random_effect_cov'] is the predicted covariance matrix of the GP / random effect
            (=None if 'predict_cov_mat=False'). I.e., to get a point prediction of the entire model, one has to
            calculate the sum result['fixed_effect'] + result['random_effect_mean'].
        """
        predictor = self._to_predictor(copy.deepcopy(kwargs))
        if num_iteration is None:
            num_iteration = self.best_iteration
        if self.has_gp_model:
            if self.train_set.data is None:
                raise GPBoostError("cannot make predictions for Gaussian process. "
                "Set free_raw_data = False when you construct the Dataset")
            fixed_effect_train = predictor.predict(self.train_set.data, num_iteration,
                                                   False, False, False, False, False)
            residual = self.train_set.label - fixed_effect_train
            random_effect_pred = self.gp_model.predict(y=residual, group_data_pred=group_data_pred,
                                                       group_rand_coef_data_pred=group_rand_coef_data_pred,
                                                       gp_coords_pred=gp_coords_pred,
                                                       gp_rand_coef_data_pred=gp_rand_coef_data_pred,
                                                       cluster_ids_pred=cluster_ids_pred,
                                                       vecchia_pred_type=vecchia_pred_type,
                                                       num_neighbors_pred=num_neighbors_pred,
                                                       predict_cov_mat=predict_cov_mat)
            fixed_effect = predictor.predict(data, num_iteration,
                                              raw_score, pred_leaf, pred_contrib,
                                              data_has_header, is_reshape)
            if len(fixed_effect) != len(random_effect_pred['mu']):
                warnings.warn("Number of data points in fixed effect (tree ensemble) and random effect "
                              "(Gaussian process) are not equal")
            return {"fixed_effect": fixed_effect, "random_effect_mean": random_effect_pred['mu'],
                     "random_effect_cov": random_effect_pred['cov']}
        else:
            return predictor.predict(data, num_iteration,
                                     raw_score, pred_leaf, pred_contrib,
                                     data_has_header, is_reshape)

    def refit(self, data, label, decay_rate=0.9, **kwargs):
        """Refit the existing Booster by new data.

        Parameters
        ----------
        data : string, numpy array, pandas DataFrame, H2O DataTable's Frame or scipy.sparse
            Data source for refit.
            If string, it represents the path to txt file.
        label : list, numpy 1-D array or pandas Series / one-column DataFrame
            Label for refit.
        decay_rate : float, optional (default=0.9)
            Decay rate of refit,
            will use ``leaf_output = decay_rate * old_leaf_output + (1.0 - decay_rate) * new_leaf_output`` to refit trees.
        **kwargs
            Other parameters for refit.
            These parameters will be passed to ``predict`` method.

        Returns
        -------
        result : Booster
            Refitted Booster.
        """
        if self.__set_objective_to_none:
            raise GPBoostError('Cannot refit due to null objective function.')
        predictor = self._to_predictor(copy.deepcopy(kwargs))
        leaf_preds = predictor.predict(data, -1, pred_leaf=True)
        nrow, ncol = leaf_preds.shape
        train_set = Dataset(data, label, silent=True)
        new_params = copy.deepcopy(self.params)
        new_params['refit_decay_rate'] = decay_rate
        new_booster = Booster(new_params, train_set, silent=True)
        # Copy models
        _safe_call(_LIB.LGBM_BoosterMerge(
            new_booster.handle,
            predictor.handle))
        leaf_preds = leaf_preds.reshape(-1)
        ptr_data, type_ptr_data, _ = c_int_array(leaf_preds)
        _safe_call(_LIB.LGBM_BoosterRefit(
            new_booster.handle,
            ptr_data,
            ctypes.c_int(nrow),
            ctypes.c_int(ncol)))
        new_booster.network = self.network
        new_booster.__attr = self.__attr.copy()
        return new_booster

    def get_leaf_output(self, tree_id, leaf_id):
        """Get the output of a leaf.

        Parameters
        ----------
        tree_id : int
            The index of the tree.
        leaf_id : int
            The index of the leaf in the tree.

        Returns
        -------
        result : float
            The output of the leaf.
        """
        ret = ctypes.c_double(0)
        _safe_call(_LIB.LGBM_BoosterGetLeafValue(
            self.handle,
            ctypes.c_int(tree_id),
            ctypes.c_int(leaf_id),
            ctypes.byref(ret)))
        return ret.value

    def _to_predictor(self, pred_parameter=None):
        """Convert to predictor."""
        predictor = _InnerPredictor(booster_handle=self.handle, pred_parameter=pred_parameter)
        predictor.pandas_categorical = self.pandas_categorical
        return predictor

    def num_feature(self):
        """Get number of features.

        Returns
        -------
        num_feature : int
            The number of features.
        """
        out_num_feature = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetNumFeature(
            self.handle,
            ctypes.byref(out_num_feature)))
        return out_num_feature.value

    def feature_name(self):
        """Get names of features.

        Returns
        -------
        result : list
            List with names of features.
        """
        num_feature = self.num_feature()
        # Get name of features
        tmp_out_len = ctypes.c_int(0)
        string_buffers = [ctypes.create_string_buffer(255) for i in range_(num_feature)]
        ptr_string_buffers = (ctypes.c_char_p * num_feature)(*map(ctypes.addressof, string_buffers))
        _safe_call(_LIB.LGBM_BoosterGetFeatureNames(
            self.handle,
            ctypes.byref(tmp_out_len),
            ptr_string_buffers))
        if num_feature != tmp_out_len.value:
            raise ValueError("Length of feature names doesn't equal with num_feature")
        return [string_buffers[i].value.decode() for i in range_(num_feature)]

    def feature_importance(self, importance_type='split', iteration=None):
        """Get feature importances.

        Parameters
        ----------
        importance_type : string, optional (default="split")
            How the importance is calculated.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.
        iteration : int or None, optional (default=None)
            Limit number of iterations in the feature importance calculation.
            If None, if the best iteration exists, it is used; otherwise, all trees are used.
            If <= 0, all trees are used (no limits).

        Returns
        -------
        result : numpy array
            Array with feature importances.
        """
        if iteration is None:
            iteration = self.best_iteration
        if importance_type == "split":
            importance_type_int = 0
        elif importance_type == "gain":
            importance_type_int = 1
        else:
            importance_type_int = -1
        result = np.zeros(self.num_feature(), dtype=np.float64)
        _safe_call(_LIB.LGBM_BoosterFeatureImportance(
            self.handle,
            ctypes.c_int(iteration),
            ctypes.c_int(importance_type_int),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if importance_type_int == 0:
            return result.astype(np.int32)
        else:
            return result

    def get_split_value_histogram(self, feature, bins=None, xgboost_style=False):
        """Get split value histogram for the specified feature.

        Parameters
        ----------
        feature : int or string
            The feature name or index the histogram is calculated for.
            If int, interpreted as index.
            If string, interpreted as name.

            .. warning::

                Categorical features are not supported.

        bins : int, string or None, optional (default=None)
            The maximum number of bins.
            If None, or int and > number of unique split values and ``xgboost_style=True``,
            the number of bins equals number of unique split values.
            If string, it should be one from the list of the supported values by ``numpy.histogram()`` function.
        xgboost_style : bool, optional (default=False)
            Whether the returned result should be in the same form as it is in XGBoost.
            If False, the returned value is tuple of 2 numpy arrays as it is in ``numpy.histogram()`` function.
            If True, the returned value is matrix, in which the first column is the right edges of non-empty bins
            and the second one is the histogram values.

        Returns
        -------
        result_tuple : tuple of 2 numpy arrays
            If ``xgboost_style=False``, the values of the histogram of used splitting values for the specified feature
            and the bin edges.
        result_array_like : numpy array or pandas DataFrame (if pandas is installed)
            If ``xgboost_style=True``, the histogram of used splitting values for the specified feature.
        """

        def add(root):
            """Recursively add thresholds."""
            if 'split_index' in root:  # non-leaf
                if feature_names is not None and isinstance(feature, string_type):
                    split_feature = feature_names[root['split_feature']]
                else:
                    split_feature = root['split_feature']
                if split_feature == feature:
                    if isinstance(root['threshold'], string_type):
                        raise GPBoostError('Cannot compute split value histogram for the categorical feature')
                    else:
                        values.append(root['threshold'])
                add(root['left_child'])
                add(root['right_child'])

        model = self.dump_model()
        feature_names = model.get('feature_names')
        tree_infos = model['tree_info']
        values = []
        for tree_info in tree_infos:
            add(tree_info['tree_structure'])

        if bins is None or isinstance(bins, integer_types) and xgboost_style:
            n_unique = len(np.unique(values))
            bins = max(min(n_unique, bins) if bins is not None else n_unique, 1)
        hist, bin_edges = np.histogram(values, bins=bins)
        if xgboost_style:
            ret = np.column_stack((bin_edges[1:], hist))
            ret = ret[ret[:, 1] > 0]
            if PANDAS_INSTALLED:
                return DataFrame(ret, columns=['SplitValue', 'Count'])
            else:
                return ret
        else:
            return hist, bin_edges

    def __inner_eval(self, data_name, data_idx, feval=None):
        """Evaluate training or validation data."""
        if data_idx >= self.__num_dataset:
            raise ValueError("Data_idx should be smaller than number of dataset")
        self.__get_eval_info()
        ret = []
        if self.__num_inner_eval > 0:
            result = np.zeros(self.__num_inner_eval, dtype=np.float64)
            tmp_out_len = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetEval(
                self.handle,
                ctypes.c_int(data_idx),
                ctypes.byref(tmp_out_len),
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
            if tmp_out_len.value != self.__num_inner_eval:
                raise ValueError("Wrong length of eval results")
            for i in range_(self.__num_inner_eval):
                ret.append((data_name, self.__name_inner_eval[i],
                            result[i], self.__higher_better_inner_eval[i]))
        if feval is not None:
            if data_idx == 0:
                cur_data = self.train_set
            else:
                cur_data = self.valid_sets[data_idx - 1]
            feval_ret = feval(self.__inner_predict(data_idx), cur_data)
            if isinstance(feval_ret, list):
                for eval_name, val, is_higher_better in feval_ret:
                    ret.append((data_name, eval_name, val, is_higher_better))
            else:
                eval_name, val, is_higher_better = feval_ret
                ret.append((data_name, eval_name, val, is_higher_better))
        return ret

    def __inner_predict(self, data_idx):
        """Predict for training and validation dataset."""
        if data_idx >= self.__num_dataset:
            raise ValueError("Data_idx should be smaller than number of dataset")
        if self.__inner_predict_buffer[data_idx] is None:
            if data_idx == 0:
                n_preds = self.train_set.num_data() * self.__num_class
            else:
                n_preds = self.valid_sets[data_idx - 1].num_data() * self.__num_class
            self.__inner_predict_buffer[data_idx] = np.zeros(n_preds, dtype=np.float64)
        # avoid to predict many time in one iteration
        if not self.__is_predicted_cur_iter[data_idx]:
            tmp_out_len = ctypes.c_int64(0)
            data_ptr = self.__inner_predict_buffer[data_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            _safe_call(_LIB.LGBM_BoosterGetPredict(
                self.handle,
                ctypes.c_int(data_idx),
                ctypes.byref(tmp_out_len),
                data_ptr))
            if tmp_out_len.value != len(self.__inner_predict_buffer[data_idx]):
                raise ValueError("Wrong length of predict results for data %d" % (data_idx))
            self.__is_predicted_cur_iter[data_idx] = True
        return self.__inner_predict_buffer[data_idx]

    def __get_eval_info(self):
        """Get inner evaluation count and names."""
        if self.__need_reload_eval_info:
            self.__need_reload_eval_info = False
            out_num_eval = ctypes.c_int(0)
            # Get num of inner evals
            _safe_call(_LIB.LGBM_BoosterGetEvalCounts(
                self.handle,
                ctypes.byref(out_num_eval)))
            self.__num_inner_eval = out_num_eval.value
            if self.__num_inner_eval > 0:
                # Get name of evals
                tmp_out_len = ctypes.c_int(0)
                string_buffers = [ctypes.create_string_buffer(255) for i in range_(self.__num_inner_eval)]
                ptr_string_buffers = (ctypes.c_char_p * self.__num_inner_eval)(*map(ctypes.addressof, string_buffers))
                _safe_call(_LIB.LGBM_BoosterGetEvalNames(
                    self.handle,
                    ctypes.byref(tmp_out_len),
                    ptr_string_buffers))
                if self.__num_inner_eval != tmp_out_len.value:
                    raise ValueError("Length of eval names doesn't equal with num_evals")
                self.__name_inner_eval = \
                    [string_buffers[i].value.decode() for i in range_(self.__num_inner_eval)]
                self.__higher_better_inner_eval = \
                    [name.startswith(('auc', 'ndcg@', 'map@')) for name in self.__name_inner_eval]

    def attr(self, key):
        """Get attribute string from the Booster.

        Parameters
        ----------
        key : string
            The name of the attribute.

        Returns
        -------
        value : string or None
            The attribute value.
            Returns None if attribute does not exist.
        """
        return self.__attr.get(key, None)

    def set_attr(self, **kwargs):
        """Set attributes to the Booster.

        Parameters
        ----------
        **kwargs
            The attributes to set.
            Setting a value to None deletes an attribute.

        Returns
        -------
        self : Booster
            Booster with set attributes.
        """
        for key, value in kwargs.items():
            if value is not None:
                if not isinstance(value, string_type):
                    raise ValueError("Only string values are accepted")
                self.__attr[key] = value
            else:
                self.__attr.pop(key, None)
        return self


class GPModel(object):
    """Gaussian process or mixed effects model ."""

    _SUPPORTED_COV_FUNCTIONS = ("exponential", "gaussian", "powered_exponential", "matern")
    _SUPPORTED_VECCHIA_ORDERING = ("none", "random")
    _VECCHIA_PRED_TYPES = ("order_obs_first_cond_obs_only",
                           "order_obs_first_cond_all", "order_pred_first",
                           "latent_order_obs_first_cond_obs_only", "latent_order_obs_first_cond_all")

    def __init__(self, group_data=None,
                 group_rand_coef_data=None,
                 ind_effect_group_rand_coef=None,
                 gp_coords=None,
                 gp_rand_coef_data=None,
                 cov_function="exponential",
                 cov_fct_shape=0.,
                 vecchia_approx=False,
                 num_neighbors=30,
                 vecchia_ordering="none",
                 vecchia_pred_type="order_obs_first_cond_obs_only",
                 num_neighbors_pred=None,
                 cluster_ids=None,
                 free_raw_data=False):
        """Initialize a GPModel.

        Parameters
        ----------
            group_data : numpy array with numeric or string data or None, optional (default=None)
                Labels of group levels for grouped random effects
            group_rand_coef_data : numpy array with numeric data or None, optional (default=None)
                Covariate data for grouped random coefficients
            ind_effect_group_rand_coef  : one dimensional numpy array (vector) with integer data or None, optional (default=None)
                Indices that relate every random coefficients to a "base" intercept grouped random effect. Counting starts at 1.
            gp_coords : numpy array with numeric data or None, optional (default=None)
                Coordinates (features) for Gaussian process
            gp_rand_coef_data : numpy array with numeric data or None, optional (default=None)
                Covariate data for Gaussian process random coefficients
            cov_function : string, optional (default="exponential")
                Covariance function for the Gaussian process. The following covariance functions are available:
                "exponential", "gaussian", "matern", and "powered_exponential". We follow the notation and
                parametrization of Diggle and Ribeiro (2007) except for the Matern covariance where we follow
                Rassmusen and Williams (2006)
            cov_fct_shape : float, optional (default=0.)
                Shape parameter of covariance function (=smoothness parameter for Matern covariance,
                irrelevant for some covariance functions such as the exponential or Gaussian)
            vecchia_approx : bool, optional (default=False)
                If true, the Vecchia approximation is used
            num_neighbors : integer, optional (default=30)
                Number of neighbors for the Vecchia approximation
            vecchia_ordering : string, optional (default="none")
                Ordering used in the Vecchia approximation. "none" means the default ordering is used,
                "random" uses a random ordering
            vecchia_pred_type : string, optional (default="order_obs_first_cond_obs_only")
                Type of Vecchia approximation used for making predictions.
                "order_obs_first_cond_obs_only" = observed data is ordered first and the neighbors are only observed
                points, "order_obs_first_cond_all" = observed data is ordered first and the neighbors are selected
                among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for
                making predictions, "latent_order_obs_first_cond_obs_only" = Vecchia approximation for the latent
                process and observed data is ordered first and neighbors are only observed points,
                "latent_order_obs_first_cond_all" = Vecchia approximation for the latent process and observed data
                is ordered first and neighbors are selected among all points
            num_neighbors_pred : integer or None, optional (default=None)
                Number of neighbors for the Vecchia approximation for making predictions
            cluster_ids : one dimensional numpy array (vector) with integer data or None, optional (default=None)
                IDs / labels indicating independent realizations of random effects / Gaussian processes
                (same values = same process realization)
            free_raw_data : bool, optional (default=False)
                If True, the data (groups, coordinates, covariate data for random coefficients) is freed in Python
                after initialization
        """

        if num_neighbors_pred is None:
            num_neighbors_pred = num_neighbors

        if group_data is None and gp_coords is None:
            raise ValueError("Both group_data and gp_coords are None. Provide at least one of them")

        # Initialize variables
        self.handle = ctypes.c_void_p()
        self.num_data = None
        self.num_group_re = 0
        self.num_group_rand_coef = 0
        self.num_cov_pars = 1
        self.num_gp = 0
        self.dim_coords = 2
        self.num_gp_rand_coef = 0
        self.has_covariates = False
        self.num_coef = None
        self.std_dev = False
        self.group_data = None
        self.group_rand_coef_data = None
        self.ind_effect_group_rand_coef = None
        self.gp_coords = None
        self.gp_rand_coef_data = None
        self.cov_function = "exponential"
        self.cov_fct_shape = 0.
        self.vecchia_approx = False
        self.num_neighbors = 30
        self.vecchia_ordering = "none"
        self.vecchia_pred_type = "order_obs_first_cond_obs_only"
        self.num_neighbors_pred = 30
        self.cov_par_names = ["Error_term"]
        self.coef_names = None
        self.cluster_ids = None
        self.free_raw_data = False
        self.num_data_pred = None
        self.params = {"optimizer_cov": "fisher_scoring",
                      "optimizer_coef": "wls",
                      "maxit": 1000,
                      "delta_rel_conv": 1e-6,
                      "init_coef": None,
                      "init_cov_pars": None,
                      "lr_coef": 0.01,
                      "lr_cov": 0.01,
                      "use_nesterov_acc": False,
                      "acc_rate_coef": 0.1,
                      "acc_rate_cov": 0.5,
                      "nesterov_schedule_version": 0,
                      "momentum_offset": 2,
                      "trace": False}
        self.prediction_data_is_set = False
        self.free_raw_data = False

        # Define default NULL values for calling C function
        group_data_c = ctypes.c_void_p()
        group_rand_coef_data_c = ctypes.c_void_p()
        ind_effect_group_rand_coef_c = ctypes.c_void_p()
        gp_coords_c = ctypes.c_void_p()
        gp_rand_coef_data_c = ctypes.c_void_p()
        cluster_ids_c = ctypes.c_void_p()
        # Set data for grouped random effects
        if group_data is not None:  ##TODO: add support for pandas
            if not isinstance(group_data, np.ndarray):
                raise ValueError("group_data needs to be a numpy.ndarray")
            if len(group_data.shape) > 2:
                raise ValueError("group_data needs to be either a vector or a two-dimensional matrix (array)")
            self.group_data = copy.deepcopy(group_data)
            if len(self.group_data.shape) == 1:
                self.group_data = self.group_data.reshape((len(self.group_data), 1))
            self.num_group_re = self.group_data.shape[1]
            self.num_data = self.group_data.shape[0]
            self.num_cov_pars = self.num_cov_pars + self.num_group_re
            if self.group_data.dtype.names is None:
                for ig in range(self.num_group_re):
                    self.cov_par_names.append('Group_' + str(ig + 1))
            else:
                self.cov_par_names.extend(list(self.group_data.dtype.names))
            self.group_data = self.group_data.astype(np.dtype(str))
            # Convert to correct format for passing to C
            group_data_c = self.group_data.flatten(order='F')
            group_data_c = string_array_c_str(group_data_c)
            # Set data for grouped random coefficients
            if group_rand_coef_data is not None:
                if not isinstance(group_rand_coef_data, np.ndarray):
                    raise ValueError("group_rand_coef_data needs to be a numpy.ndarray")
                if len(group_rand_coef_data.shape) > 2:
                    raise ValueError(
                        "group_rand_coef_data needs to be either a vector or a two-dimensional matrix (array)")
                self.group_rand_coef_data = copy.deepcopy(group_rand_coef_data)
                if len(self.group_rand_coef_data.shape) == 1:
                    self.group_rand_coef_data = self.group_rand_coef_data.reshape((len(self.group_rand_coef_data), 1))
                if self.group_rand_coef_data.shape[0] != self.num_data:
                    raise ValueError("Incorrect number of data points in group_rand_coef_data")
                self.group_rand_coef_data = self.group_rand_coef_data.astype(np.dtype(np.float64))
                if ind_effect_group_rand_coef is None:
                    raise ValueError(
                        "Indices of grouped random effects (ind_effect_group_rand_coef) for random slopes in group_rand_coef_data not provided")
                if not isinstance(ind_effect_group_rand_coef, np.ndarray) and not isinstance(ind_effect_group_rand_coef,
                                                                                             list):
                    raise ValueError("ind_effect_group_rand_coef needs to be a numpy.ndarray")
                self.ind_effect_group_rand_coef = copy.deepcopy(ind_effect_group_rand_coef)
                if isinstance(self.ind_effect_group_rand_coef, list):
                    self.ind_effect_group_rand_coef = np.array(self.ind_effect_group_rand_coef)
                if len(self.ind_effect_group_rand_coef.shape) != 1:
                    raise ValueError("ind_effect_group_rand_coef needs to be a vector / one-dimensional numpy.ndarray ")
                if self.ind_effect_group_rand_coef.shape[0] != self.group_rand_coef_data.shape[1]:
                    raise ValueError(
                        "Number of random coefficients in group_rand_coef_data does not match number in ind_effect_group_rand_coef")
                self.ind_effect_group_rand_coef = self.ind_effect_group_rand_coef.astype(np.dtype(int))
                self.num_group_rand_coef = self.group_rand_coef_data.shape[1]
                self.num_cov_pars = self.num_cov_pars + self.num_group_rand_coef
                counter_re = np.zeros(self.num_group_re)
                counter_re.astype(np.dtype(int))
                for ii in range(self.num_group_rand_coef):
                    if self.group_rand_coef_data.dtype.names is None:
                        self.cov_par_names.append(
                            self.cov_par_names[self.ind_effect_group_rand_coef[ii]] + "_rand_coef_nb_" + str(
                                int(counter_re[self.ind_effect_group_rand_coef[ii] - 1] + 1)))
                        counter_re[self.ind_effect_group_rand_coef[ii] - 1] = counter_re[
                                                                                  self.ind_effect_group_rand_coef[
                                                                                      ii] - 1] + 1
                    else:
                        self.cov_par_names.append(
                            self.cov_par_names[self.ind_effect_group_rand_coef[ii]] + "_rand_coef_" +
                            self.group_data.dtype.names[ii])
                self.group_rand_coef_data = self.group_rand_coef_data.astype(np.float64)
                group_rand_coef_data_c, _, _ = c_float_array(self.group_rand_coef_data.flatten(order='F'))
                ind_effect_group_rand_coef_c = self.ind_effect_group_rand_coef.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_int))
        # Set data for Gaussian process
        if gp_coords is not None:
            if not isinstance(gp_coords, np.ndarray):
                raise ValueError("gp_coords needs to be a numpy.ndarray")
            if len(gp_coords.shape) > 2:
                raise ValueError("gp_coords needs to be either a vector or a two-dimensional matrix (array)")
            self.gp_coords = copy.deepcopy(gp_coords)
            if len(self.gp_coords.shape) == 1:
                self.gp_coords = self.gp_coords.reshape((len(self.gp_coords), 1))
            if self.num_data is None:
                self.num_data = self.gp_coords.shape[0]
            else:
                if self.gp_coords.shape[0] != self.num_data:
                    raise ValueError("Incorrect number of data points in gp_coords")
            self.gp_coords = self.gp_coords.astype(np.float64)
            self.num_gp = 1
            self.dim_coords = gp_coords.shape[1]
            self.num_cov_pars = self.num_cov_pars + 2
            self.cov_function = cov_function
            self.cov_fct_shape = cov_fct_shape
            self.vecchia_ordering = vecchia_ordering
            self.vecchia_pred_type = vecchia_pred_type
            self.vecchia_approx = vecchia_approx
            self.num_neighbors = num_neighbors
            self.num_neighbors_pred = num_neighbors_pred
            self.cov_par_names.extend(["GP_var", "GP_range"])
            gp_coords_c, _, _ = c_float_array(self.gp_coords.flatten(order='F'))
            # Set data for GP random coefficients
            if gp_rand_coef_data is not None:
                if not isinstance(gp_rand_coef_data, np.ndarray):
                    raise ValueError("gp_rand_coef_data needs to be a numpy.ndarray")
                if len(gp_rand_coef_data.shape) > 2:
                    raise ValueError(
                        "gp_rand_coef_data needs to be either a vector or a two-dimensional matrix (array)")
                self.gp_rand_coef_data = copy.deepcopy(gp_rand_coef_data)
                if len(self.gp_rand_coef_data.shape) == 1:
                    self.gp_rand_coef_data = self.gp_rand_coef_data.reshape((len(self.gp_rand_coef_data), 1))
                if self.gp_rand_coef_data.shape[0] != self.num_data:
                    raise ValueError("Incorrect number of data points in gp_rand_coef_data")
                self.gp_rand_coef_data = self.gp_rand_coef_data.astype(np.dtype(np.float64))
                self.num_gp_rand_coef = self.gp_rand_coef_data.shape[1]
                self.num_cov_pars = self.num_cov_pars + 2 * self.num_gp_rand_coef
                gp_rand_coef_data_c, _, _ = c_float_array(self.gp_rand_coef_data.flatten(order='F'))
                for ii in range(self.num_gp_rand_coef):
                    if self.gp_rand_coef_data.dtype.names is None:
                        self.cov_par_names.extend(
                            ["GP_rand_coef_nb_" + str(ii+1) + "_var", "GP_rand_coef_nb_" + str(ii+1) + "_range"])
                    else:
                        self.cov_par_names.extend(
                            ["GP_rand_coef_" + self.gp_rand_coef_data.dtype.names[ii] + "_var",
                             "GP_rand_coef_" + self.gp_rand_coef_data.dtype.names[ii] + "_range"])

        # Set IDs for independent processes (cluster_ids)
        if cluster_ids is not None:
            if not isinstance(cluster_ids, np.ndarray):
                raise ValueError("cluster_ids needs to be a numpy.ndarray")
            self.cluster_ids = copy.deepcopy(cluster_ids)
            if len(self.cluster_ids.shape) != 1:
                raise ValueError("cluster_ids needs to be a vector / one-dimensional numpy.ndarray ")
            if self.cluster_ids.shape[0] != self.num_data:
                raise ValueError("Incorrect number of data points in cluster_ids")
            self.cluster_ids = self.cluster_ids.astype(np.dtype(int))
            cluster_ids_c = self.cluster_ids.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int))

        self._check_params()

        _safe_call(_LIB.GPB_CreateREModel(
            ctypes.c_int(self.num_data),
            cluster_ids_c,
            group_data_c,
            ctypes.c_int(self.num_group_re),
            group_rand_coef_data_c,
            ind_effect_group_rand_coef_c,
            ctypes.c_int(self.num_group_rand_coef),
            ctypes.c_int(self.num_gp),
            gp_coords_c,
            ctypes.c_int(self.dim_coords),
            gp_rand_coef_data_c,
            ctypes.c_int(self.num_gp_rand_coef),
            c_str(self.cov_function),
            ctypes.c_double(self.cov_fct_shape),
            ctypes.c_bool(self.vecchia_approx),
            ctypes.c_int(self.num_neighbors),
            c_str(self.vecchia_ordering),
            c_str(self.vecchia_pred_type),
            ctypes.c_int(self.num_neighbors_pred),
            ctypes.byref(self.handle)))

        # Should we free raw data?
        if free_raw_data:
            self.group_data = None
            self.group_rand_coef_data = None
            self.gp_coords = None
            self.gp_rand_coef_data = None
            self.cluster_ids = None

    def _check_params(self):
        if (self.cov_function not in self._SUPPORTED_COV_FUNCTIONS):
            raise ValueError("cov_function '{0:s}' not supported. ".format(self.cov_function))
        if self.cov_function == "powered_exponential":
            if self.cov_fct_shape <= 0. or self.cov_fct_shape > 2.:
                raise ValueError("cov_fct_shape needs to be larger than 0 and smaller or equal than 2 for "
                                 "cov_function=powered_exponential")
        if self.cov_function == "matern":
            if not (self.cov_fct_shape == 0.5 or self.cov_fct_shape == 1.5 or self.cov_fct_shape == 2.5):
                raise ValueError("cov_fct_shape needs to be 0.5, 1.5, or 2.5 for cov_function=matern")
        if (self.vecchia_ordering not in self._SUPPORTED_VECCHIA_ORDERING):
            raise ValueError("vecchia_ordering '{0:s}' not supported. ".format(self.vecchia_ordering))
        if (self.vecchia_pred_type not in self._VECCHIA_PRED_TYPES):
            raise ValueError("vecchia_pred_type '{0:s}' not supported. ".format(self.vecchia_pred_type))

    def __del__(self):
        try:
            if self.handle is not None:
                _safe_call(_LIB.GPB_REModelFree(self.handle))
        except AttributeError:
            pass

    def fit(self, y, X = None, std_dev = False, params = None):
        """Fit / estimate a GPModel using maximum likelihood estimation.

        Parameters
        ----------
        y : numpy array or None, optional (default=None)
            Response variable data
        X : numpy array with numeric data or None, optional (default=None)
            Covariate data for fixed effects ( = linear regression term)
        std_dev : bool (default=False)
            If True (asymptotic) standard deviations are calculated for all parameters
        params : dict or None, optional (default=None)
            Parameters for fitting / optimization:
                optimizer_cov : string, optional (default = "fisher_scoring")
                    Optimizer used for estimating covariance parameters.
                    Options: "gradient_descent" or "fisher_scoring"
                optimizer_coef : string, optional (default = "wls")
                    Optimizer used for estimating linear regression coefficients, if there are any
                    (for the GPBoost algorithm there are usually no).
                    Options: "gradient_descent" or "wls". Gradient descent steps are done simultaneously with
                    gradient descent steps for the covariance paramters. "wls" refers to doing coordinate descent
                    for the regression coefficients using weighted least squares
                maxit : integer, optional (default = 1000)
                    Maximal number of iterations for optimization algorithm
                delta_rel_conv : double, optional (default = 1e-6)
                    Convergence criterion: stop optimization if relative change in parameters is below this value
                init_coef : numpy array, optional (default = None)
                    Initial values for the regression coefficients (if there are any, can be None)
                init_cov_pars : numpy array, optional (default = None)
                    Initial values for covariance parameters of Gaussian process and random effects (can be None)
                lr_coef : double, optional (default = 0.01)
                    Learning rate for fixed effect regression coefficients
                lr_cov : double, optional (default = 0.01)
                    Learning rate for covariance parameters
                use_nesterov_acc : bool, optional (default = False)
                    If True Nesterov acceleration is used
                acc_rate_coef : double, optional (default = 0.5)
                    Acceleration rate for regression coefficients (if there are any) for Nesterov acceleration
                acc_rate_cov : double, optional (default = 0.5)
                    Acceleration rate for covariance parameters for Nesterov acceleration
                momentum_offset : integer, optional (default = 2)
                    Number of iterations for which no mometum is applied in the beginning
                trace : bool, optional (default = False)
                    If True, the value of the gradient is printed for some iterations. Useful for finding good learning rates.
        """
        if not isinstance(y, np.ndarray):
            raise ValueError("y needs to be a numpy.ndarray")
        if len(y.shape) != 1:
            raise ValueError("y needs to be a vector / one-dimensional numpy.ndarray ")
        if y.shape[0] != self.num_data:
            raise ValueError("Incorrect number of data points in y")
        y_c = y.astype(np.dtype(np.float64))
        y_c = y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        X_c = ctypes.c_void_p()
        if X is not None:
            if not isinstance(X, np.ndarray):
                raise ValueError("X needs to be a numpy.ndarray")
            if len(X.shape) > 2:
                raise ValueError("X needs to be either a vector or a two-dimensional matrix (array)")
            if len(X.shape) == 1:
                X = X.reshape((len(X), 1))
            if X.shape[0] != self.num_data:
                raise ValueError("Incorrect number of data points in X")
            self.has_covariates = True
            self.num_coef = X.shape[1]
            X_c = X.astype(np.float64)
            X_c, _, _ = c_float_array(X_c.flatten(order='F'))
            self.coef_names = []
            for ii in range(self.num_coef):
                if X.dtype.names is None:
                    self.coef_names.append("Covariate_" + str(ii + 1))
                else:
                    self.coef_names.append(X.dtype.names[ii])
        else:
            self.has_covariates = False
        self.std_dev = std_dev
        # Set parameters for optimizer
        if params is not None:
            self.set_optim_params(params)
            if X is not None:
                self.set_optim_coef_params(params)
        # Do optimization
        if X is None:
            _safe_call(_LIB.GPB_OptimCovPar(
                self.handle,
                y_c,
                ctypes.c_bool(self.std_dev)))
        else:
            _safe_call(_LIB.GPB_OptimLinRegrCoefCovPar(
                self.handle,
                y_c,
                X_c,
                ctypes.c_int(self.num_coef),
                ctypes.c_bool(self.std_dev)))

        num_it = ctypes.c_int64(0)
        _safe_call(_LIB.GPB_GetNumIt(
            self.handle,
            ctypes.byref(num_it)))
        print("Number of iterations until convergence: " + str(num_it.value))

    def set_optim_params(self, params):
        """Set parameters for estimation of the covariance paramters.

          Parameters
          ----------
          params : dict
            Parameters for fitting / optimization:
                optimizer_cov : string, optional (default = "fisher_scoring")
                    Optimizer used for estimating covariance parameters.
                    Options: "gradient_descent" or "fisher_scoring"
                maxit : integer, optional (default = 1000)
                    Maximal number of iterations for optimization algorithm
                delta_rel_conv : double, optional (default = 1e-6)
                    Convergence criterion: stop optimization if relative change in parameters is below this value
                init_cov_pars : numpy array, optional (default = None)
                    Initial values for covariance parameters of Gaussian process and random effects (can be None)
                lr_cov : double, optional (default = 0.01)
                    Learning rate for covariance parameters
                use_nesterov_acc : bool, optional (default = False)
                    If True Nesterov acceleration is used
                acc_rate_cov : double, optional (default = 0.5)
                    Acceleration rate for coefficients for Nesterov acceleration
                momentum_offset : integer, optional (default = 2)
                    Number of iterations for which no mometum is applied in the beginning
                trace : bool, optional (default = False)
                    If True, the value of the gradient is printed for some iterations. Useful for finding good learning rates.
        """
        if not isinstance(params, dict):
            raise ValueError("params needs to be a dict")
        if self.handle is None:
            raise ValueError("Gaussian process model has not been initialized")
        for param in params:
            if param == "init_cov_pars":
                if params[param] is not None:
                    if not isinstance(params[param], np.ndarray):
                        raise ValueError("params['init_cov_pars'] needs to be a numpy.ndarray")
                    if len(self.params[param].shape) != 1:
                        raise ValueError("params['init_cov_pars'] needs to be a vector / one-dimensional numpy.ndarray")
                    if self.params[param].shape[0] != self.num_cov_pars:
                        raise ValueError("params['init_cov_pars'] does not contain the correct number of parameters")
                    params[param] = params[param].astype(np.float64)
            if param in self.params:
                self.params[param] = params[param]
            else:
                raise ValueError("Unknown parameter: %s" % param)
        init_cov_pars_c = ctypes.c_void_p()
        if self.params["init_cov_pars"] is not None:
            init_cov_pars_c = self.params["init_cov_pars"].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _safe_call(_LIB.GPB_SetOptimConfig(
            self.handle,
            init_cov_pars_c,
            ctypes.c_double(self.params["lr_cov"]),
            ctypes.c_double(self.params["acc_rate_cov"]),
            ctypes.c_int(self.params["maxit"]),
            ctypes.c_double(self.params["delta_rel_conv"]),
            ctypes.c_bool(self.params["use_nesterov_acc"]),
            ctypes.c_int(self.params["nesterov_schedule_version"]),
            ctypes.c_bool(self.params["trace"]),
            c_str(self.params["optimizer_cov"]),
            ctypes.c_int(self.params["momentum_offset"])))

    def set_optim_coef_params(self, params):
        """Set parameters for estimation of the covariance paramters.

          Parameters
          ----------
          params : dict
            Parameters for fitting / optimization:
                optimizer_coef : string, optional (default = "wls")
                    Optimizer used for estimating regression coefficients.
                    Options: "gradient_descent" or "wls". Gradient descent steps are done simultaneously with
                    gradient descent steps for the covariance paramters. "wls" refers to doing coordinate descent
                    for the regression coefficients using weighted least squares
                init_coef : numpy array, optional (default = None)
                    Initial values for the regression coefficients (can be None)
                lr_coef : double, optional (default = 0.01)
                    Learning rate for fixed effect regression coefficients
                acc_rate_coef : double, optional (default = 0.5)
                    Acceleration rate for covariance parameters for Nesterov acceleration
        """
        if not isinstance(params, dict):
            raise ValueError("params needs to be a dict")
        if self.handle is None:
            raise ValueError("Gaussian process model has not been initialized")
        for param in params:
            if param == "init_coef":
                if not isinstance(params[param], np.ndarray):
                    raise ValueError("params['init_coef'] needs to be a numpy.ndarray")
                if len(self.params[param].shape) != 1:
                    raise ValueError("params['init_coef'] needs to be a vector / one-dimensional numpy.ndarray")
                if self.num_coef is None:
                    self.num_coef = self.params[param].shape[0]
                if self.params[param].shape[0] != self.num_coef:
                    raise ValueError("params['init_coef'] does not contain the correct number of parameters")
                params[param] = params[param].astype(np.float64)
            if param in self.params:
                self.params[param] = params[param]
            else:
                raise ValueError("Unknown parameter: %s" % param)
        if self.params["init_coef"] is None:
            init_coef_c = ctypes.c_void_p()
        else:
            init_coef_c = self.params["init_coef"].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _safe_call(_LIB.GPB_SetOptimCoefConfig(
            self.handle,
            ctypes.c_int(self.num_coef),
            init_coef_c,
            ctypes.c_double(self.params["lr_coef"]),
            ctypes.c_double(self.params["acc_rate_coef"]),
            c_str(self.params["optimizer_coef"])))

    def get_cov_pars(self):
        """Get (estimated) covariance parameters.

        Returns
        -------
        result : numpy array
            (estimated) covariance parameters and standard deciations if (if std_dev=True as set in 'fit')
        """
        if self.std_dev:
            optim_pars = np.zeros(2*self.num_cov_pars, dtype=np.float64)
        else:
            optim_pars = np.zeros(self.num_cov_pars, dtype=np.float64)

        _safe_call(_LIB.GPB_GetCovPar(
            self.handle,
            optim_pars.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_bool(self.std_dev)))
        if self.std_dev:
            cov_pars = np.row_stack((optim_pars[0:self.num_cov_pars],
                                     optim_pars[self.num_cov_pars:(2*self.num_cov_pars)]))
        else:
            cov_pars = optim_pars[0:self.num_cov_pars]
        return cov_pars

    def get_coef(self):
        """Get (estimated) linear regression coefficients.

        Returns
        -------
        result : numpy array
            (estimated) linear regression coefficients and standard deciations if (if std_dev=True as set in 'fit')
        """
        if self.num_coef is None:
            raise ValueError("'fit' has not been called")
        if self.std_dev:
            optim_pars = np.zeros(2*self.num_coef, dtype=np.float64)
        else:
            optim_pars = np.zeros(self.num_coef, dtype=np.float64)

        _safe_call(_LIB.GPB_GetCoef(
            self.handle,
            optim_pars.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_bool(self.std_dev)))
        if self.std_dev:
            coef = np.row_stack((optim_pars[0:self.num_coef],
                                     optim_pars[self.num_coef:(2*self.num_coef)]))
        else:
            coef = optim_pars[0:self.num_coef]
        return coef

    def summary(self):
        """Print summary of fitted model parameters.
        """
        cov_pars = self.get_cov_pars()
        message = "Covariance parameters "
        if self.std_dev:
            message = message + "and standard deviations (second row) "
        message = message + "in the following order:"
        print(message)
        print(self.cov_par_names)
        print(cov_pars)
        if self.has_covariates:
            coef = self.get_coef()
            message = "Linear regression coefficients "
            if self.std_dev:
                message = message + "and standard deviations (second row) "
            message = message + "in the following order:"
            print(message)
            print(self.coef_names)
            print(coef)

    def predict(self,
                y=None,
                group_data_pred=None,
                group_rand_coef_data_pred=None,
                gp_coords_pred=None,
                gp_rand_coef_data_pred=None,
                vecchia_pred_type=None,
                num_neighbors_pred=None,
                cluster_ids_pred=None,
                predict_cov_mat=False,
                cov_pars=None,
                X_pred=None,
                use_saved_data=False):
        """Make predictions for a GPModel.

        Parameters
        ----------
            y : numpy array or None, optional (default=None)
                Observed response variable data (can be None, e.g. when the model has been estimated already and
                the same data is used for making predictions)
            group_data_pred : numpy array with numeric or string data or None, optional (default=None)
                Labels of group levels for grouped random effects
            group_rand_coef_data_pred : numpy array with numeric data or None, optional (default=None)
                Covariate data for grouped random coefficients
            gp_coords_pred : numpy array with numeric data or None, optional (default=None)
                Coordinates (features) for Gaussian process
            gp_rand_coef_data_pred : numpy array with numeric data or None, optional (default=None)
                Covariate data for Gaussian process random coefficients
            vecchia_pred_type : string, optional (default="order_obs_first_cond_obs_only")
                Type of Vecchia approximation used for making predictions.
                "order_obs_first_cond_obs_only" = observed data is ordered first and the neighbors are only observed
                points, "order_obs_first_cond_all" = observed data is ordered first and the neighbors are selected
                among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for
                making predictions, "latent_order_obs_first_cond_obs_only" = Vecchia approximation for the latent
                process and observed data is ordered first and neighbors are only observed points,
                "latent_order_obs_first_cond_all" = Vecchia approximation for the latent process and observed data is
                ordered first and neighbors are selected among all points
            num_neighbors_pred : integer or None, optional (default=None)
                Number of neighbors for the Vecchia approximation for making predictions
            cluster_ids_pred : one dimensional numpy array (vector) with integer data or None, optional (default=None)
                IDs / labels indicating independent realizations of random effects / Gaussian processes
                (same values = same process realization)
            predict_cov_mat : bool (default=False)
                If True, the (posterior / conditional) predictive covariance is calculated in addition to the
                (posterior / conditional) predictive mean
            cov_pars : numpy array or None, optional (default = None)
                A vector containing covariance parameters (used if the GPModel has not been trained or if predictions
                should be made for other parameters than the estimated ones)
            X_pred : numpy array with numeric data or None, optional (default=None)
                Covariate data for fixed effects ( = linear regression term)
            use_saved_data : bool (default=False)
                If True, predictions are done using priorly set data via the function 'set_prediction_data'
                (this option is not used by users directly)

        Returns
        -------
        result : a dict with two entries both having numpy arrays as values
            The first entry of the dict result['mu'] is the predicted mean and the second entry result['cov'] is the
            the predicted covariance matrix (=None if 'predict_cov_mat=False')
        """
        if vecchia_pred_type is not None:
            if (vecchia_pred_type not in self._VECCHIA_PRED_TYPES):
                raise ValueError("vecchia_pred_type '{0:s}' not supported. ".format(vecchia_pred_type))
            self.vecchia_pred_type = vecchia_pred_type
        if num_neighbors_pred is not None:
            self.num_neighbors_pred = num_neighbors_pred
        y_c = ctypes.c_void_p()
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise ValueError("y needs to be a numpy.ndarray")
            if len(y.shape) != 1:
                raise ValueError("y needs to be a vector / one-dimensional numpy.ndarray ")
            if y.shape[0] != self.num_data:
                raise ValueError("Incorrect number of data points in y")
            y_c = y.astype(np.dtype(np.float64))
            y_c = y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        cov_pars_c = ctypes.c_void_p()
        if cov_pars is not None:
            if not isinstance(cov_pars, np.ndarray):
                raise ValueError("cov_pars needs to be a numpy.ndarray")
            if len(cov_pars.shape) != 1:
                raise ValueError("cov_pars needs to be a vector / one-dimensional numpy.ndarray")
            if cov_pars.shape[0] != self.num_cov_pars:
                raise ValueError("cov_pars does not contain the correct number of parameters")
            cov_pars_c = cov_pars.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        group_data_pred_c = ctypes.c_void_p()
        group_rand_coef_data_pred_c = ctypes.c_void_p()
        gp_coords_pred_c = ctypes.c_void_p()
        gp_rand_coef_data_pred_c = ctypes.c_void_p()
        cluster_ids_pred_c = ctypes.c_void_p()
        X_pred_c = ctypes.c_void_p()
        num_data_pred = self.num_data_pred
        if not use_saved_data:
            # Set data for grouped random effects
            if self.num_group_re > 0:
                if group_data_pred is None:
                    raise ValueError("group_data_pred not provided")
                if not isinstance(group_data_pred, np.ndarray):
                    raise ValueError("group_data_pred needs to be a numpy.ndarray")
                if len(group_data_pred.shape) > 2:
                    raise ValueError("group_data_pred needs to be either a vector or a two-dimensional matrix (array)")
                if len(group_data_pred.shape) == 1:
                    group_data_pred = group_data_pred.reshape((len(group_data_pred), 1))
                if group_data_pred.shape[1] != self.num_group_re:
                    raise ValueError("Number of grouped random effects in group_data_pred is not correct")
                num_data_pred = group_data_pred.shape[0]
                group_data_pred_c = group_data_pred.astype(np.dtype(str))
                group_data_pred_c = group_data_pred_c.flatten(order='F')
                group_data_pred_c = string_array_c_str(group_data_pred_c)
                # Set data for grouped random coefficients
                if self.num_group_rand_coef > 0:
                    if group_rand_coef_data_pred is None:
                        raise ValueError("group_rand_coef_data_pred not provided")
                    if not isinstance(group_rand_coef_data_pred, np.ndarray):
                        raise ValueError("group_rand_coef_data_pred needs to be a numpy.ndarray")
                    if len(group_rand_coef_data_pred.shape) > 2:
                        raise ValueError(
                            "group_rand_coef_data_pred needs to be either a vector or a two-dimensional matrix (array)")
                    if len(group_rand_coef_data_pred.shape) == 1:
                        group_rand_coef_data_pred = group_rand_coef_data_pred.reshape(
                            (len(group_rand_coef_data_pred), 1))
                    if group_rand_coef_data_pred.shape[0] != num_data_pred:
                        raise ValueError("Incorrect number of data points in group_rand_coef_data_pred")
                    if group_rand_coef_data_pred.shape[1] != self.num_group_rand_coef:
                        raise ValueError("Incorrect number of covariates in group_rand_coef_data_pred")
                    group_rand_coef_data_pred_c = group_rand_coef_data_pred.astype(np.dtype(np.float64))
                    group_rand_coef_data_c, _, _ = c_float_array(group_rand_coef_data_pred_c.flatten(order='F'))
            # Set data for Gaussian process
            if self.num_gp > 0:
                if gp_coords_pred is None:
                    raise ValueError("gp_coords_pred not provided")
                if not isinstance(gp_coords_pred, np.ndarray):
                    raise ValueError("gp_coords_pred needs to be a numpy.ndarray")
                if len(gp_coords_pred.shape) > 2:
                    raise ValueError("gp_coords_pred needs to be either a vector or a two-dimensional matrix (array)")
                if len(gp_coords_pred.shape) == 1:
                    gp_coords_pred = gp_coords_pred.reshape((len(gp_coords_pred), 1))
                if num_data_pred is None:
                    num_data_pred = gp_coords_pred.shape[0]
                else:
                    if gp_coords_pred.shape[0] != num_data_pred:
                        raise ValueError("Incorrect number of data points in gp_coords_pred")
                if gp_coords_pred.shape[1] != self.dim_coords:
                    raise ValueError("Incorrect dimension / number of coordinates (=features) in gp_coords_pred")
                gp_coords_pred_c = gp_coords_pred.astype(np.float64)
                gp_coords_pred_c, _, _ = c_float_array(gp_coords_pred_c.flatten(order='F'))
                # Set data for GP random coefficients
                if self.num_gp_rand_coef > 0:
                    if gp_rand_coef_data_pred is None:
                        raise ValueError("gp_rand_coef_data_pred not provided")
                    if not isinstance(gp_rand_coef_data_pred, np.ndarray):
                        raise ValueError("gp_rand_coef_data_pred needs to be a numpy.ndarray")
                    if len(gp_rand_coef_data_pred.shape) > 2:
                        raise ValueError(
                            "gp_rand_coef_data_pred needs to be either a vector or a two-dimensional matrix (array)")
                    if len(gp_rand_coef_data_pred.shape) == 1:
                        gp_rand_coef_data_pred = gp_rand_coef_data_pred.reshape((len(gp_rand_coef_data_pred), 1))
                    if gp_rand_coef_data_pred.shape[0] != num_data_pred:
                        raise ValueError("Incorrect number of data points in gp_rand_coef_data_pred")
                    if gp_rand_coef_data_pred.shape[1] != self.num_gp_rand_coef:
                        raise ValueError("Incorrect number of covariates in gp_rand_coef_data_pred")
                    gp_rand_coef_data_pred_c =  gp_rand_coef_data_pred.astype(np.dtype(np.float64))
                    gp_rand_coef_data_pred_c, _, _ = c_float_array(gp_rand_coef_data_pred_c.flatten(order='F'))
            # Set IDs for independent processes (cluster_ids)
            if cluster_ids_pred is not None:
                if not isinstance(cluster_ids_pred, np.ndarray):
                    raise ValueError("cluster_ids_pred needs to be a numpy.ndarray")
                if len(cluster_ids_pred.shape) != 1:
                    raise ValueError("cluster_ids_pred needs to be a vector / one-dimensional numpy.ndarray ")
                if cluster_ids_pred.shape[0] != num_data_pred:
                    raise ValueError("Incorrect number of data points in cluster_ids_pred")
                cluster_ids_preds_c = cluster_ids_pred.astype(np.dtype(int))
                cluster_ids_preds_c = cluster_ids_preds_c.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_int))
            # Set data for linear fixed-effects
            if self.has_covariates > 0:
                if X_pred is None:
                    raise ValueError("X_pred not provided")
                if not isinstance(X_pred, np.ndarray):
                    raise ValueError("X_pred needs to be a numpy.ndarray")
                if len(X_pred.shape) > 2:
                    raise ValueError("X_pred needs to be either a vector or a two-dimensional matrix (array)")
                if len(X_pred.shape) == 1:
                    X_pred = X_pred.reshape((len(X_pred), 1))
                if X_pred.shape[0] != num_data_pred:
                    raise ValueError("Incorrect number of data points in X_pred")
                if X_pred.shape[0] != self.num_coef:
                    raise ValueError("Incorrect number of covariates in X_pred")
                X_pred_c = X_pred.astype(np.float64)
                X_pred_c, _, _ = c_float_array(X_pred_c.flatten(order='F'))
        else:
            if not self.prediction_data_is_set:
                raise ValueError("No data has been set for making predictions. Call set_prediction_data first")

        if predict_cov_mat:
            preds = np.zeros(num_data_pred * (1 + num_data_pred), dtype=np.float64)
        else:
            preds = np.zeros(num_data_pred, dtype=np.float64)

        _safe_call(_LIB.GPB_PredictREModel(
            self.handle,
            y_c,
            ctypes.c_int(num_data_pred),
            preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_bool(predict_cov_mat),
            cluster_ids_pred_c,
            group_data_pred_c,
            group_rand_coef_data_pred_c,
            gp_coords_pred_c,
            gp_rand_coef_data_pred_c,
            cov_pars_c,
            X_pred_c,
            ctypes.c_bool(use_saved_data),
            c_str(self.vecchia_pred_type),
            ctypes.c_int(self.num_neighbors_pred)))

        pred_mean = preds[0:num_data_pred]
        pred_cov_mat = None
        if predict_cov_mat:
            pred_cov_mat = preds[num_data_pred:(num_data_pred * (num_data_pred + 1))].reshape((num_data_pred,num_data_pred))
        return {"mu": pred_mean, "cov":pred_cov_mat}

    def set_prediction_data(self,
                group_data_pred=None,
                group_rand_coef_data_pred=None,
                gp_coords_pred=None,
                gp_rand_coef_data_pred=None,
                cluster_ids_pred=None,
                X_pred=None):
        """(Pre-)set data for making predictions (not directly used by users).

        Parameters
        ----------
            group_data_pred : numpy array with numeric or string data or None, optional (default=None)
                Labels of group levels for grouped random effects
            group_rand_coef_data_pred : numpy array with numeric data or None, optional (default=None)
                Covariate data for grouped random coefficients
            gp_coords_pred : numpy array with numeric data or None, optional (default=None)
                Coordinates (features) for Gaussian process
            gp_rand_coef_data_pred : numpy array with numeric data or None, optional (default=None)
                Covariate data for Gaussian process random coefficients
            cluster_ids_pred : one dimensional numpy array (vector) with integer data or None, optional (default=None)
                IDs / labels indicating independent realizations of random effects / Gaussian processes
                (same values = same process realization)
            X_pred : numpy array with numeric data or None, optional (default=None)
                Covariate data for fixed effects ( = linear regression term)
        """
        group_data_pred_c = ctypes.c_void_p()
        group_rand_coef_data_pred_c = ctypes.c_void_p()
        gp_coords_pred_c = ctypes.c_void_p()
        gp_rand_coef_data_pred_c = ctypes.c_void_p()
        cluster_ids_pred_c = ctypes.c_void_p()
        X_pred_c = ctypes.c_void_p()
        num_data_pred = None
        # Set data for grouped random effects
        if self.num_group_re > 0:
            if group_data_pred is None:
                raise ValueError("group_data_pred not provided")
            if not isinstance(group_data_pred, np.ndarray):
                raise ValueError("group_data_pred needs to be a numpy.ndarray")
            if len(group_data_pred.shape) > 2:
                raise ValueError("group_data_pred needs to be either a vector or a two-dimensional matrix (array)")
            if len(group_data_pred.shape) == 1:
                group_data_pred = group_data_pred.reshape((len(group_data_pred), 1))
            if group_data_pred.shape[1] != self.num_group_re:
                raise ValueError("Number of grouped random effects in group_data_pred is not correct")
            num_data_pred = group_data_pred.shape[0]
            group_data_pred_c = group_data_pred.astype(np.dtype(str))
            group_data_pred_c = group_data_pred_c.flatten(order='F')
            group_data_pred_c = string_array_c_str(group_data_pred_c)
            # Set data for grouped random coefficients
            if self.num_group_rand_coef > 0:
                if group_rand_coef_data_pred is None:
                    raise ValueError("group_rand_coef_data_pred not provided")
                if not isinstance(group_rand_coef_data_pred, np.ndarray):
                    raise ValueError("group_rand_coef_data_pred needs to be a numpy.ndarray")
                if len(group_rand_coef_data_pred.shape) > 2:
                    raise ValueError(
                        "group_rand_coef_data_pred needs to be either a vector or a two-dimensional matrix (array)")
                if len(group_rand_coef_data_pred.shape) == 1:
                    group_rand_coef_data_pred = group_rand_coef_data_pred.reshape(
                        (len(group_rand_coef_data_pred), 1))
                if group_rand_coef_data_pred.shape[0] != num_data_pred:
                    raise ValueError("Incorrect number of data points in group_rand_coef_data_pred")
                if group_rand_coef_data_pred.shape[1] != self.num_group_rand_coef:
                    raise ValueError("Incorrect number of covariates in group_rand_coef_data_pred")
                group_rand_coef_data_pred_c = group_rand_coef_data_pred.astype(np.dtype(np.float64))
                group_rand_coef_data_c, _, _ = c_float_array(group_rand_coef_data_pred_c.flatten(order='F'))
        # Set data for Gaussian process
        if self.num_gp > 0:
            if gp_coords_pred is None:
                raise ValueError("gp_coords_pred not provided")
            if not isinstance(gp_coords_pred, np.ndarray):
                raise ValueError("gp_coords_pred needs to be a numpy.ndarray")
            if len(gp_coords_pred.shape) > 2:
                raise ValueError("gp_coords_pred needs to be either a vector or a two-dimensional matrix (array)")
            if len(gp_coords_pred.shape) == 1:
                gp_coords_pred = gp_coords_pred.reshape((len(gp_coords_pred), 1))
            if num_data_pred is None:
                num_data_pred = gp_coords_pred.shape[0]
            else:
                if gp_coords_pred.shape[0] != num_data_pred:
                    raise ValueError("Incorrect number of data points in gp_coords_pred")
            if gp_coords_pred.shape[1] != self.dim_coords:
                raise ValueError("Incorrect dimension / number of coordinates (=features) in gp_coords_pred")
            gp_coords_pred_c = gp_coords_pred.astype(np.float64)
            gp_coords_pred_c, _, _ = c_float_array(gp_coords_pred_c.flatten(order='F'))
            # Set data for GP random coefficients
            if self.num_gp_rand_coef > 0:
                if gp_rand_coef_data_pred is None:
                    raise ValueError("gp_rand_coef_data_pred not provided")
                if not isinstance(gp_rand_coef_data_pred, np.ndarray):
                    raise ValueError("gp_rand_coef_data_pred needs to be a numpy.ndarray")
                if len(gp_rand_coef_data_pred.shape) > 2:
                    raise ValueError(
                        "gp_rand_coef_data_pred needs to be either a vector or a two-dimensional matrix (array)")
                if len(gp_rand_coef_data_pred.shape) == 1:
                    gp_rand_coef_data_pred = gp_rand_coef_data_pred.reshape((len(gp_rand_coef_data_pred), 1))
                if gp_rand_coef_data_pred.shape[0] != num_data_pred:
                    raise ValueError("Incorrect number of data points in gp_rand_coef_data_pred")
                if gp_rand_coef_data_pred.shape[1] != self.num_gp_rand_coef:
                    raise ValueError("Incorrect number of covariates in gp_rand_coef_data_pred")
                gp_rand_coef_data_pred_c =  gp_rand_coef_data_pred.astype(np.dtype(np.float64))
                gp_rand_coef_data_pred_c, _, _ = c_float_array(gp_rand_coef_data_pred_c.flatten(order='F'))
        # Set IDs for independent processes (cluster_ids)
        if cluster_ids_pred is not None:
            if not isinstance(cluster_ids_pred, np.ndarray):
                raise ValueError("cluster_ids_pred needs to be a numpy.ndarray")
            if len(cluster_ids_pred.shape) != 1:
                raise ValueError("cluster_ids_pred needs to be a vector / one-dimensional numpy.ndarray ")
            if cluster_ids_pred.shape[0] != num_data_pred:
                raise ValueError("Incorrect number of data points in cluster_ids_pred")
            cluster_ids_preds_c = cluster_ids_pred.astype(np.dtype(int))
            cluster_ids_preds_c = cluster_ids_preds_c.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int))
        # Set data for linear fixed-effects
        if self.has_covariates > 0:
            if X_pred is None:
                raise ValueError("X_pred not provided")
            if not isinstance(X_pred, np.ndarray):
                raise ValueError("X_pred needs to be a numpy.ndarray")
            if len(X_pred.shape) > 2:
                raise ValueError("X_pred needs to be either a vector or a two-dimensional matrix (array)")
            if len(X_pred.shape) == 1:
                X_pred = X_pred.reshape((len(X_pred), 1))
            if X_pred.shape[0] != num_data_pred:
                raise ValueError("Incorrect number of data points in X_pred")
            if X_pred.shape[0] != self.num_coef:
                raise ValueError("Incorrect number of covariates in X_pred")
            X_pred_c = X_pred.astype(np.float64)
            X_pred_c, _, _ = c_float_array(X_pred_c.flatten(order='F'))
        self.num_data_pred = num_data_pred
        self.prediction_data_is_set = True

        _safe_call(_LIB.GPB_SetPredictionData(
            self.handle,
            ctypes.c_int(num_data_pred),
            cluster_ids_pred_c,
            group_data_pred_c,
            group_rand_coef_data_pred_c,
            gp_coords_pred_c,
            gp_rand_coef_data_pred_c,
            X_pred_c))

   # def __copy__(self):
   #     return self.__deepcopy__(None)
   #
   # def __deepcopy__(self, _):
   #     model_str = self.model_to_string(num_iteration=-1)
   #     booster = Booster(model_str=model_str)
   #     return booster
