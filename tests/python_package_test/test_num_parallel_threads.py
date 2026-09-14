# coding: utf-8
"""Tests of the number of threads that models use."""
import numpy as np
import pytest

import gpboost as gpb
from gpboost.basic import _get_auto_num_threads, _get_max_num_threads, _thread_candidates


def test_default_number_of_threads_can_be_set_and_reset():
    num_threads_auto = _get_auto_num_threads()
    num_threads_max = _get_max_num_threads()
    assert num_threads_auto >= 1
    # The automatically selected number of threads counts only the physical performance cores, the
    # maximum is the number of threads that OMP uses when the default is determined
    assert num_threads_max >= num_threads_auto

    num_threads_omp = gpb.get_num_threads()
    try:
        assert gpb.get_default_num_threads() == num_threads_auto
        gpb.set_default_num_threads(1)
        assert gpb.get_default_num_threads() == 1
        # The default of the session is limited by the largest number of threads
        gpb.set_default_num_threads(num_threads_max + 10)
        assert gpb.get_default_num_threads() == num_threads_max
        # A non-positive number uses the automatically selected number of threads again
        gpb.set_default_num_threads(-1)
        assert gpb.get_default_num_threads() == num_threads_auto
        with pytest.raises(ValueError):
            gpb.set_default_num_threads("two")

        # Setting the number of threads of the process does not change the default of the session:
        # the two are different things, the default is what models use when nothing else is requested
        gpb.set_num_threads(1)
        assert gpb.get_default_num_threads() == num_threads_auto
        # ... and a non-positive number of threads of the process means the default of the session
        gpb.set_default_num_threads(1)
        gpb.set_num_threads(-1)
        assert gpb.get_num_threads() == 1
    finally:
        # Both the default of the session and the number of threads of the process have to be
        # restored for the tests that run afterwards, also if an assertion above fails
        gpb.set_default_num_threads(-1)
        gpb.set_num_threads(num_threads_omp)


def test_numbers_of_threads_that_are_benchmarked_are_spread_out():
    assert _thread_candidates(16, 16) == [1, 2, 4, 8, 16]
    assert _thread_candidates(1, 1) == [1]
    assert _thread_candidates(6, 12) == [1, 2, 3, 4, 6, 8, 12]
    # A single thread, the automatically selected number of threads and the largest number of threads
    # are always benchmarked, and the number of candidates is limited
    candidates = _thread_candidates(64, 128)
    assert len(candidates) <= 7
    assert all(value in candidates for value in (1, 64, 128))
    assert candidates == sorted(candidates)


def test_tune_num_threads_validates_arguments_before_benchmarking():
    with pytest.raises(ValueError):
        gpb.tune_num_threads(n_rep=1.5)
    with pytest.raises(ValueError):
        gpb.tune_num_threads(tolerance=np.inf)
    with pytest.raises(ValueError):
        gpb.tune_num_threads(max_time=np.nan)
    with pytest.raises(ValueError):
        gpb.tune_num_threads(set_default=1)
    with pytest.raises(ValueError):
        gpb.tune_num_threads(num_threads_candidates=[1, 1.5])


def test_tune_num_threads_measures_without_changing_anything():
    num_threads_omp = gpb.get_num_threads()
    num_threads_default = gpb.get_default_num_threads()
    try:
        results = gpb.tune_num_threads(workloads="grouped_re", workload_size="small",
                                       num_threads_candidates=[1, 2], n_rep=2,
                                       set_default=False, verbose=False)
        assert len(results["timings"]) == 2
        assert np.all(results["timings"]["median"].values > 0)
        assert list(results["aggregate"]["num_threads"]) == [1, 2]
        # The runtimes are relative to the fastest number of threads of a workload, so the smallest
        # one is 1
        assert np.min(results["aggregate"]["relative_runtime"].values) == pytest.approx(1.)
        assert not results["default_was_set"]
        assert gpb.get_default_num_threads() == num_threads_default
        # The benchmark gives its number of threads to the models and does not change the process
        assert gpb.get_num_threads() == num_threads_omp
        with pytest.raises(ValueError):
            gpb.tune_num_threads(workloads="not_a_workload")
    finally:
        gpb.set_default_num_threads(-1)
        gpb.set_num_threads(num_threads_omp)
