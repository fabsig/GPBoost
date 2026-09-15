# coding: utf-8
"""Tests of the number of threads that models use."""
import numpy as np
import pytest

import gpboost as gpb
from gpboost.basic import (_get_auto_num_threads, _get_max_num_threads, _thread_candidates,
                          _thread_selection)


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
    if _get_max_num_threads() < 2:
        pytest.skip("two numbers of threads are needed to compare them")
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


def test_smallest_number_of_threads_within_the_tolerance_is_selected():
    candidates = [4, 8, 16]
    # 4 threads are 2% slower than the fastest, so they are selected with a tolerance of 3%, also when
    # the automatically selected number of threads is 16 and is within the tolerance as well
    normalized = np.array([[1.02, 1.00, 1.01]])
    selection = _thread_selection(normalized, candidates, 0.03, 0.25)
    assert selection["num_threads"] == 4
    assert _thread_selection(normalized, candidates, 0.01, 0.25)["num_threads"] == 8
    assert _thread_selection(normalized, candidates, 0., 0.25)["num_threads"] == 8

    # A number of threads that is much slower for a single workload is not selected, even though the
    # geometric mean over the two workloads is within the tolerance
    normalized = np.array([[1.30, 1.00, 1.02], [0.80 * 1.30, 1.00, 1.00]])
    normalized = normalized / normalized.min(axis=1, keepdims=True)
    selection = _thread_selection(normalized, candidates, 0.03, 0.25)
    assert not selection["acceptable"][0]
    assert selection["num_threads"] > 4
    assert _thread_selection(normalized, candidates, 0.30, 1e6)["num_threads"] == 4

    # The geometric mean gives every workload the same weight, irrespective of its runtime
    normalized = np.array([[1.00, 2.00], [2.00, 1.00]])
    selection = _thread_selection(normalized, [1, 2], 0.03, 1e6)
    assert selection["aggregate"] == pytest.approx([np.sqrt(2.), np.sqrt(2.)])
    assert selection["num_threads"] == 1


def test_tune_num_threads_applies_the_number_of_threads_that_it_reports():
    if _get_max_num_threads() < 2:
        pytest.skip("two numbers of threads are needed to compare them")
    num_threads_omp = gpb.get_num_threads()
    try:
        gpb.set_default_num_threads(1)
        assert gpb.get_default_num_threads() == 1
        results = gpb.tune_num_threads(workloads="grouped_re", workload_size="small",
                                       num_threads_candidates=[1, 2], n_rep=2,
                                       set_default=False, verbose=False)
        # Without set_default nothing is changed, and the default before the benchmark is reported
        assert results["num_threads_before"] == 1
        assert not results["default_was_set"]
        assert gpb.get_default_num_threads() == 1
    finally:
        gpb.set_default_num_threads(-1)
        gpb.set_num_threads(num_threads_omp)


def test_all_benchmark_workloads_run():
    if _get_max_num_threads() < 2:
        pytest.skip("two numbers of threads are needed to compare them")
    num_threads_omp = gpb.get_num_threads()
    try:
        results = gpb.tune_num_threads(workloads="all", workload_size="small",
                                       num_threads_candidates=[1, 2], n_rep=2, verbose=False)
        assert sorted(results["timings"]["workload"].unique()) == sorted(
            ["grouped_re", "vecchia_non_gaussian", "crossed_re_iterative"])
        assert len(results["timings"]) == 6
        assert np.all(results["timings"]["median"].values > 0)
        assert np.all(np.isfinite(results["aggregate"]["relative_runtime"].values))
        # The small workloads never change the default of the session
        assert not results["default_was_set"]
        assert gpb.get_num_threads() == num_threads_omp
    finally:
        gpb.set_default_num_threads(-1)
        gpb.set_num_threads(num_threads_omp)


def test_time_budget_leaves_every_workload_at_least_two_repetitions():
    if _get_max_num_threads() < 2:
        pytest.skip("two numbers of threads are needed to compare them")
    num_threads_omp = gpb.get_num_threads()
    try:
        results = gpb.tune_num_threads(workloads="all", workload_size="small",
                                       num_threads_candidates=[1, 2], n_rep=5, max_time=1e-6,
                                       verbose=False)
        assert len(results["timings"]) == 6
        assert np.all(np.isfinite(results["timings"]["median"].values))
        # Two repetitions are guaranteed, so the noise of every workload can be estimated
        assert np.all(np.isfinite(results["timings"]["relative_mad"].values))
    finally:
        gpb.set_default_num_threads(-1)
        gpb.set_num_threads(num_threads_omp)


def test_workloads_are_validated():
    for workloads in (None, [1], 3, [], ["grouped_re", 2], {"grouped_re"}):
        with pytest.raises(ValueError):
            gpb.tune_num_threads(workloads=workloads)


def test_safeguard_that_no_number_of_threads_satisfies_is_reported():
    # The workloads disagree: neither number of threads is within 25% of the fastest one for both of
    # them, which is a legitimate outcome and not a reason to call both of them acceptable
    normalized = np.array([[1.00, 1.50], [1.50, 1.00]])
    selection = _thread_selection(normalized, [4, 16], 0.03, 0.25)
    assert list(selection["acceptable"]) == [False, False]
    assert selection["safeguard_was_relaxed"]
    # Both have the same worst workload here, so the smaller number of threads is selected
    assert selection["num_threads"] == 4

    # The number of threads whose slowest workload is the least slow is the compromise
    normalized = np.array([[1.00, 1.30, 1.60], [1.60, 1.30, 1.00]])
    selection = _thread_selection(normalized, [1, 2, 4], 0.03, 0.25)
    assert list(selection["acceptable"]) == [False, False, False]
    assert selection["safeguard_was_relaxed"]
    assert selection["num_threads"] == 2

    # Nothing is relaxed when the safeguard is satisfied
    normalized = np.array([[1.00, 1.10], [1.10, 1.00]])
    selection = _thread_selection(normalized, [4, 16], 0.03, 0.25)
    assert list(selection["acceptable"]) == [True, True]
    assert not selection["safeguard_was_relaxed"]


def test_single_number_of_threads_is_selected_and_applied():
    num_threads_max = _get_max_num_threads()
    if num_threads_max < 2:
        pytest.skip("two numbers of threads are needed to compare them")
    num_threads_omp = gpb.get_num_threads()
    num_threads_auto = _get_auto_num_threads()
    try:
        # A single number of threads cannot be compared with anything, but it is still the selected
        # one and has to be applied. The default before the benchmark has to differ from the one that
        # is selected, otherwise there is nothing to change
        num_threads_single = 1 if num_threads_auto == 2 else 2
        num_threads_old = 2 if num_threads_auto == 2 else 1
        gpb.set_default_num_threads(num_threads_old)
        results = gpb.tune_num_threads(workloads="grouped_re",
                                       num_threads_candidates=[num_threads_single], n_rep=1,
                                       verbose=False)
        assert results["num_threads"] == num_threads_single
        assert gpb.get_default_num_threads() == num_threads_single
        assert results["num_threads_before"] == num_threads_old
        assert results["default_was_set"]

        # The automatic number of threads as the only candidate removes an earlier default. The
        # earlier default has to be a number of threads that the automatic one is not
        num_threads_before_auto = 2 if num_threads_auto == 1 else 1
        if num_threads_before_auto <= num_threads_max:
            gpb.set_default_num_threads(num_threads_before_auto)
            results = gpb.tune_num_threads(workloads="grouped_re",
                                           num_threads_candidates=[num_threads_auto], n_rep=1,
                                           verbose=False)
            assert results["num_threads"] == num_threads_auto
            assert gpb.get_default_num_threads() == num_threads_auto
            assert results["default_was_set"]

        # A single number of threads is not applied when nothing should be set
        gpb.set_default_num_threads(num_threads_old)
        results = gpb.tune_num_threads(workloads="grouped_re",
                                       num_threads_candidates=[num_threads_single], n_rep=1,
                                       set_default=False, verbose=False)
        assert results["num_threads"] == num_threads_single
        assert gpb.get_default_num_threads() == num_threads_old
        assert not results["default_was_set"]
        # Nothing was measured, so there are no timings
        assert results["timings"] is None
        assert results["aggregate"] is None
    finally:
        gpb.set_default_num_threads(-1)
        gpb.set_num_threads(num_threads_omp)
