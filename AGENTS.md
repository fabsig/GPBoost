# Instructions for coding agents

These instructions apply to code that is written for or with GPBoost, both inside this repository and
in analyses, examples, simulations, and benchmarks that use it.

## Thread settings

When generating or modifying GPBoost examples, analyses, simulations, or benchmarks, preserve the
user's explicit thread settings. Otherwise, leave thread parameters unspecified so that GPBoost
selects its default.

Do not insert `num_parallel_threads = 1` (the `GPModel` argument), `num_threads = 1` (the boosting
parameter), or equivalent environment-variable restrictions such as `OMP_NUM_THREADS=1` merely as a
generic precaution. Single-thread execution is not required for correctness or reproducibility, and
it can substantially increase the runtime.

If an explicit resource limit, concurrent outer workers, a specific test, or a demonstrated issue
requires a restriction, scope it to that purpose and explain it in a nearby comment. Restrictions
that are only needed for an agent's own temporary validation must not silently become defaults in
delivered code. Before finishing, check the result for unintended thread restrictions.

By default, a `GPModel` uses the number of physical performance cores, see `NumPerformanceCores()` in
`src/GPBoost/cpu_topology.cpp`: the fastest cores of the CPU, without their hyperthreads, extended by
the next fastest ones if that would leave a single thread, and limited by the CPU bandwidth quota of
the control group on Linux. The boosting parameter `num_threads` is independent of this: `0`, its
default, means the default number of threads of OpenMP.

Everything in the library that changes the number of threads of the process has to go through
`GPBoost::SetNumParallelThreads()` in `include/GPBoost/utils.h` and not call `omp_set_num_threads()`
directly. The default is determined once, on first use, from the number of threads that OpenMP
reports; a call that lowers that number first would otherwise make its own value the default for the
rest of the session.

## Generated files that must not be edited by hand

* `gpboost_r/` is a build copy of `include/` and `src/` that is created when the R package is built.
  It is excluded from git, but not from searching, so a search for anything in the library returns
  every hit twice. Edit the originals and exclude `gpboost_r/` from searches.
* `docs/Parameters.rst` and `src/LightGBM/io/config_auto.cpp` are generated from the `// desc =`
  comments in `include/LightGBM/config.h` by `helpers/parameter_generator.py`. Change the comments in
  `config.h` and run the generator; an edit of the generated files is lost at the next run.
* The files in `R-package/man/` are generated from the roxygen comments of the corresponding files in
  `R-package/R/`. They are part of the repository, so a change of a roxygen comment has to be carried
  over to the `.Rd` file, either by regenerating it or by applying the same change by hand.

## Adding a source file

A new `.cpp` file in `src/GPBoost/` has to be added to the `OBJECTS` lists of `Makevars.in`,
`Makevars.win.in` and `Makevars.win` in `R-package/src/`. CMake globs the directory, so a missing
entry does not show up when building with CMake or when building the Python package: only the build
of the R package fails, and it fails long after the change has been made.

## Tests

* Never pin the number of threads in a test to make it pass. Results that depend on the order of
  summation vary slightly with the number of threads, and the tolerances have to accommodate that
  instead. See the comments in `test_GPModel_ar1_multifidelity.R` and
  `test_GPModel_non_Gaussian_data.R`, which document the measured deviations.
* The `num_parallel_threads` argument of a model is scoped: an operation of that model sets the
  number of threads and restores the previous one when it is finished, see `ParallelThreadsScope` in
  `include/GPBoost/utils.h`, so it does not affect other models. `gpb.set.num.threads()` in R and
  `set_num_threads()` in Python, in contrast, change the number of threads of the entire process. A
  test that calls them has to restore the previous number with `on.exit()`, so that the test files
  which run afterwards are not affected, also when an expectation fails.
* Some tests are known to fail on an unmodified tree, depending on the compiler and the build
  options. Establish a baseline on `master` before attributing a failure to your change.
* The expected values in the tests of the Vecchia and the iterative (conjugate gradient) methods are
  sensitive to the build. Do not adapt them to make a test pass unless the change is understood; a
  deviation is more often a real difference than a stale expected value.
* A full build of the R package and a full run of the test suite both take a long time. Run the test
  files that are affected by a change. Note that installing the R package fails while another R
  session still has `lib_gpboost.dll` loaded, see the comments in `build_r.R`.

## Files, encodings and editing

The sources use CRLF line endings, and the encoding is not uniform: `include/GPBoost/likelihoods.h`
has a UTF-8 byte order mark, while neighbouring headers such as `include/GPBoost/utils.h` do not.
Preserve what a file has instead of assuming a convention. In a Git Bash shell on Windows, `sed -i`
strips and rewrites the carriage returns and `perl -e` loses backslashes even inside single quotes,
which corrupts these files silently; a short Python script that reads and writes bytes is reliable.

## Vendored code

`external_libs/` contains vendored third-party libraries and is not modified, with the exception of
the deliberate local changes to the bundled SuiteSparse, in particular the Eigen-backed BLAS in
`external_libs/SuiteSparse/EigenBLAS/`. `include/LightGBM/` and `src/LightGBM/` are derived from
LightGBM; changes there diverge from upstream, so prefer `include/GPBoost/` and `src/GPBoost/` for
code that belongs to GPBoost.

## Comments

Comments explain what the code does and why, not what was wrong before. The reason for a fix belongs
in the test that covers it and in the commit message, not in a comment next to the fixed line. Do not
use capitalised words for emphasis; acronyms stay capitalised.
