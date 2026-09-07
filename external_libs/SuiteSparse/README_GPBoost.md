# CHOLMOD bundled for GPBoost

This is the subset of [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse)
**v7.14.0** that GPBoost needs in order to use CHOLMOD without a separately
installed CHOLMOD, BLAS or LAPACK. The typedefs `chol_cholmod_sp_mat_t` and
`chol_cholmod_sp_mat_rm_t` in `include/GPBoost/type_defs.h` are built on it,
through Eigen's `CholmodSupport` module.

## What is included

| Directory                   | Contents                                                            |
| --------------------------- | ------------------------------------------------------------------- |
| `SuiteSparse_config/`       | the shared configuration and memory manager                          |
| `AMD/`, `COLAMD/`           | the fill-reducing orderings CHOLMOD uses                             |
| `CHOLMOD/Include/`          | the CHOLMOD headers                                                  |
| `CHOLMOD/Utility/`          | allocation, conversion and transposition of CHOLMOD objects          |
| `CHOLMOD/Check/`            | the consistency checks CHOLMOD asserts against                       |
| `CHOLMOD/Cholesky/`         | the symbolic analysis, the simplicial factorization and the solves   |
| `CHOLMOD/Supernodal/`       | the supernodal (BLAS based) factorization and solve                  |
| `EigenBLAS/`                | GPBoost's own BLAS/LAPACK kernels on top of Eigen, see below         |
| `Doc/`                      | the licenses of the bundled SuiteSparse modules                      |

## What is left out, and why

* Only the `int32` entry points (`cholmod_*`) are kept. The `int64` ones
  (`cholmod_l_*`, `amd_l_*`, `colamd_l`) are thin wrappers that compile the same
  sources with a different index type, and Eigen only calls the `int32` ones for
  the `int` storage index that GPBoost's sparse matrices use.
* `CHOLMOD/Partition` needs METIS, which is 14 MB and separately licensed;
  `-DNPARTITION` and `-DNCAMD` disable it and the CAMD interface, so CHOLMOD
  orders with AMD and COLAMD. `CAMD` and `CCOLAMD` are therefore not bundled.
* `CHOLMOD/MatrixOps` and `CHOLMOD/Modify` are disabled with `-DNMATRIXOPS` and
  `-DNMODIFY`; GPBoost does not update or downdate factorizations.
* `CHOLMOD/GPU`, `Demo`, `MATLAB`, `Tcov` and `Doc` are not needed.
* `-DNPRINT` keeps CHOLMOD from writing to stdout, which CRAN does not allow.

## BLAS and LAPACK

CHOLMOD's supernodal factorization is the fast one for the denser factors that
Gaussian process models produce, but it needs `gemm`, `gemv`, `syrk`/`herk`,
`trsm`, `trsv` and `potrf` from BLAS/LAPACK. Rather than make GPBoost depend on
an external BLAS, `EigenBLAS/gpb_eigen_blas.cpp` implements exactly those six
kernels on top of Eigen, whose own matrix product is vectorized and cache
blocked.

CHOLMOD reaches BLAS/LAPACK only through the `SUITESPARSE_BLAS_*` and
`SUITESPARSE_LAPACK_*` macros of `SuiteSparse_config.h`. Defining
`GPB_SUITESPARSE_INTERNAL_BLAS` redirects those macros to `gpb_blas_`-prefixed
names, so the bundled kernels can never collide with a real BLAS that the host
program links as well (R's `libRblas`, OpenBLAS, MKL, ...). This is the only
change made to the SuiteSparse sources; it is marked in `SuiteSparse_config.h`
under "GPBoost modification".

To link a real BLAS/LAPACK instead, configure with
`cmake -DUSE_EXTERNAL_BLAS_FOR_CHOLMOD=ON`, which leaves
`GPB_SUITESPARSE_INTERNAL_BLAS` undefined.

## Licenses

The bundled SuiteSparse modules are not all under the same license, see `Doc/`:

| Module                                              | License      |
| --------------------------------------------------- | ------------ |
| `SuiteSparse_config`, `AMD`, `COLAMD`               | BSD-3-Clause |
| `CHOLMOD/Utility`, `CHOLMOD/Check`, `CHOLMOD/Cholesky` | LGPL-2.1+ |
| `CHOLMOD/Supernodal`                                | GPL-2.0+     |

## Updating

Fetch the SuiteSparse release, then copy the directories listed above, dropping
every `cholmod_l_*.c`, `amd_l*.c` and `colamd_l.c`, and reapply the
`GPB_SUITESPARSE_INTERNAL_BLAS` block in `SuiteSparse_config.h`. The file lists
in `CMakeLists.txt` are globs and need no update, but the explicit `OBJECTS`
lists in `R-package/src/Makevars.in`, `Makevars.win` and `Makevars.win.in` do.
