## [0.3.0] - 2023-09-06

### Added

- Added functions `final_grad()` and `final_grad_norm()` to `LBFGSSolver`
  and `LBFGSBSolver` to retrieve the final gradient information
  ([#12](https://github.com/yixuan/LBFGSpp/issues/12))

### Changed

- `LBFGS++` now requires C++11
- The line search classes now have a unified API for both `LBFGSSolver` and `LBFGSBSolver`
- The Moré-Thuente line search algorithm `LineSearchMoreThuente` now can also be used
  in the L-BFGS solver `LBFGSSolver`
- Improved the numerical stability of `LineSearchNocedalWright`
  ([#27](https://github.com/yixuan/LBFGSpp/issues/27))
- Removed the unused variable `dg_hi` in `LineSearchNocedalWright`
  ([#35](https://github.com/yixuan/LBFGSpp/issues/35))
- Fixed some compiler warnings regarding shadowed variables
  ([#36](https://github.com/yixuan/LBFGSpp/issues/36))



## [0.2.0] - 2022-05-20

### Added

- Added a CMake script for installation ([#24](https://github.com/yixuan/LBFGSpp/pull/24)),
  contributed by [@steinmig](https://github.com/steinmig)

### Changed

- The default line search method for `LBFGSSolver` has been changed from `LineSearchBacktracking`
  to `LineSearchNocedalWright`, per the suggestion of [@mpayrits](https://github.com/mpayrits)
  ([#25](https://github.com/yixuan/LBFGSpp/pull/25))
- Fixed a few critical issues ([#9](https://github.com/yixuan/LBFGSpp/issues/9),
  [#15](https://github.com/yixuan/LBFGSpp/issues/15),
  [#21](https://github.com/yixuan/LBFGSpp/issues/21)), with big thanks to
  [@mpayrits](https://github.com/mpayrits) ([#25](https://github.com/yixuan/LBFGSpp/pull/25))
- Fixed one inconsistency with Moré and Thuente (1994) in the `LineSearchMoreThuente`
  line search algorithm, pointed out by [@mpayrits](https://github.com/mpayrits)
  ([#23](https://github.com/yixuan/LBFGSpp/issues/23))
- The source code is now formatted using [Clang-Format](https://clang.llvm.org/docs/ClangFormat.html)



## [0.1.0] - 2021-08-19

### Added

- Initial Github release
