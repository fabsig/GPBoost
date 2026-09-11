#!/bin/sh
# Build and run 'helpers/cg_convergence_check.cpp', which checks the conjugate gradient stopping
# rules on cases that cannot be constructed through the R or Python interface.
#
# Usage:
#   sh helpers/run_cg_convergence_check.sh
#   CXX=/path/to/g++ sh helpers/run_cg_convergence_check.sh
#
# Exits with a non-zero status if any check fails.
set -e
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------------------------
# Find a suitable compiler. On Windows the 'g++' that is first on PATH can be an old, unrelated
# compiler, so prefer the Rtools one that the R package is actually built with
# ---------------------------------------------------------------------------------------------
if [ -z "${CXX}" ]; then
  for CANDIDATE in \
      /c/rtools45/x86_64-w64-mingw32.static.posix/bin/g++.exe \
      /c/rtools44/x86_64-w64-mingw32.static.posix/bin/g++.exe \
      /c/rtools43/x86_64-w64-mingw32.static.posix/bin/g++.exe ; do
    if [ -x "${CANDIDATE}" ]; then
      CXX="${CANDIDATE}"
      break
    fi
  done
fi
if [ -z "${CXX}" ]; then
  CXX=$(command -v g++ || command -v clang++ || true)
fi
if [ -z "${CXX}" ]; then
  echo "ERROR: no C++ compiler found. Set the CXX environment variable." >&2
  exit 2
fi
# Make sure the assembler / linker matching the compiler are found before any other toolchain
PATH="$(dirname "${CXX}"):${PATH}"
export PATH
echo "Compiler: ${CXX}"

OUT_DIR=$(mktemp -d 2>/dev/null || echo "./temp")
mkdir -p "${OUT_DIR}"
EXE="${OUT_DIR}/cg_convergence_check.exe"
# also remove the build products when a check fails, since 'set -e' skips the cleanup below
trap 'rm -f "${EXE}"' EXIT

# 'CG_utils' is required to stay C++11-compatible, so compile with that standard
"${CXX}" -std=c++11 -O2 -fopenmp -Wall -Wextra -Wno-unknown-pragmas -Wno-ignored-attributes \
  -DEIGEN_MPL2_ONLY \
  -I include \
  -I external_libs/eigen \
  -I external_libs/CSparse/Include \
  -I external_libs/OptimLib \
  -I external_libs/LBFGSpp/include \
  -I external_libs/SuiteSparse/CHOLMOD/Include \
  -I external_libs/SuiteSparse/SuiteSparse_config \
  helpers/cg_convergence_check.cpp src/GPBoost/CG_utils.cpp -o "${EXE}"

"${EXE}"
