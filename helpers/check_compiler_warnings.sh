#!/bin/sh
# Compile the GPBoost C++ sources with an extensive set of compiler warnings enabled.
#
# The GPBoost code is almost entirely header-based templates. Warnings inside a template are only
# emitted when the template is instantiated, which is why this script compiles the translation units
# in 'src/GPBoost/' (they instantiate 'REModelTemplate' / 'Likelihood' for all matrix types) instead
# of only doing a syntax check on the headers.
#
# Usage:
#   sh helpers/check_compiler_warnings.sh            # warnings for all files
#   sh helpers/check_compiler_warnings.sh --summary  # only the counts per warning type
#   CXX=/path/to/g++ sh helpers/check_compiler_warnings.sh   # use a specific compiler
#
# Exits with a non-zero status if any warning (that is not filtered out, see WARNING_FILTER below)
# is emitted, so that this can be used in a CI job.
#
# Note: on Windows, the 'g++' that is first on PATH can be an old compiler that is unrelated to the
# one R uses. This script therefore prefers the Rtools compiler, which is the one the R package is
# actually built with.

set -e
cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

SUMMARY_ONLY=0
if [ "$1" = "--summary" ]; then
  SUMMARY_ONLY=1
fi

# ---------------------------------------------------------------------------------------------
# Find a suitable compiler
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
CXX_BIN_DIR=$(dirname "${CXX}")
PATH="${CXX_BIN_DIR}:${PATH}"
export PATH

CXX_VERSION=$("${CXX}" --version 2>/dev/null | head -1)
echo "Compiler: ${CXX}"
echo "          ${CXX_VERSION}"

# GCC < 8 does not know several of the warning options used below
CXX_MAJOR=$("${CXX}" -dumpversion 2>/dev/null | cut -d. -f1)
case "${CXX_MAJOR}" in
  ''|*[!0-9]*) CXX_MAJOR=0 ;;
esac
if [ "${CXX_MAJOR}" -lt 8 ] 2>/dev/null; then
  echo "ERROR: '${CXX}' is version ${CXX_MAJOR}, which is too old for this check (need >= 8)." >&2
  echo "       On Windows, install Rtools and/or set CXX to the Rtools g++." >&2
  exit 2
fi

# ---------------------------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------------------------
INCLUDES="-I${REPO_ROOT}/include \
 -I${REPO_ROOT}/external_libs/eigen \
 -I${REPO_ROOT}/external_libs/CSparse/Include \
 -I${REPO_ROOT}/external_libs/LBFGSpp/include \
 -I${REPO_ROOT}/external_libs/OptimLib \
 -I${REPO_ROOT}/external_libs/fmt/include"

# 'LightGBM/utils/log.h' includes <R_ext/Error.h> when LGB_R_BUILD is defined
R_INCLUDE=$(R CMD config --cppflags 2>/dev/null | sed 's/^-I//' | tr -d '\r' || true)
if [ -z "${R_INCLUDE}" ] || [ ! -d "${R_INCLUDE}" ]; then
  R_HOME_DIR=$(R RHOME 2>/dev/null | tr -d '\r' || true)
  if [ -n "${R_HOME_DIR}" ] && [ -d "${R_HOME_DIR}/include" ]; then
    R_INCLUDE="${R_HOME_DIR}/include"
  fi
fi
if [ -n "${R_INCLUDE}" ] && [ -d "${R_INCLUDE}" ]; then
  INCLUDES="${INCLUDES} -I${R_INCLUDE}"
  DEFINES="-DEIGEN_MPL2_ONLY -DMM_PREFETCH=1 -DMM_MALLOC=1 -DUSE_SOCKET -DLGB_R_BUILD"
else
  echo "Note: no R include directory found, compiling without -DLGB_R_BUILD"
  DEFINES="-DEIGEN_MPL2_ONLY -DMM_PREFETCH=1 -DMM_MALLOC=1 -DUSE_SOCKET"
fi

# ---------------------------------------------------------------------------------------------
# Warning options
#
# -Wall -Wextra                              the usual baseline
# -Wduplicated-cond / -Wduplicated-branches  catch copy-paste errors in the many similar branches
# -Wlogical-op                               e.g. '&&' where '&' was meant, suspicious comparisons
# -Wnull-dereference                         potential null pointer dereferences
# -Wshadow=local                             a local variable hiding another local variable
# -Wcast-qual                                casting away const
#
# Optimization is enabled since warnings such as -Wmaybe-uninitialized are only produced by the
# optimizer, i.e. compiling with -O0 silently loses them
# ---------------------------------------------------------------------------------------------
WARNING_FLAGS="-Wall -Wextra \
 -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wnull-dereference \
 -Wshadow=local -Wcast-qual"

# Warnings that are noise for this code base and are therefore switched off:
# -Wignored-attributes : thousands of hits from Eigen's SIMD types used as template arguments
# -Wunknown-pragmas    : '#pragma warning(...)' for MSVC in type_defs.h
SUPPRESSED_FLAGS="-Wno-ignored-attributes -Wno-unknown-pragmas"

# Only warnings for GPBoost's own code are reported. Eigen, OptimLib, LBFGSpp, CSparse and LightGBM
# produce a fair number of warnings that cannot be fixed here, and that would otherwise hide the
# warnings that matter. Header paths are matched, so a warning in a GPBoost header that is triggered
# from a third-party file is still reported
OWN_CODE_PATTERN='(include|src)[\\/]GPBoost[\\/]'

COMPILE_FLAGS="-std=gnu++17 -O2 -fopenmp -fsyntax-only"

# ---------------------------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------------------------
SOURCES=$(ls "${REPO_ROOT}"/src/GPBoost/*.cpp 2>/dev/null)
if [ -z "${SOURCES}" ]; then
  echo "ERROR: no sources found in ${REPO_ROOT}/src/GPBoost/" >&2
  exit 2
fi

# The full compiler output is kept in this file so that it can still be inspected after the run
# (in particular when there are many warnings and they scroll out of the terminal)
LOG_FILE="${REPO_ROOT}/compiler_warnings.log"
: > "${LOG_FILE}"

for SRC in ${SOURCES}; do
  echo "Checking $(basename "${SRC}") ..."
  # '|| true': a failing compilation is reported below, it should not abort the loop
  "${CXX}" ${COMPILE_FLAGS} ${DEFINES} ${INCLUDES} ${WARNING_FLAGS} ${SUPPRESSED_FLAGS} \
    "${SRC}" >> "${LOG_FILE}" 2>&1 || true
done

# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
if grep -qE '(^|[^-])error:|fatal error:' "${LOG_FILE}"; then
  echo ""
  echo "ERROR: compilation failed:"
  grep -E '(^|[^-])error:|fatal error:' "${LOG_FILE}" | head -20
  echo ""
  echo "Full compiler output: ${LOG_FILE}"
  exit 2
fi

FILTERED_LOG="${REPO_ROOT}/compiler_warnings_gpboost.log"
# Keep only warning lines that refer to a file in include/GPBoost or src/GPBoost.
# 'sort -u': a warning in a header is repeated for every translation unit that includes it
grep 'warning:' "${LOG_FILE}" | grep -E "${OWN_CODE_PATTERN}" | sort -u > "${FILTERED_LOG}" || true
NUM_WARNINGS=$(grep -c 'warning:' "${FILTERED_LOG}" || true)
NUM_TOTAL=$(grep -c 'warning:' "${LOG_FILE}" || true)
NUM_THIRD_PARTY=$((NUM_TOTAL - NUM_WARNINGS))

echo ""
echo "=============================================================="
echo "(${NUM_THIRD_PARTY} warning(s) in third-party code were ignored)"
if [ "${NUM_WARNINGS}" -eq 0 ]; then
  echo "No compiler warnings in GPBoost code."
  echo "=============================================================="
  echo "Full compiler output (incl. third-party): ${LOG_FILE}"
  rm -f "${FILTERED_LOG}"
  exit 0
fi

echo "${NUM_WARNINGS} compiler warning(s) in GPBoost code, by type:"
grep -o '\[-W[a-z0-9=+-]*\]' "${FILTERED_LOG}" | sort | uniq -c | sort -rn
echo "=============================================================="
if [ "${SUMMARY_ONLY}" -eq 0 ]; then
  echo ""
  cat "${FILTERED_LOG}"
fi
echo ""
echo "GPBoost warnings only              : ${FILTERED_LOG}"
echo "Full compiler output (incl. third-party): ${LOG_FILE}"
exit 1
