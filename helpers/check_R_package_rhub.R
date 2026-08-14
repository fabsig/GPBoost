# Checking the R package with the AddressSanitizer (ASan) and the UndefinedBehaviorSanitizer
# (UBSan) in the R-hub Docker containers.
#
# Note: this file only contains documentation and shell commands (all commented out), it is not
# meant to be sourced. The containers used below are the same ones that R-hub and CRAN use, i.e.
# they contain an R that has been built with the corresponding sanitizer.


##############################################################################################
## Installing Docker
##############################################################################################
#### Step 1 - Install WSL2 ####
# Open PowerShell as Administrator (right-click -> "Run as administrator"), then:
#
#   wsl --install
#
# Then reboot. This installs the WSL2 kernel and an Ubuntu distro (admin rights are required).
#
# Verify after reboot:
#
#   wsl --status
#
#### Step 2 - Install Docker Desktop ####
# Download Docker Desktop for Windows - AMD64 from docker.com/products/docker-desktop and run the
# installer. Keep "Use WSL 2 instead of Hyper-V" checked (the default).
#
# Verify in Git Bash or PowerShell:
#
#   docker run --rm hello-world


##############################################################################################
## Running the ASan and UBSan checks with both clang and gcc
##   AddressSanitizer: detects buffer overflows, use-after-free, ...
##   UndefinedBehaviorSanitizer: detects signed integer overflow, invalid shifts, ...
##############################################################################################
# First create the package tarball:
#     sh build-cran-package.sh
# This creates 'gpboost_<version>.tar.gz' in the current folder.
#
# The 'r-check' script inside the container does three things:
#   1. it installs all dependencies of the package (including 'Suggests', so that 'testthat' is
#      available and the tests are actually run - without it, 'R CMD check' silently skips them),
#   2. it runs 'R CMD check' with the arguments given in the environment variable 'CHECK_ARGS',
#   3. it searches the check output for sanitizer reports, prints them, and exits with status 1
#      if it found any. The exit status can thus be trusted, no manual searching is needed.
# 'r-check' checks ALL '*.tar.gz' files in the mounted folder (the wildcard is expanded inside the
# container, so no file name has to be given on the command line).
#
#
# --- Windows PowerShell ('`' is the line continuation character) ---
#
    # docker run --rm -v "${PWD}:/check" `
    #   -e GPBOOST_ALL_TESTS=GPBOOST_ALL_TESTS `
    #   -e MAKEFLAGS=-j4 `
    #   -e OMP_NUM_THREADS=10 `
    #   -e OMP_THREAD_LIMIT=10 `
    #   -e ASAN_OPTIONS=detect_leaks=0 `
    #   -e UBSAN_OPTIONS=print_stacktrace=1 `
    #   -e CHECK_ARGS="--no-manual --no-build-vignettes" `
    #   ghcr.io/r-hub/containers/clang-asan `
    #   r-check
    # 
    # docker run --rm -v "${PWD}:/check" `
    #   -e GPBOOST_ALL_TESTS=GPBOOST_ALL_TESTS `
    #   -e MAKEFLAGS=-j4 `
    #   -e OMP_NUM_THREADS=10 `
    #   -e OMP_THREAD_LIMIT=10 `
    #   -e ASAN_OPTIONS=detect_leaks=0 `
    #   -e UBSAN_OPTIONS=print_stacktrace=1 `
    #   -e CHECK_ARGS="--no-manual --no-build-vignettes" `
    #   ghcr.io/r-hub/containers/gcc-asan `
    #   r-check
#
# IMPORTANT for PowerShell: PowerShell splits '"${PWD}":/check' into two arguments instead of
# concatenating them. Docker then treats ':/check' as the image name and aborts with
# "docker: invalid reference format". The WHOLE value therefore has to be quoted: -v "${PWD}:/check"
# 
#
# --- Git Bash ('\' is the line continuation character) ---
#
#     docker run --rm -v "$(pwd)":/check \
#       -e GPBOOST_ALL_TESTS=GPBOOST_ALL_TESTS \
#       -e MAKEFLAGS=-j4 \
#       -e ASAN_OPTIONS=detect_leaks=0 \
#       -e UBSAN_OPTIONS=print_stacktrace=1 \
#       -e CHECK_ARGS="--no-manual --no-build-vignettes" \
#       ghcr.io/r-hub/containers/clang-asan \
#       r-check
#
#     docker run --rm -v "$(pwd)":/check \
#       -e GPBOOST_ALL_TESTS=GPBOOST_ALL_TESTS \
#       -e MAKEFLAGS=-j4 \
#       -e ASAN_OPTIONS=detect_leaks=0 \
#       -e UBSAN_OPTIONS=print_stacktrace=1 \
#       -e CHECK_ARGS="--no-manual --no-build-vignettes" \
#       ghcr.io/r-hub/containers/gcc-asan \
#       r-check
#
#
# WHERE THE RESULTS ARE
# ---------------------
# The container writes everything into the folder that is mounted on '/check', i.e. into
# 'gpboost.Rcheck' in the CURRENT folder on Windows. '--rm' only removes the container, the results
# remain on disk and can be looked at after the run has finished:
#
#   gpboost.Rcheck/00install.out       output of the compilation / installation
#   gpboost.Rcheck/00check.log         the individual check steps and the final "Status:" line
#   gpboost.Rcheck/tests/*.Rout        output of the tests (if the test file ran through)
#   gpboost.Rcheck/tests/*.Rout.fail   output of the tests (written instead of '.Rout' if the tests
#                                        failed, this is where a sanitizer abort ends up)
#   gpboost.Rcheck/gpboost-Ex.Rout     output of running the examples of the help pages
#
# The reports of the sanitizers are contained in these '.Rout' / '.Rout.fail' files, at the place
# where the corresponding code was run. In addition, 'r-check' prints them to the console and exits
# with status 1 (see above), i.e. it is usually enough to look at the console output.
#
# ASan and UBSan do NOT appear as separate steps in '00check.log': they are compiled into the
# library and they only write something if they actually find a problem. "Status: OK" without any
# sanitizer report therefore means that nothing was found -- PROVIDED that the tests really ran.
# This should always be checked in 'gpboost.Rcheck/tests/testthat.Rout':
#
#     [ FAIL 0 | WARN 0 | SKIP 0 | PASS 3489 ]     <- correct, the tests ran
#     [ FAIL 0 | WARN 0 | SKIP 20 | PASS 1 ]       <- 'GPBOOST_ALL_TESTS' was not set, nothing ran
#
# '00install.out' contains a lot of "readelf: Warning: Unrecognized form: 0x22 / 0x23" and, as a
# consequence of these, "Bogus end-of-siblings marker" and "DIE ... refers to abbreviation number
# ... which does not exist". These warnings can be IGNORED: the containers combine a very new clang
# (which writes DWARF 5 debug information, 0x22 = DW_FORM_loclistx, 0x23 = DW_FORM_rnglistx) with
# the 'readelf' of the older binutils of the base image, which does not know these forms and then
# loses its position in the '.debug_info' section. They only concern the step "checking absolute
# paths in shared objects and dynamic libraries" of 'R CMD INSTALL', they are not related to
# GPBoost and they do not influence the result ('* DONE (gpboost)' follows afterwards).
# They can be filtered out with:
#
#     grep -v "^readelf: Warning" gpboost.Rcheck/00install.out
#
# Searching the results afterwards (Git Bash):
#
#     grep -rE "AddressSanitizer|runtime error" gpboost.Rcheck/tests/ gpboost.Rcheck/*.Rout
#
# ... and in PowerShell:
#
#     Select-String -Path gpboost.Rcheck\tests\*.Rout*, gpboost.Rcheck\*.Rout `
#         -Pattern "AddressSanitizer", "runtime error"
#
# Following a running check (the files are written while the check is running):
#
#     tail -f gpboost.Rcheck/00install.out                            # Git Bash
#     Get-Content gpboost.Rcheck\00check.log -Wait -Tail 20           # PowerShell
#
# Notes:
#   - Both sanitizers run in the SAME container: the compilers of the 'clang-asan' image are
#     defined as 'clang++-22 ... -fsanitize=address,undefined ...' in R's 'Makeconf'. A separate
#     run with the 'clang-ubsan' image is therefore largely redundant.
#   - The 'gcc-asan' image ('ghcr.io/r-hub/containers/gcc-asan') is NOT redundant, it uses
#     '-fsanitize=address,undefined,bounds-strict' and, in contrast to the clang image, it does not
#     switch off the alignment and the float-division-by-zero checks. It can thus find additional
#     problems and it is worth running as well.
#   - 'ASAN_OPTIONS=detect_leaks=0' switches off the leak detector. R itself does not free all of
#     its memory at the end, so the output would otherwise be dominated by leaks in R and not in
#     GPBoost. 'UBSAN_OPTIONS=print_stacktrace=1' adds stack traces to the UBSan reports (ASan
#     always prints them, UBSan does not do so by default).
#   - Do not run two checks on the same folder at the same time: 'R CMD check' writes everything
#     into a single '<package>.Rcheck' folder and the two runs overwrite each other's files.
#   - The checks are slow: the whole C++ code is recompiled inside the container and the
#     instrumented code is roughly 2x slower and needs about 3x more memory. Only set
#     'GPBOOST_ALL_TESTS' once a short run works.
#   - CPU / parallelization:
#       * Compilation. 'MAKEFLAGS' is NOT set in the containers, i.e. by default the roughly 40
#         source files are compiled one after the other, which dominates the total run time.
#         '-e MAKEFLAGS=-j4' compiles them in parallel. Do not use a much larger number: every
#         instrumented compiler process needs a lot of memory (check the memory that is assigned
#         to Docker with 'docker info').
#       * Running the tests. The containers set 'OMP_THREAD_LIMIT=2', which is a hard upper limit
#         for the size of an OpenMP team (CRAN also restricts the checks to 2 cores). GPBoost
#         therefore runs with 2 threads and writes
#           "OMP: Warning #96: Cannot form a team with 16 threads, using 2 instead"
#         into the test output. This is also the reason why 'nproc' reports 2 in the container
#         while 'nproc --all', '/proc/cpuinfo' and R's 'parallel::detectCores()' report all cores,
#         and why '-e OMP_NUM_THREADS=...' has no effect (a limit beats a request).
#         '-e OMP_THREAD_LIMIT=8' removes the limit, but it is better to keep it: it is what CRAN
#         does, and fewer threads make the output of the sanitizers more reproducible.
#   - With '--rm', the dependencies installed in step 1 above are lost after every run. Adding
#     '-v gpboost-rlib:/usr/local/lib/R/site-library' caches them in a named Docker volume.
#   - See https://github.com/r-hub/containers for the list of containers and further options.


##############################################################################################
## Running the rchk check (analysis of the use of the R C API)
##   rchk statically checks the C/C++ code for missing 'PROTECT' calls, i.e. for R objects that
##   can be garbage collected while they are still in use. It does NOT run the tests, it only
##   compiles the package (with clang/wllvm at '-O0') and then analyzes the resulting bitcode.
##############################################################################################
#
# --- Windows PowerShell ---
#
    # docker run --rm -v "${PWD}:/check" `
    #   -e MAKEFLAGS=-j4 `
    #   ghcr.io/r-hub/containers/rchk `
    #   r-check
#
# --- Git Bash ---
#
#     docker run --rm -v "$(pwd)":/check -e MAKEFLAGS=-j4 \
#       ghcr.io/r-hub/containers/rchk r-check
#
# The findings are printed to the console (lines such as "[UP] unprotected variable ... while
# calling allocating function ..." or "[PB] ... has possibly unprotected ...").
#
# A run without any findings looks like this:
#
#     Running bcheck
#     ==== rchk bcheck =========================================
#     ERROR: too many states (abstraction error?) in function strptime_internal
#     ERROR: too many states (abstraction error?) in function bcEval_loop
#     ERROR: too many states (abstraction error?) in function RunGenCollect
#     Analyzed 111073 functions, traversed 354876 states.
#     ------------------------------------------------------
#
# i.e. NO '[UP]' / '[PB]' line at all. The three "too many states" errors are not related to
# GPBoost: 'strptime_internal', 'bcEval_loop' and 'RunGenCollect' are functions of R itself (the
# date parser, the bytecode interpreter and the garbage collector), which are analyzed together
# with the package. They are so large that rchk's abstract interpreter reaches its limit on the
# number of states and gives up ON THESE FUNCTIONS ONLY. This happens for every package.
#
# 'objdump: Warning: Unrecognized form: 0x22 / 0x23' and the "DIE at offset ... refers to
# abbreviation number ... which does not exist" that follows from them can be ignored, for the same
# reason as the 'readelf' warnings described further above (DWARF 5 vs. old binutils).
#
# Note that the container only runs 'bcheck' (its 'rchk.sh' sets 'TOOLS=bcheck'); the additional
# rchk tools 'maacheck' and 'fficheck' are not run.
#
# Since "no findings" produces no output of its own, every run should be checked for the following
# failure mode, in which rchk never runs at all:
#
#   'rchk.sh' inside the container determines the package name with
#       PACKAGE=`cat DESCRIPTION | grep "^Package:" | cut -d: -f2 | tr -d '[:blank:]'`
#   '[:blank:]' is space and tab, it does NOT include the carriage return. On Windows, 'R CMD build'
#   rewrites DESCRIPTION and writes it in text mode, i.e. with CRLF line endings. Without the
#   normalization that 'build-cran-package.sh' now does at the end (it unpacks the tarball, converts
#   DESCRIPTION back to LF and packs it again), PACKAGE becomes 'gpboost\r', the directory
#   'packages/lib/gpboost\r' does not exist, and rchk aborts with
#       Cannot find package gpboost (/opt/R/devel-rchk/packages/lib/gpboost does not exist).
#   directly after '* DONE (gpboost)', i.e. AFTER a successful compilation and BEFORE any
#   analysis. On the console this message looks garbled, because the '\r' it contains moves the
#   cursor back to the beginning of the line. If this line appears, the tarball is too old and has
#   to be rebuilt with 'sh build-cran-package.sh'.
#
# Note: the compilation warnings that this container prints come from clang, they are unrelated to
# rchk itself and are also shown by 'helpers/check_compiler_warnings.sh'.


##############################################################################################
## Why 'rhub::rhub_check()' is not used here
##############################################################################################
# All functions of R-hub version 1 ('check()', 'check_for_cran()', 'check_with_sanitizers()',
# 'validate_email()', ...) are defunct since R-hub version 2.
# The replacement 'rhub_check()' checks the R package that lies in the ROOT of a GitHub repository.
# The root of the GPBoost repository is not an R package: the R package is first assembled from
# 'R-package/' and the C++ sources by 'build-cran-package.sh' into the (git-ignored) folder
# 'gpboost_r'. Using 'rhub_check()' would thus require a custom GitHub Actions workflow that first
# runs 'build-cran-package.sh' and then checks the resulting package.
# The Docker commands above work directly with the tarball created by 'build-cran-package.sh' and
# are therefore the simpler option for this package.
