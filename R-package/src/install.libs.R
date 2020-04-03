# User options
use_precompile <- FALSE
use_gpu <- FALSE
use_mingw <- FALSE

if (.Machine$sizeof.pointer != 8){
  stop("Only support 64-bit R, please check your the version of your R and Rtools.")
}

R_int_UUID <- .Internal(internalsID())
R_ver <- as.double(R.Version()$major) + as.double(R.Version()$minor)/10

if (!(R_int_UUID == "0310d4b8-ccb1-4bb8-ba94-d36a55f60262"
      || R_int_UUID == "2fdf6c18-697a-4ba7-b8ef-11c0d92f1327")){
  print("Warning: unmatched R_INTERNALS_UUID, may cannot run normally.")
}

# Move in CMakeLists.txt
if (!file.copy("../inst/bin/CMakeLists.txt", "CMakeLists.txt", overwrite = TRUE)){
  stop("Copying CMakeLists failed")
}

# Get some paths
source_dir <- file.path(R_PACKAGE_SOURCE, "src", fsep = "/")
build_dir <- file.path(source_dir, "build", fsep = "/")

# Check for precompilation
if (!use_precompile) {
  
  # Prepare building package
  dir.create(
    build_dir
    , recursive = TRUE
    , showWarnings = FALSE
  )
  setwd(build_dir)
  
  # Prepare installation steps
  cmake_cmd <- "cmake "
  build_cmd <- "make _gpboost"
  lib_folder <- file.path(R_PACKAGE_SOURCE, "src", fsep = "/")
  
  if (use_gpu) {
    cmake_cmd <- paste0(cmake_cmd, " -DUSE_GPU=ON ")
  }
  if (R_ver >= 3.5) {
    cmake_cmd <- paste0(cmake_cmd, " -DUSE_R35=ON ")
  }
  
  cmake_cmd <- paste0(cmake_cmd, " -DBUILD_FOR_R=ON ")
  if (!WINDOWS | use_mingw) {##MSVS cannot find R library
    cmake_cmd <- paste0(cmake_cmd, " -DBUILD_FOR_R_REGISTER=ON ")
  }
  
  # Pass in R version, used to help find R executable for linking
  R_version_string <- paste(
    R.Version()[["major"]]
    , R.Version()[["minor"]]
    , sep = "."
  )
  # cmake_cmd <- sprintf(
  #   paste0(cmake_cmd, " -DCMAKE_R_VERSION='%s' ")
  #   , R_version_string
  # )
  
  # Could NOT find OpenMP_C on Mojave workaround
  # Using this kind-of complicated pattern to avoid matching to
  # things like "pgcc"
  using_gcc <- grepl(
    pattern = '^gcc$|[/\\]+gcc$|^gcc\\-[0-9]+$|[/\\]+gcc\\-[0-9]+$'
    , x = Sys.getenv('CC', '')
  )
  using_gpp <- grepl(
    pattern = '^g\\+\\+$|[/\\]+g\\+\\+$|^g\\+\\+\\-[0-9]+$|[/\\]+g\\+\\+\\-[0-9]+$'
    , x = Sys.getenv('CXX', '')
  )
  on_mac <- Sys.info()['sysname'] == 'Darwin'
  if (on_mac && !(using_gcc & using_gpp)) {
    cmake_cmd <- paste(cmake_cmd, ' -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" ')
    cmake_cmd <- paste(cmake_cmd, ' -DOpenMP_C_LIB_NAMES="omp" ')
    cmake_cmd <- paste(cmake_cmd, ' -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" ')
    cmake_cmd <- paste(cmake_cmd, ' -DOpenMP_CXX_LIB_NAMES="omp" ')
    cmake_cmd <- paste(cmake_cmd, ' -DOpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib" ')
  }
  
  # Check if Windows installation (for gcc vs Visual Studio)
  if (WINDOWS) {
    if (use_mingw) {
      cmake_cmd <- paste0(cmake_cmd, " -G \"MinGW Makefiles\" ")
      build_cmd <- "mingw32-make.exe _gpboost"
      system(paste0(cmake_cmd, " ..")) # Must build twice for Windows due sh.exe in Rtools
    } else {
      try_vs <- 0
      local_vs_def <- ""
      vs_versions <- c("Visual Studio 16 2019", "Visual Studio 15 2017", "Visual Studio 14 2015")
      for(vs in vs_versions){
        vs_def <- paste0(" -G \"", vs, "\" -A x64")
        tmp_cmake_cmd <- paste0(cmake_cmd, vs_def)
        try_vs <- system(paste0(tmp_cmake_cmd, " .."))
        if (try_vs == 0) {
          local_vs_def = vs_def
          break
        } else {
          unlink("./*", recursive = TRUE) # Clean up build directory
        }
      }
      if (try_vs == 1) {
        cmake_cmd <- paste0(cmake_cmd, " -G \"MinGW Makefiles\" ") # Switch to MinGW on failure, try build once
        system(paste0(cmake_cmd, " ..")) # Must build twice for Windows due sh.exe in Rtools
        build_cmd <- "mingw32-make.exe _gpboost"
      } else {
        cmake_cmd <- paste0(cmake_cmd, local_vs_def)
        build_cmd <- "cmake --build . --target _gpboost --config Release"
        lib_folder <- file.path(R_PACKAGE_SOURCE, "src/Release", fsep = "/")
      }
    }
  }
  
  # Install
  system(paste0(cmake_cmd, " .."))
  
  # R CMD check complains about the .NOTPARALLEL directive created in the cmake
  # Makefile. We don't need it here anyway since targets are built serially, so trying
  # to remove it with this hack
  generated_makefile <- file.path(
    R_PACKAGE_SOURCE
    , "src"
    , "build"
    , "Makefile"
  )
  if (file.exists(generated_makefile)) {
    makefile_txt <- readLines(
      con = generated_makefile
    )
    makefile_txt <- gsub(
      pattern = ".*NOTPARALLEL.*"
      , replacement = ""
      , x = makefile_txt
    )
    
    # Make sure that makefile has the required UNIX style line endings for CRAN
    output.file <- file(generated_makefile, "wb")
    writeLines(
      text = makefile_txt
      , con = output.file
      , sep = "\n"
    )
    close(output.file)
  }
  
  # Make sure that files have the required UNIX style line endings for CRAN
  files_with_pot_bad_ending <- c()
  # OpenMP related files generated by cmake
  OpenMPfiles <- c("OpenMPCheckVersion.c"
                   , "OpenMPCheckVersion.cpp"
                   , "OpenMPTryFlag.c"
                   , "OpenMPTryFlag.cpp")
  for (file in OpenMPfiles) {
    files_with_pot_bad_ending <- c(files_with_pot_bad_ending,
                                   file.path(
                                     R_PACKAGE_SOURCE
                                     , "src"
                                     , "build"
                                     , "CMakeFiles"
                                     , "FindOpenMP",
                                     file
                                   ))
  }
  # Other files generated by cmake, first need to find the directories
  path_cmake <- file.path(
    R_PACKAGE_SOURCE
    , "src"
    , "build"
    , "CMakeFiles")
  dirs <- list.dirs(path_cmake, full.names = TRUE, recursive = FALSE)
  for(dir in dirs){
    has_cmake_version_nbr <- grepl(pattern = 'CMakeFiles/3.|CMakeFiles/2.', x = dir)
    if(has_cmake_version_nbr){
      first=regexpr(pattern = '3.|2.', text = dir)[1]
      cmake_vers <- substring(dir,first, last = 1000000L)
      files_with_pot_bad_ending <- c(files_with_pot_bad_ending,
                                     file.path(
                                       R_PACKAGE_SOURCE
                                       , "src"
                                       , "build"
                                       , "CMakeFiles"
                                       , cmake_vers
                                       , "CompilerIdC"
                                       , "CMakeCCompilerId.c"
                                     ))
      files_with_pot_bad_ending <- c(files_with_pot_bad_ending,
                                     file.path(
                                       R_PACKAGE_SOURCE
                                       , "src"
                                       , "build"
                                       , "CMakeFiles"
                                       , cmake_vers
                                       , "CompilerIdCXX"
                                       , "CMakeCXXCompilerId.cpp"
                                     ))
    }
  }
  
  for (file_bad in files_with_pot_bad_ending) {
    if (file.exists(file_bad)) {
      file_bad_txt <- readLines(
        con = file_bad
      )
      
      output.file <- file(file_bad, "wb")
      writeLines(
        text = file_bad_txt
        , con = output.file
        , sep = "\n"
      )
      close(output.file)
    }
  }
  
  system(build_cmd)
  src <- file.path(lib_folder, paste0("lib_gpboost", SHLIB_EXT), fsep = "/")
  
} else {
  
  # Has precompiled package
  lib_folder <- file.path(R_PACKAGE_SOURCE, "../", fsep = "/")
  if (file.exists(file.path(lib_folder, paste0("lib_gpboost", SHLIB_EXT), fsep = "/"))) {
    src <- file.path(lib_folder, paste0("lib_gpboost", SHLIB_EXT), fsep = "/")
  } else if (file.exists(file.path(lib_folder, paste0("Release/lib_gpboost", SHLIB_EXT), fsep = "/"))) {
    src <- file.path(lib_folder, paste0("Release/lib_gpboost", SHLIB_EXT), fsep = "/")
  } else {
    src <- file.path(lib_folder, paste0("/windows/x64/DLL/lib_gpboost", SHLIB_EXT), fsep = "/") # Expected result: installation will fail if it is not here or any other
  }
  
}

# Check installation correctness
dest <- file.path(R_PACKAGE_DIR, paste0("libs", R_ARCH), fsep = "/")
dir.create(dest, recursive = TRUE, showWarnings = FALSE)
if (file.exists(src)) {
  print(paste0("Found library file: ", src, " to move to ", dest))
  file.copy(src, dest, overwrite = TRUE)
} else {
  stop(paste0("Cannot find lib_gpboost", SHLIB_EXT))
}

# clean up the "build" directory
if (dir.exists(build_dir)) {
  print("Removing 'build/' directory")
  unlink(
    x = build_dir
    , recursive = TRUE
    , force = TRUE
  )
}