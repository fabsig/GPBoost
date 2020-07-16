# User options
use_precompile <- FALSE
use_gpu <- FALSE
use_mingw <- FALSE # use this option for CRAN submission

# 32-bit version can be installed now (the following is no longer needed)
# if (.Machine$sizeof.pointer != 8){
#   stop("Only support 64-bit R, please check your the version of your R and Rtools.")
# }

R_int_UUID <- .Internal(internalsID())
R_ver <- as.double(R.Version()$major) + as.double(R.Version()$minor)/10

if (!(R_int_UUID == "0310d4b8-ccb1-4bb8-ba94-d36a55f60262"
      || R_int_UUID == "2fdf6c18-697a-4ba7-b8ef-11c0d92f1327")){
  print("Warning: unmatched R_INTERNALS_UUID, may not run normally.")
}

# Move in CMakeLists.txt
if (!file.copy("../inst/bin/CMakeLists.txt", "CMakeLists.txt", overwrite = TRUE)){
  stop("Copying CMakeLists failed")
}
##the following is needed for passing CRAN checks (multi-arch / 32-bit)
file.copy("../inst/bin/CMakeLists.txt", "../src/CMakeLists.txt", overwrite = TRUE)

# Get some paths
source_dir <- file.path(R_PACKAGE_SOURCE, "src", fsep = "/")
build_dir <- file.path(source_dir, "build", fsep = "/")

WINDOWS_BUILD_TOOLS <- list(
  "MinGW" = c(
    build_tool = "mingw32-make.exe"
    , makefile_generator = "MinGW Makefiles"
  )
  , "MSYS2" = c(
    build_tool = "make.exe"
    , makefile_generator = "MSYS Makefiles"
  )
)
# Rtools 4.0 moved from MinGW to MSYS toolchain.
if (R_ver >= 4.0) {
  windows_toolchain <- "MSYS2"
} else {
  windows_toolchain <- "MinGW"
}
windows_build_tool <- WINDOWS_BUILD_TOOLS[[windows_toolchain]][["build_tool"]]
windows_makefile_generator <- WINDOWS_BUILD_TOOLS[[windows_toolchain]][["makefile_generator"]]

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
  if (!WINDOWS | use_mingw) {##MSVS cannot load R library
    cmake_cmd <- paste0(cmake_cmd, " -DBUILD_FOR_R_REGISTER=ON ")
  }
  if (R_ARCH == "/i386"){
    cmake_cmd <- paste0(cmake_cmd, " -DBUILD_32BIT_R=ON ")
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
      ##Find correct version of mingw (32- or 64-bit)
      build_tool_locs <- system(paste0("where ",windows_build_tool),intern=TRUE)
      if (R_ver >= 4.0) {
        for (loc in build_tool_locs) {
          if (grepl("make.exe",loc)) {
            build_tool_exe <- loc
            break
          }
        }
        mingw_path <- substr(build_tool_exe,1,gregexpr("make",build_tool_exe)[[1]][1]-10)
        if (R_ARCH == "/i386"){
          mingw_path <- file.path(mingw_path,"mingw32","bin")
        }else{
          mingw_path <- file.path(mingw_path,"mingw64","bin")
        }
      }else{
        if (R_ARCH == "/i386"){
          for (loc in build_tool_locs) {
            if (grepl("mingw_32",loc)) {
              build_tool_exe <- loc
              break
            }
          }
        }
        else {
          for (loc in build_tool_locs) {
            if (grepl("mingw_64",loc)) {
              build_tool_exe <- loc
              break
            }
          }
        }
        mingw_path <- substr(build_tool_exe,1,gregexpr("mingw32-make",build_tool_exe)[[1]][1]-1)
      }
      cmake_cmd <- paste0(cmake_cmd, " -G ", shQuote(windows_makefile_generator)," -DMINGW_PATH=",mingw_path)
      build_cmd <- paste0(build_tool_exe," _gpboost")# no absolute path: e.g. build_cmd <- "mingw32-make.exe _gpboost"
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
        cmake_cmd <- paste0(cmake_cmd, " -G ", shQuote(windows_makefile_generator)," ") # Switch to MinGW on failure, try build once
        system(paste0(cmake_cmd, " ..")) # Must build twice for Windows due sh.exe in Rtools
        build_cmd <- paste0(build_tool_exe," _gpboost")
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
    
    # Make sure that makefile has the required UNIX style line endings for passing R CMD check 
    output.file <- file(generated_makefile, "wb")
    writeLines(
      text = makefile_txt
      , con = output.file
      , sep = "\n"
    )
    close(output.file)
  }
  
  # Make sure that the following files have the required UNIX style line endings for passing R CMD check 
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