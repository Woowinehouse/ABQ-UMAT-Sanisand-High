# 指定cmake最低版本
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# include our cmake snippets
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# REQUIRE OUT-OF-SOURCE BUILDS
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)

# 启用 UTF-8 支持（CMake 3.20+）
set(CMAKE_UTF_8 ON)

# 强制开启颜色输出
set(CMAKE_COLOR_DIAGNOSTICS ON CACHE BOOL "Enable colored diagnostics")

# PROJECT AND VERSION
include(CMakeDependentOption)

if(EXISTS "${LOC_PATH}")
  message(
    FATAL_ERROR
    "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory."
  )
endif()

# 项目信息
project(
  ABQ-UMAT-Sanisand-High
  DESCRIPTION "ABQ-UMAT-Sanisand-High"
  HOMEPAGE_URL "https://github.com/Woowinehouse/ABQ-UMAT-Sanisand-High"
  VERSION "2026.4"
  LANGUAGES C CXX Fortran)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

#[[
-----------------------------------------------------------------------------
set language standards
-----------------------------------------------------------------------------
cmake does not provide any mechanism to set the fortran standard. Adding the
`std` option to compiler flags is the only way to control it. So leave them
be.
]]
# C_plus_plus standard
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CXX_EXTENSIONS OFF)
  set(POSITION_INDEPENDENT_CODE 1)
endif()

# C standard
if(NOT DEFINED CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_C_STANDARD_REQUIRED ON)
endif()

# fortran standard
if(NOT DEFINED CMAKE_Fortran_STANDARD)
  set(CMAKE_Fortran_STANDARD 2018)
  set(CMAKE_Fortran_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# remove NDEBUG flag
string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE
  "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_Fortran_FLAGS_RELEASE
  "${CMAKE_Fortran_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO
  "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO
  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-DNDEBUG" "" CMAKE_Fortran_FLAGS_RELWITHDEBINFO
  "${CMAKE_Fortran_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_MINSIZEREL
  "${CMAKE_C_FLAGS_MINSIZEREL}")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_MINSIZEREL
  "${CMAKE_CXX_FLAGS_MINSIZEREL}")
string(REPLACE "-DNDEBUG" "" CMAKE_Fortran_FLAGS_MINSIZEREL
  "${CMAKE_Fortran_FLAGS_MINSIZEREL}")

message(STATUS "======================================================================")
message(STATUS "          Here are some compiler information :")
message(STATUS "======================================================================")

if(CMAKE_CXX_COMPILER_LOADED)
  message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "The C++ compiler version is: ${CMAKE_CXX_COMPILER_VERSION}")
  message(STATUS "The C++ compiler flags are: ${CMAKE_CXX_FLAGS}")
  message(STATUS "The C++ complier path is: ${CMAKE_CXX_COMPILER}")
endif()

message(STATUS " ")

if(CMAKE_C_COMPILER_LOADED)
  message(STATUS "The C compiler ID is: ${CMAKE_C_COMPILER_ID}")
  message(STATUS "The C compiler version is: ${CMAKE_C_COMPILER_VERSION}")
  message(STATUS "The C compiler flags are: ${CMAKE_C_FLAGS}")
  message(STATUS "The C complier path is: ${CMAKE_C_COMPILER}")
endif()

message(STATUS " ")

if(CMAKE_Fortran_COMPILER_LOADED)
  message(STATUS "The Fortran compiler ID is: ${CMAKE_Fortran_COMPILER_ID}")
  message(STATUS "The Fortran compiler version is: ${CMAKE_Fortran_COMPILER_VERSION}")
  message(STATUS "The Fortran compiler flags are: ${CMAKE_Fortran_FLAGS}")
  message(STATUS "The Fortran complier path is: ${CMAKE_Fortran_COMPILER}")
endif()

# --------------------------------------------------------------------------------
# ---[ Include dependencies
# Handle third-party dependencies (GoogleTest, etc.)
include(cmake/Dependencies.cmake)
include(cmake/modules/colors.cmake)
define_colors()

# ------------------------------------------------------------------------------
# check parameters and print build information
# ------------------------------------------------------------------------------
# Set default variable_requires()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(__ENABLE_DEBUG__ ON)
  option(BUILD_TESTING "Build tests" ON)

# define variable
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  option(BUILD_TESTING "Build tests" OFF)
endif()

if(NOT USE_MIMALLOC)
  set(USE_MIMALLOC OFF)
endif()

if(NOT USE_ABAQUS)
  set(USE_ABAQUOS OFF)
endif()

if(NOT STRICT)
  set(STRICT ON)
endif()

if(NOT BUILD_FORTRAN)
  set(BUILD_FORTRAN OFF)
endif()

if(NOT BUILD_CXX)
  set(BUILD_CXX OFF)
endif()

# BUILD_SHARED_LIBS is now a cache variable that can be set by the user
# Default is OFF (build static libraries)
if(NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries instead of static libraries")
endif()

if(STRICT STREQUAL "ON")
  set(__STRICT_VALIDATION__ ON)
endif()

if(USE_ABAQUS STREQUAL "ON")
  set(__USE_FORTRAN__ ON)
endif()

message(STATUS "======================================================================")
message(STATUS "              Here Are Some Build Options :")
message(STATUS "======================================================================")
message(STATUS "Build Type         : ${CMAKE_BUILD_TYPE}")
message(STATUS "Use_Mimalloc       : ${USE_MIMALLOC}")
message(STATUS "Use_ABAQUS         : ${USE_ABAQUS}")
message(STATUS "BUILD_SHARED_LIBS  : ${BUILD_SHARED_LIBS}")

# Enable testing (similar to PyTorch's BUILD_TEST option)
if(BUILD_TESTING)
  message(STATUS "BUILD_TESTING      : ON")
  set(__GTEST__BUILD__ ON)
  enable_testing()
else()
  message(STATUS "BUILD_TESTING      : OFF")
endif()

# =================================================================================================
set(CMAKE_PREFIX_PATH "${LIBTORCH_PATH}" CACHE PATH "Path to LibTorch CMake configuration files")
set(CMAKE_PREFIX_PATH "${MKL_CONFIG_PATH}" CACHE PATH "Path to MKLConfig.cmake")
set(LIBTORCH_LIB_PATH "${LIBTORCH_PATH}" CACHE PATH "LibTorch DLL directory")
set(LOG_PATH "${ABA_LOG_PATH}" CACHE PATH "ABAQUS LOG_PATH ")

#
configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/config.h.in
  ${CMAKE_SOURCE_DIR}/src/cpp/utils/config.h
  @ONLY
)

# check
# 打印SDK和系统信息
message(STATUS "======================================================================")
message(STATUS "           Here Are Some Build System Information :")
message(STATUS "======================================================================")
message(STATUS "Generator          : ${CMAKE_GENERATOR}")

# check configure platform
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Configuring SYSTEM : Linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message(STATUS "Configuring SYSTEM : Windows")
else()
  message(STATUS "Configuring SYSTEM :${CMAKE_SYSTEM_NAME}")
endif()

message(STATUS "System Version     : ${CMAKE_SYSTEM_VERSION}")

# ---[ Build options

# 构建时生成compile_commands.json（重要）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置输出目录
# Use the GNUInstallDirs module to set the standard directories where CMake saves static and dynamic libraries,
# as well as executable files. We also instruct CMake to place the module files compiled by Fortran in the "modules" directory:
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)

#[[
-------------------------------------------------------------------------------
compile
-------------------------------------------------------------------------------
]]
if(BUILD_FORTRAN)
  # complie include(FortranCInterface) files into a shared library
  add_subdirectory(src/fortran)
endif()

if(BUILD_CXX)
  # complie implement files into a shared library
  add_subdirectory(src/cpp/core)

  # complie main files into a shared library
  add_subdirectory(src/cpp)
endif()

#[[
=================================================================================================
define some variables that need to be replaced
=================================================================================================
contains
ABAQUS_SCRATCH_DIR         : Directory to be used for scratch files
ABAQUS_FORTRAN_COMPILER_ID :
ABAQUS_CXX_COMPILER_ID     :
]]
set(ABAQUS_SCRATCH_DIR "${CMAKE_BINARY_DIR}/scratch" CACHE PATH "Abaqus scratch directory")

# make the scartch path
make_directory(${ABAQUS_SCRATCH_DIR})

# --- 2. 自动检测并设置 Fortran 编译器 ---
if(CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
  set(ABAQUS_FORTRAN_COMPILER_ID "ifx")
  message(STATUS "Detected Intel oneAPI (ifx)")
elseif(CMAKE_Fortran_COMPILER_COMPILER_ID STREQUAL "Intel")
  set(ABAQUS_FORTRAN_COMPILER_ID "ifort")
  message(STATUS "Detected Intel Classic (ifort)")
elseif(CMAKE_Fortran_COMPILER_COMPILER_ID STREQUAL "GNU")
else()
  message(WARNING "Unknown Fortran compiler, defaulting to 'ifort'")
  set(ABAQUS_FORTRAN_COMPILER_ID "ifort")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(ABAQUS_CXX_COMPILER_ID "GCC")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(ABAQUS_CXX_COMPILER_ID "Clang")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(ABAQUS_CXX_COMPILER_ID "cl")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  set(ABAQUS_CXX_COMPILER_ID "icx")
else()
  set(ABAQUS_CXX_COMPILER_ID "cl")
  message(STATUS "Using Intel oneAPI compiler (cl)")
endif()

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/abaqus_v6.env.in
  ${CMAKE_SOURCE_DIR}/abaqus_v6.env
  @ONLY
)

# 根据BUILD_TESTING选项启用测试
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  if(BUILD_TESTING)
    add_subdirectory(tests)
    message(STATUS "Tests enabled (BUILD_TESTING is ON)")
  else()
    message(STATUS "Tests disabled (BUILD_TESTING is OFF)")
  endif()
endif()
