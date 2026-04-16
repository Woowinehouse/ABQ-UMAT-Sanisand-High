# FindABAQUS.cmake - Simplified Abaqus find module
#
# This module expects ABAQUS_PATHS to be provided via -D flag:
# cmake -DABAQUS_PATHS="E:/ABAQUS2025"
#
# Sets the following variables:
# Abaqus_FOUND          - TRUE if Abaqus is found
# ABAQUS_INCLUDE_DIRS   - Abaqus include directories
# ABAQUS_LIBRARY_DIRS   - Abaqus library directories
# ABAQUS_LIBRARIES      - Abaqus libraries (all .lib files in library directory)

cmake_minimum_required(VERSION 3.15)

# Get path from -D flag or environment variable
if(NOT DEFINED ABAQUS_PATHS)
  if(DEFINED ENV{ABAQUS_PATHS})
    set(ABAQUS_PATHS $ENV{ABAQUS_PATHS})
  else()
    message(FATAL_ERROR "Abaqus paths not specified. Please set ABAQUS_PATHS environment variable or pass -DABAQUS_PATHS=\"path\" to CMake.")
  endif()
endif()

# message(STATUS "Abaqus paths: ${ABAQUS_PATHS}")

# Set product directory
set(ABAQUS_PRODUCT_DIR "${ABAQUS_PATHS}/product")

# message(STATUS "Abaqus product directory: ${ABAQUS_PRODUCT_DIR}")

# Set platform-specific directory (assuming Windows 64-bit)
set(ABAQUS_PLATFORM "win_b64")
set(ABAQUS_CODE_DIR "${ABAQUS_PRODUCT_DIR}/${ABAQUS_PLATFORM}/code")

# Set include directories (exactly as user specified)
set(ABAQUS_INCLUDE_DIRS
  "${ABAQUS_CODE_DIR}/include"
  "${ABAQUS_PRODUCT_DIR}"
)

# Set library directory
set(ABAQUS_LIBRARY_DIR "${ABAQUS_CODE_DIR}/lib")
set(ABAQUS_LIBRARY_DIRS ${ABAQUS_LIBRARY_DIR})

# Check if directories exist
if(EXISTS "${ABAQUS_CODE_DIR}/include" AND EXISTS "${ABAQUS_LIBRARY_DIR}")
  set(Abaqus_FOUND TRUE)

  # Collect all library files
  if(WIN32)
    file(GLOB ABAQUS_LIBRARIES "${ABAQUS_LIBRARY_DIR}/*.lib")
  else()
    file(GLOB ABAQUS_LIBRARIES "${ABAQUS_LIBRARY_DIR}/*.so" "${ABAQUS_LIBRARY_DIR}/*.dylib")
  endif()

# message(STATUS "Found Abaqus include directories: ${ABAQUS_INCLUDE_DIRS}")
# message(STATUS "Found Abaqus library directory: ${ABAQUS_LIBRARY_DIRS}")
else()
  set(Abaqus_FOUND FALSE)

  if(NOT EXISTS "${ABAQUS_CODE_DIR}/include")
    message(WARNING "Abaqus include directory not found: ${ABAQUS_CODE_DIR}/include")
  endif()

  if(NOT EXISTS "${ABAQUS_LIBRARY_DIR}")
    message(WARNING "Abaqus library directory not found: ${ABAQUS_LIBRARY_DIR}")
  endif()
endif()

# Handle standard CMake package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Abaqus
  FOUND_VAR Abaqus_FOUND
  REQUIRED_VARS
  ABAQUS_INCLUDE_DIRS
  ABAQUS_LIBRARY_DIRS
  ABAQUS_LIBRARIES
)

# Mark variables as advanced
mark_as_advanced(
  ABAQUS_PRODUCT_DIR
  ABAQUS_CODE_DIR
  ABAQUS_LIBRARY_DIR
)
