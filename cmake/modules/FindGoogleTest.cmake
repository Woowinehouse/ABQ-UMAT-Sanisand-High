# FindGoogleTest.cmake - Find Google Test (gtest) and Google Mock (gmock) libraries
#
# This module finds the Google Test and Google Mock libraries in the
# third_party/googletest directory. It provides a simple way to add
# Google Test as a subdirectory to your CMake project.
#
# Variables set by this module:
#   GOOGLETEST_ROOT_DIR - Path to the googletest root directory
#   GTEST_FOUND - TRUE if Google Test headers are found
#   GMOCK_FOUND - TRUE if Google Mock headers are found
#   GoogleTest_FOUND - TRUE if either Google Test or Google Mock is found
#
# Usage:
#   find_package(GoogleTest)
#   if(GoogleTest_FOUND)
#     add_google_test_subdirectory()
#     target_link_libraries(my_target gtest gtest_main)
#   endif()
#
# Or for Google Mock:
#   find_package(GoogleTest)
#   if(GoogleTest_FOUND)
#     add_google_mock_subdirectory()
#     target_link_libraries(my_target gmock gmock_main)
#   endif()

# Helper macro to set variable with parent scope if available
macro(_google_test_set_var var value)
    if(DEFINED CMAKE_SCRIPT_MODE_FILE)
        set(${var} ${value})
    else()
        # Check if we're in a function (has parent scope)
        if(CMAKE_CURRENT_FUNCTION)
            set(${var} ${value} PARENT_SCOPE)
        else()
            set(${var} ${value})
        endif()
    endif()
endmacro()

# Set paths based on the third_party/googletest directory structure
set(GOOGLETEST_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest")
# Check if the googletest directory exists
if(NOT EXISTS "${GOOGLETEST_ROOT_DIR}")
    if(DEFINED CMAKE_SCRIPT_MODE_FILE)
        set(GoogleTest_FOUND FALSE)
        set(GTEST_FOUND FALSE)
        set(GMOCK_FOUND FALSE)
    else()
        _google_test_set_var(GoogleTest_FOUND FALSE)
        _google_test_set_var(GTEST_FOUND FALSE)
        _google_test_set_var(GMOCK_FOUND FALSE)
    endif()
    if(GoogleTest_FIND_REQUIRED)
        message(FATAL_ERROR "GoogleTest directory not found: ${GOOGLETEST_ROOT_DIR}")
    else()
        message(WARNING "GoogleTest directory not found: ${GOOGLETEST_ROOT_DIR}")
    endif()
    return()
endif()

# Check for Google Test components
if(EXISTS "${GOOGLETEST_ROOT_DIR}/googletest/include/gtest/gtest.h")
    set(GTEST_INCLUDE_DIR "${GOOGLETEST_ROOT_DIR}/googletest/include")
    set(GTEST_FOUND TRUE)
else()
    set(GTEST_FOUND FALSE)
endif()

# Check for Google Mock components
if(EXISTS "${GOOGLETEST_ROOT_DIR}/googlemock/include/gmock/gmock.h")
    set(GMOCK_INCLUDE_DIR "${GOOGLETEST_ROOT_DIR}/googlemock/include")
    set(GMOCK_FOUND TRUE)
else()
    set(GMOCK_FOUND FALSE)
endif()

# Set GoogleTest_FOUND if either component is found
if(GTEST_FOUND OR GMOCK_FOUND)
    set(GoogleTest_FOUND TRUE)
else()
    set(GoogleTest_FOUND FALSE)
    if(GoogleTest_FIND_REQUIRED)
        message(FATAL_ERROR "Neither Google Test nor Google Mock headers found in ${GOOGLETEST_ROOT_DIR}")
    endif()
endif()

# Export variables to parent scope
# Check if we're in script mode (no project context)
if(DEFINED CMAKE_SCRIPT_MODE_FILE)
    # In script mode, set variables in current scope
    set(GoogleTest_FOUND ${GoogleTest_FOUND})
    set(GTEST_FOUND ${GTEST_FOUND})
    set(GMOCK_FOUND ${GMOCK_FOUND})
    set(GOOGLETEST_ROOT_DIR ${GOOGLETEST_ROOT_DIR})
    if(GTEST_FOUND)
        set(GTEST_INCLUDE_DIR ${GTEST_INCLUDE_DIR})
    endif()
    if(GMOCK_FOUND)
        set(GMOCK_INCLUDE_DIR ${GMOCK_INCLUDE_DIR})
    endif()
else()
    # In project mode, export to parent scope using helper macro
    _google_test_set_var(GoogleTest_FOUND ${GoogleTest_FOUND})
    _google_test_set_var(GTEST_FOUND ${GTEST_FOUND})
    _google_test_set_var(GMOCK_FOUND ${GMOCK_FOUND})
    _google_test_set_var(GOOGLETEST_ROOT_DIR ${GOOGLETEST_ROOT_DIR})
    if(GTEST_FOUND)
        _google_test_set_var(GTEST_INCLUDE_DIR ${GTEST_INCLUDE_DIR})
    endif()
    if(GMOCK_FOUND)
        _google_test_set_var(GMOCK_INCLUDE_DIR ${GMOCK_INCLUDE_DIR})
    endif()
endif()

# Handle the QUIET and REQUIRED arguments
if(GoogleTest_FIND_QUIETLY)
    set(_GOOGLETEST_QUIET TRUE)
endif()

if(GoogleTest_FIND_REQUIRED AND NOT GoogleTest_FOUND)
    message(FATAL_ERROR "GoogleTest not found")
endif()

if(NOT GoogleTest_FIND_QUIETLY)
    if(GoogleTest_FOUND)
        message(STATUS "Found GoogleTest: ${GOOGLETEST_ROOT_DIR}")
        if(GTEST_FOUND)
            message(STATUS "  Google Test headers: ${GTEST_INCLUDE_DIR}")
        endif()
        if(GMOCK_FOUND)
            message(STATUS "  Google Mock headers: ${GMOCK_INCLUDE_DIR}")
        endif()
    else()
        message(STATUS "GoogleTest not found")
    endif()
endif()

# Provide a function to add Google Test as a subdirectory
# This function follows PyTorch-style integration patterns
function(add_google_test_subdirectory)
    if(NOT TARGET gtest AND NOT TARGET gtest_main)
        if(GTEST_FOUND)
            message(STATUS "Adding Google Test subdirectory from ${GOOGLETEST_ROOT_DIR}")

            # Preserve current BUILD_SHARED_LIBS setting
            set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

            # Set PyTorch-style options for GoogleTest
            # Build as static libs to embed directly into test binaries
            set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
            set(INSTALL_GTEST OFF CACHE BOOL "Install gtest." FORCE)
            set(BUILD_GMOCK ON CACHE BOOL "Build gmock." FORCE)

            # Add the subdirectory
            add_subdirectory("${GOOGLETEST_ROOT_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/googletest" EXCLUDE_FROM_ALL)

            # Restore BUILD_SHARED_LIBS
            set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)
        else()
            message(WARNING "Google Test not found, cannot add subdirectory")
        endif()
    endif()
endfunction()

# Provide a function to add Google Mock as a subdirectory
# This function follows PyTorch-style integration patterns
function(add_google_mock_subdirectory)
    if(NOT TARGET gmock AND NOT TARGET gmock_main)
        if(GMOCK_FOUND)
            message(STATUS "Adding Google Mock subdirectory from ${GOOGLETEST_ROOT_DIR}")

            # Preserve current BUILD_SHARED_LIBS setting
            set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

            # Set PyTorch-style options for GoogleTest
            # Build as static libs to embed directly into test binaries
            set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
            set(INSTALL_GTEST OFF CACHE BOOL "Install gtest." FORCE)
            set(BUILD_GMOCK ON CACHE BOOL "Build gmock." FORCE)

            # Add the subdirectory
            add_subdirectory("${GOOGLETEST_ROOT_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/googletest" EXCLUDE_FROM_ALL)

            # Restore BUILD_SHARED_LIBS
            set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)
        else()
            message(WARNING "Google Mock not found, cannot add subdirectory")
        endif()
    endif()
endfunction()

# Legacy function names for backward compatibility
function(add_gtest_subdirectory)
    add_google_test_subdirectory()
endfunction()

function(add_gmock_subdirectory)
    add_google_mock_subdirectory()
endfunction()

# Mark variables as advanced
mark_as_advanced(
    GOOGLETEST_ROOT_DIR
    GTEST_INCLUDE_DIR
    GMOCK_INCLUDE_DIR
)
