# Dependencies.cmake - Handle third-party dependencies for a_simple_tensor
#
# This file manages the integration of third-party libraries such as GoogleTest.
# It follows patterns similar to PyTorch's dependency management.
# It works together with cmake/modules/findgoogle.cmake to find and integrate GoogleTest.

# ---[ GoogleTest integration
# Check if testing is enabled
cmake_minimum_required(VERSION 3.15)

if(BUILD_TESTING)
  message(STATUS "Building with tests enabled - integrating GoogleTest")

  # First, find GoogleTest using our custom find module
  find_package(GoogleTest QUIET)

  if(GoogleTest_FOUND)
    message(STATUS "Found GoogleTest at: ${GOOGLETEST_ROOT_DIR}")

    # Preserve build options (PyTorch style)
    set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

    # We will build gtest as static libs and embed it directly into the test binaries.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)

    # For gtest, we will simply embed it into our test binaries, so we won't
    # need to install it.
    set(INSTALL_GTEST OFF CACHE BOOL "Install gtest." FORCE)
    set(BUILD_GMOCK ON CACHE BOOL "Build gmock." FORCE)

    # Add GoogleTest subdirectory using the function from findgoogle.cmake
    add_google_test_subdirectory()

    # Include GoogleTest and GoogleMock headers
    # These are already included by add_google_test_subdirectory, but we keep them for clarity
    if(GTEST_FOUND)
      include_directories(BEFORE SYSTEM ${GTEST_INCLUDE_DIR})
    endif()

    if(GMOCK_FOUND)
      include_directories(BEFORE SYSTEM ${GMOCK_INCLUDE_DIR})
    endif()

    # Recover build options.
    set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)

    message(STATUS "GoogleTest integration complete")
  else()
    message(WARNING "GoogleTest not found. Tests will not be built.")
  endif()
else()
  message(STATUS "Building without tests - skipping GoogleTest integration")
endif()

# ---[ Mimalloc integration
# Option to enable mimalloc for improved memory allocation performance
option(USE_MIMALLOC "Use mimalloc memory allocator for improved performance" ON)

if(USE_MIMALLOC AND BUILD_CXX)
  message(STATUS "Building with mimalloc enabled - integrating mimalloc")

  # First, find mimalloc using our custom find module
  find_package(Mimalloc QUIET)

  if(Mimalloc_FOUND)
    message(STATUS "Found mimalloc at: ${MIMALLOC_ROOT_DIR}")

    # Add mimalloc as a subdirectory
    add_mimalloc_subdirectory()

    # Set mimalloc options based on build type
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      # In debug builds, we might want more safety features
      set(MIMALLOC_SECURE ON CACHE BOOL "Secure mode" FORCE)
    else()
      # In release builds, prioritize performance
      set(MIMALLOC_SECURE OFF CACHE BOOL "Secure mode" FORCE)
    endif()

    # Prefer static library for easier integration
    set(MIMALLOC_USE_STATIC ON CACHE BOOL "Use static mimalloc library" FORCE)

    # Don't override malloc by default to avoid conflicts with custom allocators
    set(MIMALLOC_OVERRIDE OFF CACHE BOOL "Override malloc" FORCE)

    # Link mimalloc to the main library and add compile definition
    if(TARGET mytensor)
      if(TARGET mimalloc-static)
        target_link_libraries(mytensor PRIVATE mimalloc-static)
        target_compile_definitions(mytensor PRIVATE USE_MIMALLOC)
        message(STATUS "Linked mimalloc-static to mytensor library and added USE_MIMALLOC definition")
      elseif(TARGET mimalloc)
        target_link_libraries(mytensor PRIVATE mimalloc)
        target_compile_definitions(mytensor PRIVATE USE_MIMALLOC)
        message(STATUS "Linked mimalloc to mytensor library and added USE_MIMALLOC definition")
      endif()
    endif()

    # Also link mimalloc to test executables if they exist
    if(BUILD_TESTING)
      # This will be handled in tests/CMakeLists.txt if needed
      message(STATUS "mimalloc will be available for test executables")
    endif()

    message(STATUS "mimalloc integration complete")
  else()
    message(WARNING "mimalloc not found. Using default memory allocator.")
  endif()
else()
  message(STATUS "Building without mimalloc - using default memory allocator")
endif()

# ---[ Other dependencies can be added here following the same pattern
# For example:
# if(USE_SOME_LIBRARY)
# # Integration code for that library
# endif()

# ---[ MKL integration
# Option to enable Intel Math Kernel Library (MKL) for optimized math operations
option(USE_MKL "Use Intel Math Kernel Library for optimized math operations" ON)

if(USE_MKL)
  message(STATUS "Building with MKL enabled - integrating Intel Math Kernel Library")

  # First, find MKL using our custom find module
  find_package(MKL QUIET)

  if(MKL_FOUND)
    message(STATUS "  Found MKL at: ${MKL_ROOT_DIR}")
    message(STATUS "  MKL version: ${MKL_VERSION}")
    message(STATUS "  MKL interface: ${MKL_INTERFACE}")
    message(STATUS "  MKL threading: ${MKL_THREADING}")

    # Link MKL to the main library and add compile definition
    if(TARGET umat)
      # Use the target_link_mkl function provided by FindMKL.cmake
      target_link_mkl(umat)
      message(STATUS "Linked MKL to umat library")
    endif()

    # Also link MKL to test executables if they exist
    if(BUILD_TESTING)
      # This will be handled in tests/CMakeLists.txt if needed
      message(STATUS "MKL will be available for test executables")
    endif()

    message(STATUS "MKL integration complete")
  else()
    message(WARNING "MKL not found. Using default math libraries.")
  endif()
else()
  message(STATUS "Building without MKL - using default math libraries")
endif()

# ---[ ABAQUS integration
# Option to enable ABAQUS for finite element analysis integration
option(USE_ABAQUS "Use ABAQUS for finite element analysis integration" OFF)

if(USE_ABAQUS)
  message(STATUS "------------------------------------------------------------------")
  message(STATUS "Building with ABAQUS enabled - integrating ABAQUS")

  # First, find ABAQUS using our custom find module
  find_package(Abaqus QUIET)

  if(Abaqus_FOUND)
    message(STATUS "Found ABAQUS at: ${ABAQUS_PATHS}")
    message(STATUS "ABAQUS include directories: ${ABAQUS_INCLUDE_DIRS}")
    message(STATUS "ABAQUS library directory: ${ABAQUS_LIBRARY_DIRS}")

    # Link ABAQUS to the main library and add compile definition
    if(TARGET umat)
      # Add include directories
      target_include_directories(umat PRIVATE ${ABAQUS_INCLUDE_DIRS})

      # Add library directories
      target_link_directories(umat PRIVATE ${ABAQUS_LIBRARY_DIRS})

      # Link all ABAQUS libraries
      target_link_libraries(umat PRIVATE ${ABAQUS_LIBRARIES})

      # Add compile definition
      target_compile_definitions(umat PRIVATE USE_ABAQUS)

      message(STATUS "Linked ABAQUS to umat library and added USE_ABAQUS definition")
    endif()

    # Also link ABAQUS to test executables if they exist
    if(BUILD_TESTING)
      # This will be handled in tests/CMakeLists.txt if needed
      message(STATUS "ABAQUS will be available for test executables")
    endif()

    message(STATUS "ABAQUS integration complete")
  else()
    message(WARNING "ABAQUS not found. ABAQUS integration will be disabled.")
  endif()

  message(STATUS "------------------------------------------------------------------")
else()
  message(STATUS "Building without ABAQUS - ABAQUS integration disabled")
endif()

# ---[ LibTorch integration
# Option to enable LibTorch for PyTorch tensor operations
option(USE_LIBTORCH "Use LibTorch for PyTorch tensor operations" ON)

if(USE_LIBTORCH AND BUILD_CXX)
  message(STATUS "Building with LibTorch enabled - integrating PyTorch")

  # First, find LibTorch using our custom find module
  find_package(LibTorch QUIET)

  if(LibTorch_FOUND)
    message(STATUS "Library base name     : LibToch")
    message(STATUS "LibTorch version      : ${LIBTORCH_VERSION}")
    message(STATUS "Found LibTorch at     : ${LIBTORCH_ROOT_DIR}")
    message(STATUS "LibTorch include Dir  : ${LIBTORCH_INCLUDE_DIRS}")

    # Link LibTorch to the main library and add compile definition
    if(TARGET mytensor)
      # Link to the main torch library
      target_link_libraries(mytensor PRIVATE torch)

      # Add compile definition
      target_compile_definitions(mytensor PRIVATE USE_LIBTORCH)

      # Add include directories (torch target should handle this, but we can be explicit)
      if(TORCH_INCLUDE_DIRS)
        target_include_directories(mytensor PRIVATE ${TORCH_INCLUDE_DIRS})
      endif()

      message(STATUS "Linked LibTorch to mytensor library and added USE_LIBTORCH definition")
    endif()

    # Also link to umat target if it exists
    if(TARGET umat)
      target_link_libraries(umat PRIVATE torch)
      target_compile_definitions(umat PRIVATE USE_LIBTORCH)
      message(STATUS "Linked LibTorch to umat library")
    endif()

    # Also link LibTorch to test executables if they exist
    if(BUILD_TESTING)
      # This will be handled in tests/CMakeLists.txt if needed
      message(STATUS "LibTorch will be available for test executables")
    endif()

    message(STATUS "LibTorch integration complete")
  else()
    message(WARNING "LibTorch not found. Using internal tensor implementation.")
  endif()
else()
  message(STATUS "Building without LibTorch - using internal tensor implementation")
endif()
