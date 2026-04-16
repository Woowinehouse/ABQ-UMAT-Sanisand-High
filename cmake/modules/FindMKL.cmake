# FindMKL.cmake - Find Intel MKL (Math Kernel Library)
#
# This module finds Intel MKL library, with priority:
# 1. Official MKL CMake config (if available)
# 2. System-installed MKL (via pkg-config/Intel oneAPI)
# 3. Local MKL source in third_party/mkl
# 4. Download MKL via FetchContent (oneAPI MKL open source)
#
# Variables set by this module:
# MKL_ROOT_DIR - Path to MKL root directory
# MKL_FOUND - TRUE if MKL is found
# MKL_INCLUDE_DIR - Path to MKL include directories
# MKL_LIBRARIES - List of MKL libraries
# MKL_VERSION - Version of MKL if available
#
# Usage:
# find_package(MKL)
# if(MKL_FOUND)
# add_mkl_subdirectory()
# target_link_libraries(my_target MKL::MKL)
# endif()
#
# Options:
# MKL_USE_STATIC - Prefer static MKL libraries (default: ON)
# MKL_PARALLEL - Enable MKL parallelism (default: ON)

cmake_minimum_required(VERSION 3.15)

# ====================== 第一步：检查是否已经找到MKL（避免重复查找） ======================
# 如果MKL已经找到且目标存在，直接返回，避免重复执行查找逻辑和消息输出
if(MKL_FOUND)
  # 检查MKL目标是否已经存在
  if(TARGET MKL::MKL OR TARGET MKL::mkl_intel_ilp64)
    if(NOT MKL_FIND_QUIETLY)
      message(STATUS "MKL already found at: ${MKL_ROOT_DIR}, skipping re-find")
    endif()
    return()
  else()
    # MKL标记为找到但目标不存在，可能是缓存状态不一致
    # 不清除MKL_FOUND，而是继续执行查找逻辑
    if(NOT MKL_FIND_QUIETLY)
      message(STATUS "MKL_FOUND is TRUE but MKL targets don't exist. Continuing to find MKL...")
    endif()
  endif()
endif()

# ====================== 第二步：先打印调试信息（排查问题必备） ======================
option(MKL_DEBUG_FIND "Enable detailed debug output for MKL finding" OFF)

if(MKL_DEBUG_FIND)
  message(STATUS "===== MKL Find Debug Info =====")
  message(STATUS "  Preset/Command Line MKL_ROOT: ${MKL_ROOT}")
  message(STATUS "  Preset/Command Line ONEAPI_ROOT: ${ONEAPI_ROOT}")
  message(STATUS "  Environment MKL_ROOT: $ENV{MKL_ROOT}")
  message(STATUS "  Environment ONEAPI_ROOT: $ENV{ONEAPI_ROOT}")
  message(STATUS "  CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
  message(STATUS "===============================")
endif()

include(FindPackageHandleStandardArgs)

if(NOT
  (CMAKE_C_COMPILER_LOADED
  OR CMAKE_CXX_COMPILER_LOADED
  OR CMAKE_Fortran_COMPILER_LOADED))
  message(FATAL_ERROR "FindMKL requires Fortran, C, or C++ to be enabled.")
endif()

# ====================== 第二步：定义可配置选项（绝对不覆盖 preset 值） ======================
# 只有当变量未被 preset/命令行设置时，才使用默认值
if(NOT DEFINED MKL_USE_STATIC)
  set(MKL_USE_STATIC ON CACHE BOOL "Prefer static MKL libraries")
endif()

if(NOT DEFINED MKL_PARALLEL)
  set(MKL_PARALLEL ON CACHE BOOL "Enable MKL parallelism")
endif()

if(NOT DEFINED MKL_INTERFACE)
  set(MKL_INTERFACE "ilp64" CACHE STRING "MKL interface type (ilp64/lp64)")
endif()

if(NOT DEFINED MKL_GIT_TAG)
  set(MKL_GIT_TAG "v2025.3.0" CACHE STRING "Git tag for oneMath download")
endif()

# 初始化全局 CACHE 变量（消除 PARENT_SCOPE 警告）
set(MKL_FOUND FALSE CACHE BOOL "Whether MKL was found" FORCE)
set(MKL_VERSION "unknown" CACHE STRING "MKL version" FORCE)
set(MKL_ROOT_DIR "" CACHE PATH "Root directory of MKL" FORCE)
set(MKL_INCLUDE_DIR "" CACHE PATH "MKL include directory" FORCE)
set(MKL_LIBRARIES "" CACHE STRING "MKL libraries" FORCE)

# 标记为高级变量（不在 CMake GUI 显示）
mark_as_advanced(
  MKL_FOUND
  MKL_ROOT_DIR
  MKL_INCLUDE_DIR
  MKL_LIBRARIES
  MKL_VERSION
)

# ====================== 第零步：优先使用官方 MKL CMake 配置 ======================
# 首先尝试查找并使用官方的 MKLConfig.cmake
if(NOT MKL_FOUND)
  # 尝试通过常见路径查找 MKLConfig.cmake
  set(_MKL_CONFIG_PATHS)

  # ==========================================
  # 优先级 1: 用户显式指定 (CMakePresets.json / 命令行 -D)
  # ==========================================

  # 1.1 检查 USER_ONEAPI_ROOT (oneAPI 根目录)
  if(DEFINED USER_ONEAPI_ROOT)
    file(TO_CMAKE_PATH "${USER_ONEAPI_ROOT}" _NORMALIZED_PATH)
    set(_CANDIDATE_MKL_DIR "${_NORMALIZED_PATH}/mkl/latest/lib/cmake/mkl")

    if(IS_DIRECTORY "${_CANDIDATE_MKL_DIR}")
      message(STATUS "Found MKL via USER_ONEAPI_ROOT: ${_CANDIDATE_MKL_DIR}")
      list(APPEND _MKL_CONFIG_PATHS "${_CANDIDATE_MKL_DIR}")
    else()
      message(WARNING "USER_ONEAPI_ROOT is set, but MKL config dir not found at: ${_CANDIDATE_MKL_DIR}")
    endif()
  endif()

  # 1.2 检查 USER_MKLROOT (直接指定 MKL 根目录)
  if(DEFINED USER_MKLROOT)
    file(TO_CMAKE_PATH "${USER_MKLROOT}" _NORMALIZED_PATH)

    # 注意：MKLROOT 通常已经指向 mkl 目录本身，不需要再加 /mkl
    set(_CANDIDATE_MKL_DIR "${_NORMALIZED_PATH}/lib/cmake/mkl")

    # 兼容某些带 latest 后缀的结构
    set(_CANDIDATE_MKL_DIR_LATEST "${_NORMALIZED_PATH}/latest/lib/cmake/mkl")

    if(IS_DIRECTORY "${_CANDIDATE_MKL_DIR}")
      message(STATUS "Found MKL via USER_MKLROOT: ${_CANDIDATE_MKL_DIR}")
      list(APPEND _MKL_CONFIG_PATHS "${_CANDIDATE_MKL_DIR}")
    elseif(IS_DIRECTORY "${_CANDIDATE_MKL_DIR_LATEST}")
      message(STATUS "Found MKL via USER_MKLROOT (latest): ${_CANDIDATE_MKL_DIR_LATEST}")
      list(APPEND _MKL_CONFIG_PATHS "${_CANDIDATE_MKL_DIR_LATEST}")
    else()
      message(WARNING "USER_MKLROOT is set, but MKL config dir not found at: ${_CANDIDATE_MKL_DIR} or ${_CANDIDATE_MKL_DIR_LATEST}")
    endif()
  endif()

  # ==========================================
  # 优先级 2: 系统环境变量
  # ==========================================

  # 2.1 检查系统环境变量 ONEAPI_ROOT
  if(DEFINED ENV{ONEAPI_ROOT})
    file(TO_CMAKE_PATH "$ENV{ONEAPI_ROOT}" _NORMALIZED_PATH)
    list(APPEND _MKL_CONFIG_PATHS "${_NORMALIZED_PATH}/mkl/latest/lib/cmake/mkl")
  endif()

  # 2.2 检查系统环境变量 MKLROOT
  if(DEFINED ENV{MKLROOT})
    file(TO_CMAKE_PATH "$ENV{MKLROOT}" _NORMALIZED_PATH)
    list(APPEND _MKL_CONFIG_PATHS "${_NORMALIZED_PATH}/lib/cmake/mkl")
    list(APPEND _MKL_CONFIG_PATHS "${_NORMALIZED_PATH}/latest/lib/cmake/mkl")
  endif()

  # ==========================================
  # 优先级 3: 硬编码的常见系统安装路径
  # ==========================================
  if(WIN32)
    list(APPEND _MKL_CONFIG_PATHS
      "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/cmake/mkl"
      "C:/Program Files/Intel/oneAPI/mkl/latest/lib/cmake/mkl"
      "E:/oneapi2022/mkl/latest/lib/cmake/mkl"
    )
  else()
    list(APPEND _MKL_CONFIG_PATHS
      "/opt/intel/oneapi/mkl/latest/lib/cmake/mkl"
      "/opt/intel/mkl/lib/cmake/mkl"
      "/usr/local/lib/cmake/mkl"
    )
  endif()

  # ==========================================
  # 最终查找
  # ==========================================
  # 清除缓存以确保重新查找
  unset(MKL_CONFIG_FILE CACHE)

  # 调试：显示搜索路径
  if(NOT MKL_FIND_QUIETLY)
    message(STATUS "Searching for MKLConfig.cmake in paths:")

    foreach(_path ${_MKL_CONFIG_PATHS})
      message(STATUS "  ${_path}")
    endforeach()
  endif()

  find_file(MKL_CONFIG_FILE
    NAMES MKLConfig.cmake mkl-config.cmake
    PATHS ${_MKL_CONFIG_PATHS}
    NO_DEFAULT_PATH
    DOC "Path to MKLConfig.cmake"
  )

  if(MKL_CONFIG_FILE)
    message(STATUS "Found MKL CMake config: ${MKL_CONFIG_FILE}")

    # 在包含官方配置之前设置MKL_INTERFACE为ilp64
    # 这样MKLConfig.cmake会创建MKL::mkl_intel_ilp64目标
    set(MKL_INTERFACE ilp64 CACHE STRING "MKL interface type" FORCE)

    # 包含官方配置
    include(${MKL_CONFIG_FILE})

    # 设置我们的变量以与现有代码兼容
    if(TARGET MKL::MKL)
      set(MKL_FOUND TRUE CACHE BOOL "Whether MKL was found" FORCE)
      get_filename_component(_MKL_ROOT_DIR "${MKL_CONFIG_FILE}" DIRECTORY)
      get_filename_component(_MKL_ROOT_DIR "${_MKL_ROOT_DIR}/../../.." ABSOLUTE)
      set(MKL_ROOT_DIR "${_MKL_ROOT_DIR}" CACHE PATH "Root directory of MKL" FORCE)

      # 获取包含目录
      get_target_property(_MKL_INCLUDE_DIRS MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)

      if(_MKL_INCLUDE_DIRS)
        set(MKL_INCLUDE_DIR "${_MKL_INCLUDE_DIRS}" CACHE PATH "MKL include directory" FORCE)
        message(STATUS "found MKL include directory: ${_MKL_INCLUDE_DIRS}")
      endif()

      # 获取库信息
      get_target_property(_MKL_LINK_LIBRARIES MKL::MKL INTERFACE_LINK_LIBRARIES)

      if(_MKL_LINK_LIBRARIES)
        set(MKL_LIBRARIES "${_MKL_LINK_LIBRARIES}" CACHE STRING "MKL libraries" FORCE)

        # 使用foreach打印每个库
        message(STATUS "Individual MKL libraries:")
        set(_index 1)

        foreach(link ${MKL_LIBRARIES})
          message(STATUS "[${_index}] ${link}")
          math(EXPR _index "${_index} + 1")
        endforeach()
      endif()

      # 创建MKL::mkl_intel_ilp64别名以兼容Torch
      # Torch期望MKL::mkl_intel_ilp64目标，但MKLConfig.cmake创建的是MKL::MKL
      if(NOT TARGET MKL::mkl_intel_ilp64)
        if(TARGET MKL::MKL)
          add_library(MKL::mkl_intel_ilp64 INTERFACE IMPORTED)
          target_link_libraries(MKL::mkl_intel_ilp64 INTERFACE MKL::MKL)
          message(STATUS "Created MKL::mkl_intel_ilp64 alias for Torch compatibility")
        elseif(TARGET MKL::MKL_DPCPP)
          # 如果MKL::MKL_DPCPP存在，也为其创建别名
          add_library(MKL::mkl_intel_ilp64 INTERFACE IMPORTED)
          target_link_libraries(MKL::mkl_intel_ilp64 INTERFACE MKL::MKL_DPCPP)
          message(STATUS "Created MKL::mkl_intel_ilp64 alias from MKL::MKL_DPCPP for Torch compatibility")
        endif()
      endif()

      # 确保MKL_LIBRARIES变量包含正确的目标引用
      if(TARGET MKL::mkl_intel_ilp64)
        set(MKL_LIBRARIES "MKL::mkl_intel_ilp64" CACHE STRING "MKL libraries" FORCE)
      elseif(TARGET MKL::MKL)
        set(MKL_LIBRARIES "MKL::MKL" CACHE STRING "MKL libraries" FORCE)
      endif()

      message(STATUS "Using official MKL CMake configuration has successfully")
      return()
    endif()
  endif()
endif()

# ====================== 第一步：优先查找系统安装的 MKL ======================
# 1. 尝试通过 Intel oneAPI 环境变量查找
if(DEFINED ENV{ONEAPI_ROOT})
  # 规范化路径分隔符
  file(TO_CMAKE_PATH "$ENV{ONEAPI_ROOT}" _ONEAPI_ROOT_NORMALIZED)
  set(_ONEAPI_MKL_ROOT "${_ONEAPI_ROOT_NORMALIZED}/mkl")

  if(EXISTS "${_ONEAPI_MKL_ROOT}/latest/include/mkl.h")
    set(MKL_ROOT_DIR "${_ONEAPI_MKL_ROOT}" CACHE PATH "Root directory of MKL" FORCE)
    set(MKL_INCLUDE_DIR "${_ONEAPI_MKL_ROOT}/latest/include" CACHE PATH "MKL include directory" FORCE)

    # 根据系统架构选择库目录
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
      set(_MKL_LIB_DIR "${_ONEAPI_MKL_ROOT}/latest/lib/intel64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      set(_MKL_LIB_DIR "${_ONEAPI_MKL_ROOT}/latest/lib/arm64")
    endif()

    # 查找静态/动态库
    if(MKL_USE_STATIC OR NOT DEFINED MKL_USE_STATIC)
      # 在Windows上，静态库是.lib文件；在Unix-like系统上是.a文件
      if(WIN32)
        find_library(MKL_CORE_LIB NAMES mkl_core.lib mkl_core_static.lib PATHS ${_MKL_LIB_DIR})
        find_library(MKL_SEQ_LIB NAMES mkl_sequential.lib mkl_sequential_static.lib PATHS ${_MKL_LIB_DIR})
        find_library(MKL_THREAD_LIB NAMES mkl_intel_thread.lib mkl_intel_thread_static.lib PATHS ${_MKL_LIB_DIR})
      else()
        find_library(MKL_CORE_LIB NAMES mkl_core.a PATHS ${_MKL_LIB_DIR})
        find_library(MKL_SEQ_LIB NAMES mkl_sequential.a PATHS ${_MKL_LIB_DIR})
        find_library(MKL_THREAD_LIB NAMES mkl_intel_thread.a PATHS ${_MKL_LIB_DIR})
      endif()
    else()
      find_library(MKL_CORE_LIB NAMES mkl_core.so mkl_core.dylib mkl_core.lib PATHS ${_MKL_LIB_DIR})
      find_library(MKL_SEQ_LIB NAMES mkl_sequential.so mkl_sequential.dylib mkl_sequential.lib PATHS ${_MKL_LIB_DIR})
      find_library(MKL_THREAD_LIB NAMES mkl_intel_thread.so mkl_intel_thread.dylib mkl_intel_thread.lib PATHS ${_MKL_LIB_DIR})
    endif()

    # 组装库列表
    if(MKL_CORE_LIB AND MKL_SEQ_LIB)
      set(MKL_LIBRARIES
        ${MKL_THREAD_LIB} ${MKL_SEQ_LIB} ${MKL_CORE_LIB}
        CACHE STRING "MKL libraries" FORCE
      )
      set(MKL_FOUND TRUE CACHE BOOL "Whether MKL was found" FORCE)

      # 提取版本（从 oneAPI 版本文件）
      if(EXISTS "${_ONEAPI_MKL_ROOT}/version.txt")
        file(STRINGS "${_ONEAPI_MKL_ROOT}/version.txt" _mkl_version_line REGEX "Version: [0-9.]+")

        if(_mkl_version_line)
          string(REGEX REPLACE "Version: ([0-9.]+)" "\\1" MKL_VERSION "${_mkl_version_line}")
          set(MKL_VERSION "${MKL_VERSION}" CACHE STRING "MKL version" FORCE)
        endif()
      endif()

      if(NOT MKL_FIND_QUIETLY)
        message(STATUS "Found MKL (Intel oneAPI): ${MKL_ROOT_DIR} (version: ${MKL_VERSION})")
      endif()
      return()
    endif()
  endif()
endif()

# 2. 尝试通过 pkg-config 查找系统 MKL
find_package(PkgConfig QUIET)
pkg_check_modules(PC_MKL QUIET mkl)

if(PC_MKL_FOUND)
  # 创建标准别名 target
  add_library(MKL::MKL INTERFACE IMPORTED)
  target_link_libraries(MKL::MKL INTERFACE ${PC_MKL_LIBRARIES})
  target_include_directories(MKL::MKL INTERFACE ${PC_MKL_INCLUDE_DIRS})

  # 更新全局变量
  set(MKL_ROOT_DIR "${PC_MKL_PREFIX}" CACHE PATH "Root directory of MKL" FORCE)
  set(MKL_INCLUDE_DIR "${PC_MKL_INCLUDE_DIRS}" CACHE PATH "MKL include directory" FORCE)
  set(MKL_LIBRARIES "${PC_MKL_LIBRARIES}" CACHE STRING "MKL libraries" FORCE)
  set(MKL_VERSION "${PC_MKL_VERSION}" CACHE STRING "MKL version" FORCE)
  set(MKL_FOUND TRUE CACHE BOOL "Whether MKL was found" FORCE)

  if(NOT MKL_FIND_QUIETLY)
    message(STATUS "Found MKL (system pkg-config): ${MKL_ROOT_DIR} (version: ${MKL_VERSION})")
  endif()
  return()
endif()

# ====================== 第二步：下载/使用本地 MKL 源码 ======================
# 检查本地目录是否存在
if(NOT EXISTS "${MKL_ROOT_DIR}")
  # 引入 FetchContent 下载 MKL (Intel oneAPI MKL open source)
  include(FetchContent)

  # 声明 MKL 依赖（使用官方开源仓库）
  FetchContent_Declare(
    mkl
    GIT_REPOSITORY https://github.com/oneapi-src/oneMKL.git
    GIT_TAG v2025.0.0 # 稳定版本，可按需修改
    GIT_SHALLOW ON # 浅克隆加速下载
    SOURCE_DIR ${MKL_ROOT_DIR} # 指定下载路径
  )

  # 下载源码（不自动配置，避免提前构建）
  FetchContent_GetProperties(mkl)

  if(NOT mkl_POPULATED)
    FetchContent_Populate(mkl)
    set(MKL_ROOT_DIR "${mkl_SOURCE_DIR}" CACHE PATH "Root directory of MKL" FORCE)
  endif()
endif()

# 检查 MKL 核心头文件
if(EXISTS "${MKL_ROOT_DIR}/include/mkl.h" OR EXISTS "${MKL_ROOT_DIR}/src/include/mkl.h")
  # 修正 include 路径（适配开源版 MKL 目录结构）
  if(EXISTS "${MKL_ROOT_DIR}/src/include/mkl.h")
    set(MKL_INCLUDE_DIR "${MKL_ROOT_DIR}/src/include" CACHE PATH "MKL include directory" FORCE)
  else()
    set(MKL_INCLUDE_DIR "${MKL_ROOT_DIR}/include" CACHE PATH "MKL include directory" FORCE)
  endif()

  set(MKL_FOUND TRUE CACHE BOOL "Whether MKL was found" FORCE)

  # 提取版本信息（从 CMakeLists.txt）
  if(EXISTS "${MKL_ROOT_DIR}/CMakeLists.txt")
    file(STRINGS "${MKL_ROOT_DIR}/CMakeLists.txt" _mkl_version_line REGEX "project\\(oneMKL VERSION [0-9.]+\\)")

    if(_mkl_version_line)
      string(REGEX REPLACE "project\\(oneMKL VERSION ([0-9.]+)\\)" "\\1" _mkl_version "${_mkl_version_line}")
      set(MKL_VERSION "${_mkl_version}" CACHE STRING "MKL version" FORCE)
    endif()
  endif()
else()
  # 未找到头文件
  set(MKL_FOUND FALSE CACHE BOOL "Whether MKL was found" FORCE)

  if(MKL_FIND_REQUIRED)
    message(FATAL_ERROR "MKL headers not found in ${MKL_ROOT_DIR}")
  endif()
endif()

# ====================== 处理 find_package 参数 ======================
if(MKL_FIND_REQUIRED AND NOT MKL_FOUND)
  message(FATAL_ERROR "MKL not found (required by project)")
endif()

if(NOT MKL_FIND_QUIETLY)
  if(MKL_FOUND)
    message(STATUS "Found MKL: ${MKL_ROOT_DIR}")
    message(STATUS "  MKL version: ${MKL_VERSION}")
    message(STATUS "  MKL headers: ${MKL_INCLUDE_DIR}")
  else()
    message(STATUS "MKL not found")
  endif()
endif()

# ====================== 提供添加子目录的函数 ======================
function(add_mkl_subdirectory)
  if(NOT TARGET MKL::MKL)
    if(MKL_FOUND)
      message(STATUS "Adding MKL subdirectory from ${MKL_ROOT_DIR}")

      # 保存当前构建配置
      set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

      # 设置 MKL 构建选项
      set(MKL_BUILD_STATIC ON CACHE BOOL "Build static MKL library" FORCE)
      set(MKL_BUILD_SHARED OFF CACHE BOOL "Build shared MKL library" FORCE)
      set(MKL_BUILD_TESTS OFF CACHE BOOL "Build MKL tests" FORCE)
      set(MKL_BUILD_EXAMPLES OFF CACHE BOOL "Build MKL examples" FORCE)

      # 应用自定义配置项
      if(DEFINED MKL_USE_STATIC)
        set(MKL_BUILD_STATIC ${MKL_USE_STATIC} CACHE BOOL "Build static MKL library" FORCE)
        set(MKL_BUILD_SHARED $<NOT:${MKL_USE_STATIC}> CACHE BOOL "Build shared MKL library" FORCE)
      endif()

      if(DEFINED MKL_PARALLEL)
        set(MKL_ENABLE_PARALLEL ${MKL_PARALLEL} CACHE BOOL "Enable MKL parallelism" FORCE)
      else()
        set(MKL_ENABLE_PARALLEL ON CACHE BOOL "Enable MKL parallelism" FORCE)
      endif()

      # 添加 MKL 子目录（适配开源版目录结构）
      if(EXISTS "${MKL_ROOT_DIR}/src/CMakeLists.txt")
        add_subdirectory("${MKL_ROOT_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/mkl" EXCLUDE_FROM_ALL)
      else()
        add_subdirectory("${MKL_ROOT_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/mkl" EXCLUDE_FROM_ALL)
      endif()

      # 恢复 BUILD_SHARED_LIBS
      set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)

      # 创建标准别名 target
      if(TARGET mkl_static)
        set(MKL_LIBRARIES mkl_static CACHE STRING "MKL libraries" FORCE)
        add_library(MKL::MKL ALIAS mkl_static)
      elseif(TARGET mkl_shared)
        set(MKL_LIBRARIES mkl_shared CACHE STRING "MKL libraries" FORCE)
        add_library(MKL::MKL ALIAS mkl_shared)
      endif()
    else()
      message(WARNING "MKL not found, cannot add subdirectory")
    endif()
  else()
    message(STATUS "MKL already added as subdirectory")
  endif()
endfunction()

# 兼容旧函数名
function(add_mkl)
  add_mkl_subdirectory()
endfunction()

# ====================== 提供链接MKL的函数 ======================
function(target_link_mkl target_name)
  if(NOT TARGET ${target_name})
    message(WARNING "Target ${target_name} does not exist, cannot link MKL")
    return()
  endif()

  if(MKL_FOUND)
    if(TARGET MKL::MKL)
      target_link_libraries(${target_name} PRIVATE MKL::MKL)
      message(STATUS "Linked MKL::MKL to target ${target_name}")
    elseif(MKL_LIBRARIES)
      target_link_libraries(${target_name} PRIVATE ${MKL_LIBRARIES})
      target_include_directories(${target_name} PRIVATE ${MKL_INCLUDE_DIR})
      message(STATUS "Linked MKL libraries to target ${target_name}")
    else()
      message(WARNING "MKL found but no libraries available to link")
    endif()
  else()
    message(WARNING "MKL not found, cannot link to target ${target_name}")
  endif()
endfunction()

# ====================== 设置MKL接口和线程变量 ======================
# 这些变量用于Dependencies.cmake中的消息输出
if(NOT DEFINED MKL_INTERFACE)
  set(MKL_INTERFACE "unknown" CACHE STRING "MKL interface type" FORCE)
endif()

if(NOT DEFINED MKL_THREADING)
  set(MKL_THREADING "unknown" CACHE STRING "MKL threading type" FORCE)
endif()

# 根据找到的库推断接口和线程类型
if(MKL_FOUND)
  # 检查是否找到线程库
  if(MKL_THREAD_LIB)
    set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading type" FORCE)
  elseif(MKL_SEQ_LIB)
    set(MKL_THREADING "sequential" CACHE STRING "MKL threading type" FORCE)
  endif()

  # 检查接口类型（简化检测）
  if(MKL_CORE_LIB)
    if(MKL_CORE_LIB MATCHES "mkl_core")
      set(MKL_INTERFACE "lp64" CACHE STRING "MKL interface type" FORCE)
    endif()
  endif()
endif()
