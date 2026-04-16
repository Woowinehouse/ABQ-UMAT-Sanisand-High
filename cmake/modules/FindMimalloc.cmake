# FindMimalloc.cmake - Find mimalloc memory allocator library
#
# Variables set by this module:
# MIMALLOC_ROOT_DIR - Path to the mimalloc root directory
# MIMALLOC_FOUND - TRUE if mimalloc is found
# MIMALLOC_INCLUDE_DIR - Path to mimalloc include directory
# MIMALLOC_LIBRARIES - List of mimalloc libraries (if pre-built)
# MIMALLOC_VERSION - Version of mimalloc if available
#
# Usage:
# find_package(Mimalloc)
# if(Mimalloc_FOUND)
# add_mimalloc_subdirectory()
# target_link_libraries(my_target mimalloc-static)  # or mimalloc
# endif()
#
# Options:
# MIMALLOC_USE_STATIC - Prefer static library (default: ON)
# MIMALLOC_OVERRIDE - Enable malloc override (default: OFF)
# MIMALLOC_SECURE - Enable secure mode (default: OFF)

cmake_minimum_required(VERSION 3.15)

# 初始化核心变量（全局可见，无需 PARENT_SCOPE）
set(Mimalloc_FOUND FALSE CACHE BOOL "Whether mimalloc was found" FORCE)
set(MIMALLOC_FOUND FALSE CACHE BOOL "Whether mimalloc was found" FORCE)
set(MIMALLOC_VERSION "unknown" CACHE STRING "mimalloc version" FORCE)
set(MIMALLOC_ROOT_DIR "${CMAKE_SOURCE_DIR}/third_party/mimalloc" CACHE PATH "Root directory of mimalloc" FORCE)
set(MIMALLOC_INCLUDE_DIR "" CACHE PATH "mimalloc include directory" FORCE)
set(MIMALLOC_LIBRARIES "" CACHE STRING "mimalloc libraries" FORCE)

# 标记为高级变量（不在 GUI 显示）
mark_as_advanced(
  Mimalloc_FOUND
  MIMALLOC_FOUND
  MIMALLOC_ROOT_DIR
  MIMALLOC_INCLUDE_DIR
  MIMALLOC_LIBRARIES
  MIMALLOC_VERSION
)

# 优先查找系统已安装的 mimalloc
find_package(PkgConfig QUIET)
pkg_check_modules(PC_MIMALLOC QUIET mimalloc)

if(PC_MIMALLOC_FOUND)
  # 使用系统版本
  add_library(Mimalloc::Mimalloc INTERFACE IMPORTED)
  target_link_libraries(Mimalloc::Mimalloc INTERFACE ${PC_MIMALLOC_LIBRARIES})
  target_include_directories(Mimalloc::Mimalloc INTERFACE ${PC_MIMALLOC_INCLUDE_DIRS})

  # 更新全局变量（无需 PARENT_SCOPE）
  set(MIMALLOC_ROOT_DIR "${PC_MIMALLOC_PREFIX}" CACHE PATH "Root directory of mimalloc" FORCE)
  set(MIMALLOC_INCLUDE_DIR "${PC_MIMALLOC_INCLUDE_DIRS}" CACHE PATH "mimalloc include directory" FORCE)
  set(MIMALLOC_LIBRARIES "${PC_MIMALLOC_LIBRARIES}" CACHE STRING "mimalloc libraries" FORCE)
  set(MIMALLOC_VERSION "${PC_MIMALLOC_VERSION}" CACHE STRING "mimalloc version" FORCE)
  set(MIMALLOC_FOUND TRUE CACHE BOOL "Whether mimalloc was found" FORCE)
  set(Mimalloc_FOUND TRUE CACHE BOOL "Whether mimalloc was found" FORCE)

  if(NOT Mimalloc_FIND_QUIETLY)
    message(STATUS "Found mimalloc (system): ${MIMALLOC_ROOT_DIR} (version: ${MIMALLOC_VERSION})")
  endif()
  return()
endif()

# ====================== 处理本地/下载的 mimalloc ======================
# 检查本地目录是否存在
if(NOT EXISTS "${MIMALLOC_ROOT_DIR}")
  # 引入 FetchContent 模块下载 mimalloc
  include(FetchContent)

  # 声明 mimalloc 依赖
  FetchContent_Declare(
    mimalloc
    GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
    GIT_TAG v2.1.6 # 稳定版本
    GIT_SHALLOW ON # 浅克隆加速下载
    SOURCE_DIR ${MIMALLOC_ROOT_DIR} # 指定下载路径
  )

  # 下载并配置（不自动 add_subdirectory）
  FetchContent_GetProperties(mimalloc)

  if(NOT mimalloc_POPULATED)
    FetchContent_Populate(mimalloc)
    set(MIMALLOC_ROOT_DIR "${mimalloc_SOURCE_DIR}" CACHE PATH "Root directory of mimalloc" FORCE)
  endif()
endif()

# 检查 mimalloc 核心头文件
if(EXISTS "${MIMALLOC_ROOT_DIR}/include/mimalloc.h")
  # 更新全局变量
  set(MIMALLOC_INCLUDE_DIR "${MIMALLOC_ROOT_DIR}/include" CACHE PATH "mimalloc include directory" FORCE)
  set(MIMALLOC_FOUND TRUE CACHE BOOL "Whether mimalloc was found" FORCE)
  set(Mimalloc_FOUND TRUE CACHE BOOL "Whether mimalloc was found" FORCE)

  # 提取版本信息
  if(EXISTS "${MIMALLOC_ROOT_DIR}/CMakeLists.txt")
    file(STRINGS "${MIMALLOC_ROOT_DIR}/CMakeLists.txt" _mi_version_line REGEX "project\\(mimalloc VERSION [0-9.]+\\)")

    if(_mi_version_line)
      string(REGEX REPLACE "project\\(mimalloc VERSION ([0-9.]+)\\)" "\\1" _mi_version "${_mi_version_line}")
      set(MIMALLOC_VERSION "${_mi_version}" CACHE STRING "mimalloc version" FORCE)
    endif()
  endif()
else()
  # 未找到头文件
  set(MIMALLOC_FOUND FALSE CACHE BOOL "Whether mimalloc was found" FORCE)
  set(Mimalloc_FOUND FALSE CACHE BOOL "Whether mimalloc was found" FORCE)

  if(Mimalloc_FIND_REQUIRED)
    message(FATAL_ERROR "mimalloc headers not found in ${MIMALLOC_ROOT_DIR}")
  endif()
endif()

# ====================== 处理 find_package 参数 ======================
if(Mimalloc_FIND_REQUIRED AND NOT Mimalloc_FOUND)
  message(FATAL_ERROR "mimalloc not found (required by project)")
endif()

if(NOT Mimalloc_FIND_QUIETLY)
  if(Mimalloc_FOUND)
    message(STATUS "Found mimalloc: ${MIMALLOC_ROOT_DIR}")
    message(STATUS "  mimalloc version: ${MIMALLOC_VERSION}")
    message(STATUS "  mimalloc headers: ${MIMALLOC_INCLUDE_DIR}")
  else()
    message(STATUS "mimalloc not found")
  endif()
endif()

# ====================== 提供添加子目录的函数 ======================
function(add_mimalloc_subdirectory)
  if(NOT TARGET mimalloc AND NOT TARGET mimalloc-static)
    if(MIMALLOC_FOUND)
      message(STATUS "Adding mimalloc subdirectory from ${MIMALLOC_ROOT_DIR}")

      # 保存当前 BUILD_SHARED_LIBS 配置
      set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

      # 设置 mimalloc 构建选项
      set(MI_BUILD_STATIC ON CACHE BOOL "Build static library" FORCE)
      set(MI_BUILD_SHARED OFF CACHE BOOL "Build shared library" FORCE)
      set(MI_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
      set(MI_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)

      # 应用自定义配置项
      if(DEFINED MIMALLOC_USE_STATIC)
        set(MI_BUILD_STATIC ${MIMALLOC_USE_STATIC} CACHE BOOL "Build static library" FORCE)

        if(${MIMALLOC_USE_STATIC})
          set(MI_BUILD_SHARED OFF CACHE BOOL "Build shared library" FORCE)
        else()
          set(MI_BUILD_SHARED ON CACHE BOOL "Build shared library" FORCE)
        endif()
      endif()

      if(DEFINED MIMALLOC_OVERRIDE)
        set(MI_OVERRIDE ${MIMALLOC_OVERRIDE} CACHE BOOL "Override malloc" FORCE)
      else()
        set(MI_OVERRIDE OFF CACHE BOOL "Override malloc" FORCE)
      endif()

      if(DEFINED MIMALLOC_SECURE)
        set(MI_SECURE ${MIMALLOC_SECURE} CACHE BOOL "Secure mode" FORCE)
      endif()

      # 添加子目录
      add_subdirectory("${MIMALLOC_ROOT_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/mimalloc" EXCLUDE_FROM_ALL)

      # 恢复 BUILD_SHARED_LIBS
      set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)

      # 更新库变量
      if(TARGET mimalloc-static)
        set(MIMALLOC_LIBRARIES mimalloc-static CACHE STRING "mimalloc libraries" FORCE)
        add_library(Mimalloc::Mimalloc ALIAS mimalloc-static)
      elseif(TARGET mimalloc)
        set(MIMALLOC_LIBRARIES mimalloc CACHE STRING "mimalloc libraries" FORCE)
        add_library(Mimalloc::Mimalloc ALIAS mimalloc)
      endif()
    else()
      message(WARNING "mimalloc not found, cannot add subdirectory")
    endif()
  else()
    message(STATUS "mimalloc already added as subdirectory")
  endif()
endfunction()

# 兼容旧函数名
function(add_mimalloc)
  add_mimalloc_subdirectory()
endfunction()
