# - Try to find LibTorch (PyTorch C++ API)
#
# 优先级查找策略:
# 1. Official TorchConfig.cmake (推荐)
# 2. System pkg-config
# 3. Manual search in common paths
#
# 输出变量：
# LIBTORCH_FOUND
# LIBTORCH_VERSION
# LIBTORCH_ROOT_DIR
# LIBTORCH_INCLUDE_DIRS
# LIBTORCH_CXX_FLAGS      <-- 已导出
# TORCH_CXX_FLAGS         <-- 已导出（官方同名）
# LibTorch::LibTorch      <-- 自动携带 flags + includes + libraries
#
# 输入：
# USER_LIBTORCH_ROOT / LIBTORCH_ROOT

cmake_minimum_required(VERSION 3.18)

# -------------------------------------------------------------------------
# 避免重复查找
# -------------------------------------------------------------------------
if(LIBTORCH_FOUND AND TARGET LibTorch::LibTorch)
  if(NOT LIBTORCH_FIND_QUIETLY)
    message(STATUS "[LibTorch] Already found: ${LIBTORCH_ROOT_DIR}")
  endif()
  return()
endif()

include(FindPackageHandleStandardArgs)

# 内部临时变量
set(_LIBTORCH_FOUND FALSE)
set(_LIBTORCH_VERSION "unknown")
set(_LIBTORCH_ROOT "")
set(_LIBTORCH_INCLUDES "")
set(_LIBTORCH_CXX_FLAGS "")
set(_TORCH_CXX_FLAGS "")

# -------------------------------------------------------------------------
# 1. 官方 TorchConfig.cmake（最高优先级）
# -------------------------------------------------------------------------
if(NOT _LIBTORCH_FOUND)
  message(STATUS "[LibTorch] Try official TorchConfig...")

  set(_HINTS "")

  if(DEFINED USER_LIBTORCH_ROOT)
    file(TO_CMAKE_PATH "${USER_LIBTORCH_ROOT}" _NORM)
    list(APPEND _HINTS ${_NORM})
  endif()

  if(DEFINED ENV{LIBTORCH_ROOT})
    file(TO_CMAKE_PATH "$ENV{LIBTORCH_ROOT}" _NORM)
    list(APPEND _HINTS ${_NORM})
  endif()

  set(_OLD_PREFIX ${CMAKE_PREFIX_PATH})
  list(APPEND CMAKE_PREFIX_PATH ${_HINTS})

  find_package(Torch QUIET CONFIG NO_DEFAULT_PATH)

  if(NOT Torch_FOUND)
    find_package(Torch QUIET CONFIG)
  endif()

  set(CMAKE_PREFIX_PATH ${_OLD_PREFIX})

  if(TARGET torch)
    set(_LIBTORCH_FOUND TRUE)
    set(_LIBTORCH_VERSION ${Torch_VERSION})
    get_filename_component(_LIBTORCH_ROOT ${Torch_DIR}/../.. ABSOLUTE)
    get_target_property(_LIBTORCH_INCLUDES torch INTERFACE_INCLUDE_DIRECTORIES)

    # ====================== 关键：导出官方 TORCH_CXX_FLAGS ======================
    if(DEFINED TORCH_CXX_FLAGS)
      set(_TORCH_CXX_FLAGS ${TORCH_CXX_FLAGS})
      set(_LIBTORCH_CXX_FLAGS ${TORCH_CXX_FLAGS})
    endif()

    if(NOT TARGET LibTorch::LibTorch)
      add_library(LibTorch::LibTorch INTERFACE IMPORTED GLOBAL)
      target_link_libraries(LibTorch::LibTorch INTERFACE torch)
      target_include_directories(LibTorch::LibTorch INTERFACE ${_LIBTORCH_INCLUDES})

      # 自动绑定 CXX flags
      if(_TORCH_CXX_FLAGS)
        target_compile_options(LibTorch::LibTorch INTERFACE $<BUILD_INTERFACE:${_TORCH_CXX_FLAGS}>)
      endif()
    endif()
  endif()
endif()

# -------------------------------------------------------------------------
# 2. pkg-config
# -------------------------------------------------------------------------
if(NOT _LIBTORCH_FOUND)
  find_package(PkgConfig QUIET)

  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_TORCH QUIET torch torchcpu)

    if(PC_TORCH_FOUND)
      set(_LIBTORCH_FOUND TRUE)
      set(_LIBTORCH_VERSION ${PC_TORCH_VERSION})
      set(_LIBTORCH_ROOT ${PC_TORCH_PREFIX})
      set(_LIBTORCH_INCLUDES ${PC_TORCH_INCLUDE_DIRS})
      set(_LIBTORCH_CXX_FLAGS ${PC_TORCH_CFLAGS_OTHER})
      set(_TORCH_CXX_FLAGS ${PC_TORCH_CFLAGS_OTHER})

      if(NOT TARGET LibTorch::LibTorch)
        add_library(LibTorch::LibTorch INTERFACE IMPORTED GLOBAL)
        target_include_directories(LibTorch::LibTorch INTERFACE ${_LIBTORCH_INCLUDES})
        target_link_libraries(LibTorch::LibTorch INTERFACE ${PC_TORCH_LIBRARIES})
        target_compile_options(LibTorch::LibTorch INTERFACE ${_LIBTORCH_CXX_FLAGS})
      endif()
    endif()
  endif()
endif()

# -------------------------------------------------------------------------
# 3. 手动查找预编译库
# -------------------------------------------------------------------------
if(NOT _LIBTORCH_FOUND)
  message(STATUS "[LibTorch] Try common paths...")

  set(_PATHS "")

  if(WIN32)
    list(APPEND _PATHS C:/libtorch D:/libtorch "$ENV{USERPROFILE}/libtorch")
  else()
    list(APPEND _PATHS /usr/local/libtorch /opt/libtorch "$ENV{HOME}/libtorch")
  endif()

  foreach(_P ${_PATHS})
    if(EXISTS "${_P}/include/torch/torch.h")
      set(_INC "${_P}/include")
      find_library(LIB_TORCH NAMES torch libtorch PATHS ${_P}/lib NO_DEFAULT_PATH)
      find_library(LIB_TORCH_CPU NAMES torch_cpu libtorch_cpu PATHS ${_P}/lib NO_DEFAULT_PATH)

      if(LIB_TORCH AND LIB_TORCH_CPU)
        set(_LIBTORCH_FOUND TRUE)
        set(_LIBTORCH_ROOT ${_P})
        set(_LIBTORCH_INCLUDES ${_INC})

        # 手动查找也自动设置 C++17 等标准（与官方一致）
        set(_CXX_FLAGS "-std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1")
        set(_LIBTORCH_CXX_FLAGS ${_CXX_FLAGS})
        set(_TORCH_CXX_FLAGS ${_CXX_FLAGS})

        if(NOT TARGET LibTorch::LibTorch)
          add_library(LibTorch::LibTorch INTERFACE IMPORTED GLOBAL)
          target_include_directories(LibTorch::LibTorch INTERFACE ${_INC})
          target_link_libraries(LibTorch::LibTorch INTERFACE ${LIB_TORCH} ${LIB_TORCH_CPU})
          target_compile_options(LibTorch::LibTorch INTERFACE ${_CXX_FLAGS})
        endif()
        break()
      endif()
    endif()
  endforeach()
endif()

# -------------------------------------------------------------------------
# 最终导出变量
# -------------------------------------------------------------------------
set(LIBTORCH_FOUND ${_LIBTORCH_FOUND} CACHE BOOL "LibTorch found" FORCE)
set(LIBTORCH_VERSION ${_LIBTORCH_VERSION} CACHE STRING "LibTorch version" FORCE)
set(LIBTORCH_ROOT_DIR ${_LIBTORCH_ROOT} CACHE PATH "LibTorch root" FORCE)
set(LIBTORCH_INCLUDE_DIRS ${_LIBTORCH_INCLUDES} CACHE STRING "LibTorch includes" FORCE)

# ====================== 关键：导出 TORCH_CXX_FLAGS ======================
set(TORCH_CXX_FLAGS ${_TORCH_CXX_FLAGS} CACHE STRING "Official Torch CXX flags" FORCE)
set(LIBTORCH_CXX_FLAGS ${_LIBTORCH_CXX_FLAGS} CACHE STRING "LibTorch CXX flags" FORCE)

find_package_handle_standard_args(LibTorch
  REQUIRED_VARS LIBTORCH_ROOT_DIR LIBTORCH_INCLUDE_DIRS
  VERSION_VAR LIBTORCH_VERSION
)

mark_as_advanced(
  LIBTORCH_FOUND
  LIBTORCH_VERSION
  LIBTORCH_ROOT_DIR
  LIBTORCH_INCLUDE_DIRS
  LIBTORCH_CXX_FLAGS
  TORCH_CXX_FLAGS
)