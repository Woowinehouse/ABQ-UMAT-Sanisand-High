# CMakeLists.txt

# project name
project(CXX_umat
  LANGUAGES CXX C Fortran)

message(STATUS "======================================================================")

if(BUILD_SHARED_LIBS)
  message(STATUS "begin to build shared library : ${PROJECT_NAME}.dll ........")
else()
  message(STATUS "begin to build static library : ${PROJECT_NAME}.lib ........")
endif()

message(STATUS "======================================================================")

include(FortranCInterface)
FortranCInterface_VERIFY(CXX)

# 1. define library(Target)
if(BUILD_SHARED_LIBS)
  add_library(${PROJECT_NAME} SHARED) # create shared library
else()
  add_library(${PROJECT_NAME}_static STATIC) # create static library
endif()

# 2. Specify the source file
target_sources(${PROJECT_NAME}
  PRIVATE
  ops/Elastic_ops.cpp
  umat/math.cpp
  umat/UmatImpl.cpp
)

# 3. Header file path
target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# 4. 编译特性的传递
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# 检查LibTorch是否已经找到，避免重复查找
if(NOT LibTorch_FOUND)
  find_package(LibTorch QUIET)
else()
  message(STATUS "LibTorch already found, skipping re-find in ${PROJECT_NAME}")
endif()

find_package(OpenMP REQUIRED)

# 如果本模块依赖其他库（如第三方库或其他模块），在这里链接它们
target_link_libraries(${PROJECT_NAME}
  PUBLIC
  CXX_umat_core
  Fortran_umat
)

# On Windows, define DLL export macro
target_compile_definitions(${PROJECT_NAME} PUBLIC
  $<$<CXX_COMPILER_ID:MSVC>: MYUMAT_EXPORTS>
  $<$<NOT:$<BOOL:${BUILD_TESTING}>>:MYUTILS_BUILD_LIBRARY>
)

# 根据条件动态生成值
# 根据条件动态生成值
target_compile_options(${PROJECT_NAME}
  PUBLIC
  $<$<CXX_COMPILER_ID:MSVC>: /W4 /wd4267 /wd4251 /wd4702>
  PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Werror>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:
  /Od # 禁用所有优化
  /Zi # 生成 PDB 调试文件
  /MDd # 动态链接调试版运行时库
  /RTC1 # 启用栈/未初始化变量检查
  /sdl # 强化安全检查（可选）
  /DDEBUG_BUILD
  >
  $<$<CONFIG:Release>:-DRELEASE_BUILD>
)

if(BUILD_SHARED_LIBS)
  message(STATUS "library : ${PROJECT_NAME}.dll complie finished!")
else()
  message(STATUS "library : ${PROJECT_NAME}.lib complie finished!")
endif()
