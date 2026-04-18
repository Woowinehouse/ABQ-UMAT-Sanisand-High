# src/core/impl/CMakeLists.txt

project(CXX_umat_ LANGUAGES CXX C)
message(STATUS "======================================================================")

if(BUILD_SHARED_LIBS)
  message(STATUS "begin to build shared library : ${PROJECT_NAME}.dll ........")
else()
  message(STATUS "begin to build static library : ${PROJECT_NAME}.lib ........")
endif()

message(STATUS "======================================================================")

# 1. define library(Target)
if(BUILD_SHARED_LIBS)
  add_library(${PROJECT_NAME} SHARED) # create shared library
else()
  add_library(${PROJECT_NAME}_static STATIC) # create STATIClibrary
endif()

# 2. set c_plus_plus standard
set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE 0
)

# 3. Specify the source file
target_sources(${PROJECT_NAME}
  PRIVATE
  impl/ShareVarImpl.cpp
  impl/StateVarImpl.cpp
  impl/TensorOptions_ops.cpp
)

# 4. Header file path
target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cpp>
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/impl>
)

# 5. 编译特性的传递
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# 5. 依赖管理
#
if(USE_MIMALLOC)
  find_package(Mimalloc QUIET)

  if(Mimalloc_FOUND)
    add_mimalloc_subdirectory()
    target_link_libraries(${PROJECT_NAME} PRIVATE mimalloc-static)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_MIMALLOC)
    message(STATUS "Linked mimalloc-static to ${PROJECT_NAME} and added USE_MIMALLOC definition")
  else()
    message(WARNING "mimalloc not found. Memory allocation performance may be suboptimal.")
  endif()
else()
  message(STATUS "Building without mimalloc - skipping mimalloc integration")
endif()

# 检查MKL是否已经找到，避免重复查找
if(NOT MKL_FOUND)
  find_package(MKL CONFIG REQUIRED)
else()
  message(STATUS "MKL already found, skipping re-find in ${PROJECT_NAME}")
endif()

# 检查LibTorch是否已经找到，避免重复查找
if(NOT LibTorch_FOUND)
  find_package(LibTorch QUIET)
else()
  message(STATUS "LibTorch already found, skipping re-find in ${PROJECT_NAME}")
endif()

target_include_directories(${PROJECT_NAME}
  SYSTEM PUBLIC
  ${LIBTORCH_INCLUDE_DIR}
  $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
  LibTorch::LibTorch
  $<LINK_ONLY:MKL::MKL>
)

# On Windows, define DLL export macro
target_compile_definitions(${PROJECT_NAME} PUBLIC
  $<$<CXX_COMPILER_ID:MSVC>: MYUMAT_EXPORTS>
)

# 根据条件动态生成值
target_compile_options(${PROJECT_NAME}
  PUBLIC
  $<$<CXX_COMPILER_ID:MSVC>: /W4 /wd4267 /wd4251 /wd4702>
  PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/gbk>
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Werror>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:
  /MD
  /Od # 禁用所有优化
  /Zi # 生成 PDB 调试文件
  /MDd # 动态链接调试版运行时库
  /RTC1 # 启用栈/未初始化变量检查
  /sdl # 强化安全检查（可选）
  /DDEBUG_BUILD
  >
  $<$<CONFIG:Release>:-DRELEASE_BUILD>
)

# 额外：确保 Debug/Release 运行时库匹配（CMake 推荐方式）
set_target_properties(${PROJECT_NAME} PROPERTIES
  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"

  # 等价于：Debug→/MDd，Release→/MD
)

if(BUILD_SHARED_LIBS)
  message(STATUS "library : ${PROJECT_NAME}.dll complie finished!")
else()
  message(STATUS "library : ${PROJECT_NAME}.lib complie finished!")
endif()
