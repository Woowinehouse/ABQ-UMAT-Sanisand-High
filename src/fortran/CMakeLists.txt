# CMakeLists.txt
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# project name
project(Fortran_umat LANGUAGES Fortran)
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
  add_library(${PROJECT_NAME} STATIC) # create static library
endif()

# 2. set fortran standard
if(NOT DEFINED CMAKE_Fortran_STANDARD)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    Fortran_STANDARD 2008
    Fortran_STANDARD_REQUIRED ON
    Fortran_EXTENSIONS OFF
  )
endif()

# 3. Specify the source file
target_sources(${PROJECT_NAME}
  PRIVATE
  utils/base_config.f90
  utils/tensor_opt.f90
  utils/tensor_opt_impl.f90
  utils/exception.f90
  utils/exception_impl.f90
  utils/preprocess.f90
  utils/material_config.f90
  core/container.f90
  core/container_impl.f90
  core/elastic.f90
  core/elastic_impl.f90
  core/plastic.f90
  core/plastic_impl.f90
  core/math.f90
  core/math_impl.f90
  umat/umatImpl.f90
  umat/CppToFortran.f90
)

# Set module output directory explicitly
set_target_properties(${PROJECT_NAME} PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}
  CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# Add include directories
target_include_directories(${PROJECT_NAME}
  PUBLIC ${CMAKE_Fortran_MODULE_DIRECTORY}
)

# 如果本模块依赖其他库（如第三方库或其他模块），在这里链接它们
# 检查MKL是否已经找到，避免重复查找
if(NOT MKL_FOUND)
  find_package(MKL CONFIG REQUIRED)
else()
  message(STATUS "MKL already found, skipping re-find in ${PROJECT_NAME}")
endif()

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
  $<LINK_ONLY:MKL::MKL>)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
  $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(
  ${PROJECT_NAME}
  PRIVATE
  $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
  $<$<AND:$<Fortran_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:
  /free, # Fortran free-form source input
  /c,
  /fpp,
  /extend-source,
  /DABQ_WIN86_64,
  /DABQ_FORTRAN,
  /iface:cref,
  /recursive,
  /Qauto, # <-- important for thread-safety of parallel user subroutines
  /align:array64byte,
  /Qpc64, # set FPU precision to 53 bit significand
  /Qprec-div,
  /Qprec-sqrt, # improve precision of FP divides and sqrt
  /Qfma-, # disable floating point fused multiply-add
  /fp:precise, # floating point model: precise
  /Qimf-arch-consistency:true, # math library consistent results
  /Qfp-speculation:safe, # floating point speculations only when safe
  /Qprotect-parens, # honor parenthesis during expression evaluation
  /Qfp-stack-check, # enable stack overflow protection checks
  /reentrancy:threaded, # important for thread-safety
  /Qinit=zero,/Qinit=arrays, # automatically initialize all arrays to zero
  /Qinit=snan, /Qinit=arrays, # automatically initialize all arrays to SNAN
  /Od # 禁用所有优化
  /Zi # 生成 PDB 调试文件
  /Qmkl
  >
)

if(BUILD_SHARED_LIBS)
  message(STATUS "library : ${PROJECT_NAME}.dll complie finished!")
else()
  message(STATUS "library : ${PROJECT_NAME}.lib complie finished!")
endif()
