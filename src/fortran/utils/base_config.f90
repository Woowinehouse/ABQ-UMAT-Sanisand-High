!DIR$ FREEFORM
!*******************************************************************************
!> @brief Base configuration module
!>
!> @details
!> This module defines fundamental constants, precision types, and
!> configuration parameters used throughout the UMAT implementation.
!> It provides standardized precision types (DP, SP), integer types (I4, I8),
!> mathematical constants (PI, EPS), and physical constants (PA).
!> The module also defines tensor-related constants and utility parameters.
!>
!> @author wuwenhao
!> @date 2025/11/17
!*******************************************************************************
module Base_config
  use iso_c_binding, only: c_double, c_float, c_int32_t, c_int64_t, c_int
  use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64, int8
  implicit none
  private
  public :: PA, PI, EPS, DELTA, ZERO, ONE, TWO, THREE, FOUR
  public :: DP, SP, I4, I32, I64
  public :: TENSNO, MAX_INDEX, MAX_DATA
  !> Precision type definitions
  integer, parameter :: DP = real64 !< Double precision (64-bit real)
  integer, parameter :: SP = real32 !< Single precision (32-bit real)
  integer, parameter :: I4 = int32  !< 32-bit integer
  integer, parameter :: I32 = int32  !< 32-bit integer
  integer, parameter :: I64 = int64  !< 64-bit integer
  !> Maximum value constants
  integer, parameter :: MAX_INDEX = huge(I32) !< Maximum value for 32-bit integer
  integer, parameter :: MAX_DATA = huge(DP)  !< Maximum value for double precision

  !> Fundamental mathematical and physical constants
  real(DP), parameter :: PA = 101.325d0   !< Atmospheric pressure (kPa)
  real(DP), parameter :: PI = 3.1415927d0 !< Pi constant
  real(DP), parameter :: EPS = 1.0D-12     !< Small tolerance value for comparisons
  real(DP), parameter :: ZERO = 0.0d0     !< Zero constant
  real(DP), parameter :: ONE = 1.0d0      !< One constant
  real(DP), parameter :: TWO = 2.0d0      !< Two constant
  real(DP), parameter :: THREE = 3.0d0   !< Three constant
  real(DP), parameter :: FOUR = 4.0d0     !< Four constant
  !> Kronecker delta tensor (identity matrix for 3D)
  real(DP), parameter :: DELTA(3, 3) = reshape( &
                         [1.0d0, 0.0d0, 0.0d0, &
                          0.0d0, 1.0d0, 0.0d0, &
                          0.0d0, 0.0d0, 1.0d0],[3, 3]) !< 3x3 identity matrix

  !> Small value for tensor operations (tolerance for near-zero tensors)
  real(DP), parameter :: TENSNO = 1.0D-2 !< Tensor near-zero threshold
contains

endmodule Base_config ! module base_config
