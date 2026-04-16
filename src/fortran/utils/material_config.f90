!*****************************************************************************
!> @brief Material configuration module
!>
!> @details This module defines material constants for the constitutive model
!> including the following parameters:
!> (1) G0 : Shear modulus constant
!> (2) NU : Poissons ratio
!> (3) C  : Ratio between critical state stress ratio in triaxial extension
!>          M_e and that in triaxial compression M_c
!> @author wuwenhao
!> @date 2025/10/24
!*****************************************************************************
module Material_config
  use Base_config, only: DP
  implicit none
  public :: PARAM
  private
  !---------------------------------------------------------------------------
  !> @brief Parameter container type
  !>
  !> @details Derived type containing all material parameters for the
  !> constitutive model. Includes elastic, plastic, and hardening parameters.
  !>
  !> @author wuwenhao
  !> @date 2025/11/27
  !---------------------------------------------------------------------------
  type param_
    real(DP) :: K
    real(DP) :: FM, FN
    real(DP) :: ALPHAC, C, W
    real(DP) :: VOIDREF, LAMDACS, KSI, VOIDRL
    real(DP) :: LAMDAR, BETA
    real(DP) :: Y, PE, PR
    real(DP) :: NB, H0, CH, PS
    real(DP) :: nu_min, nu_max, nu_v
    real(DP) :: Nd, Ad, Pd, X, V
  endtype param_
  ! initial type
  type(param_), parameter :: PARAM = param_ &
                             (K=0.15d0, FM=0.05d0, FN=20d0, &
                              ALPHAC=1.2d0, C=0.712d0, W=-0.25d0, &
                              VOIDREF=0.934d0, LAMDACS=0.0268d0, KSI=0.7d0, VOIDRL=0.15d0, &
                              LAMDAR=0.012d0, BETA=22d0, Y=0.3d0, PE=1d0, PR=1d0, &
                              NB=1.25d0, H0=26.0d0, CH=13.0d0, PS=0.15d0, &
                              nu_min=0.15d0, nu_max=0.45d0, nu_v=0.5d0, &
                              Nd=3.5d0, Ad=0.6d0, Pd=0.06d0, X=0.35d0, V=1000d0)
  !

contains

endmodule Material_config
