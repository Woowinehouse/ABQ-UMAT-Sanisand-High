!*****************************************************************************
!> @brief tensor_opt_mod
!>
!> @details Module for tensor operations in continuum mechanics
!>
!> @author wuwenhao
!> @date 2025/11/27
!*****************************************************************************
module tensor_opt_mod
  use Base_config
  implicit none
  public :: operator(.ddot.), operator(.dyad.)
  ! default private
  private
  !---------------------------------------------------------------------------
  !> @brief Torch
  !>
  !> @details Type for tensor operations providing various stress tensor
  !> calculations and utilities
  !>
  !> @author wuwenhao
  !> @date 2025/12/05
  !---------------------------------------------------------------------------
  type, public :: Torch
  contains
    private
    ! static function method
    procedure, public, nopass :: Print => Print_Impl
    procedure, public, nopass :: Trace => Trace_Impl
    procedure, public, nopass :: pressure => pressure_impl
    procedure, public, nopass :: Deviatoric => Deviatoric_impl
    procedure, public, nopass :: Get_J2 => Get_J2_impl
    procedure, public, nopass :: Get_J3 => Get_J3_impl
    procedure, public, nopass :: Get_ratio => Ratio_impl
    procedure, public, nopass :: Sin3theta => Get_sin3t_impl
    procedure, public, nopass :: Shear => Shear_impl
    procedure, public, nopass :: Normalize => normalize_impl
    procedure, public, nopass :: Norm => Norm_impl
    procedure, public, nopass :: Get_cost => Get_cost_impl
    procedure, public, nopass :: Get_Rm => Get_Rm_impl
    procedure, public, nopass :: Get_unit_devivator => Get_unit_devivator_impl
  endtype Torch
  !
  interface operator(.ddot.)
    module procedure tensor4_ddot_tensor2
    module procedure Tensor2_ddot_tensor4
  endinterface
  interface operator(.dyad.)
    module procedure Tensor2_dyad_tensor2
  endinterface
  !=============================================================================
  ! Abstract interface definition (implemented in the sub-module)
  !=============================================================================
  interface
    !***************************************************************************
    !> @brief Tensor4_ddot_tensor2
    !>
    !> @details Calculate the double dot product of a fourth-order tensor and
    !> a second-order tensor
    !> @param[in]  tensor4  Fourth-order tensor
    !> @param[in]  tensor2  Second-order tensor
    !>
    !> @return Resulting second-order tensor
    !***************************************************************************
    module function tensor4_ddot_tensor2(tensor4, tensor2) result(res)
      real(DP), dimension(3, 3, 3, 3), intent(in) :: tensor4
      real(DP), dimension(3, 3), intent(in) :: tensor2
      real(DP), dimension(3, 3) :: res
    endfunction tensor4_ddot_tensor2
    !***************************************************************************
    !> @brief Tensor2_ddot_tensor4
    !>
    !> @details Calculate the double dot product of a second-order tensor and
    !> a fourth-order tensor
    !> @param[in]  tensor2  Second-order tensor
    !> @param[in]  tensor4  Fourth-order tensor
    !>
    !> @return Resulting second-order tensor
    !***************************************************************************
    module function Tensor2_ddot_tensor4(tensor2, tensor4) result(res)
      real(DP), dimension(3, 3, 3, 3), intent(in) :: tensor4
      real(DP), dimension(3, 3), intent(in) :: tensor2
      real(DP), dimension(3, 3) :: res
    endfunction Tensor2_ddot_tensor4
    !***************************************************************************
    !> @brief Dyadic product of two second-order tensors
    !>
    !> @details Calculate the dyadic (outer) product of two second-order
    !> tensors, resulting in a fourth-order tensor.
    !>
    !> @param[in]  tensorA First second-order tensor
    !> @param[in]  tensorB Second second-order tensor
    !>
    !> @return Fourth-order tensor result
    !***************************************************************************
    module function Tensor2_dyad_tensor2(tensorA, tensorB) result(res)
      real(DP), dimension(3, 3), intent(in) :: tensorA
      real(DP), dimension(3, 3), intent(in) :: tensorB
      real(DP), dimension(3, 3, 3, 3) :: res
    endfunction Tensor2_dyad_tensor2
    !***************************************************************************
    !> @brief Print tensor
    !>
    !> @details Print the components of a 3x3 tensor to standard output
    !> in matrix format for debugging and visualization.
    !>
    !> @param[in]  tensor  3x3 tensor to print
    !***************************************************************************
    module Subroutine Print_Impl(tensor)
      real(DP), intent(in), dimension(3, 3) :: tensor
    endSubroutine
    !***************************************************************************
    !> @brief Trace_Impl
    !>
    !> @details Calculate the trace of a stress tensor
    !>
    !> @param[in]  stress  Stress tensor
    !>
    !> @return Trace of the stress tensor
    !***************************************************************************
    module function Trace_Impl(tensor) result(val)
      real(DP), intent(in), dimension(3, 3) :: tensor
      real(DP) :: val
    endfunction Trace_Impl
    !***************************************************************************
    module function pressure_impl(tensor) result(val)
      real(DP), intent(in), dimension(3, 3) :: tensor
      real(DP) :: val
    endfunction pressure_impl
    !***************************************************************************
    !> @brief Sec_dev_invar_impl
    !>
    !> @details Calculate the second deviatoric invariant (J2) of a stress tensor
    !>
    !> @param[in]  stress  Stress tensor
    !>
    !> @return Second deviatoric invariant (J2)
    !***************************************************************************
    module function Get_J2_impl(tensor) result(val)
      real(DP), intent(in), dimension(3, 3) :: tensor
      real(DP) :: val
    endfunction Get_J2_impl
    !***************************************************************************
    !> @brief Trd_dev_invar_impl
    !>
    !> @details Calculate the third deviatoric invariant (J3) of a stress tensor
    !>
    !> @param[in]  stress  Stress tensor
    !>
    !> @return Third deviatoric invariant (J3)
    !***************************************************************************
    module function Get_J3_impl(tensor) result(val)
      real(DP), intent(in), dimension(3, 3) :: tensor
      real(DP) :: val
    endfunction Get_J3_impl
    !***************************************************************************
    !> @brief Deviatoric_impl
    !>
    !> @details Calculate the deviatoric part of a stress tensor
    !>
    !> @param[in]  stress  Stress tensor
    !>
    !> @return Deviatoric stress tensor
    !***************************************************************************
    module function Deviatoric_impl(tensor) result(res)
      real(DP), intent(in), dimension(3, 3) :: tensor
      real(DP), dimension(3, 3) :: res
    endfunction Deviatoric_impl
    !***************************************************************************
    !> @brief Ratio_impl
    !>
    !> @details Calculate the stress ratio tensor (deviatoric stress divided by
    !> mean stress)
    !> @param[in]  stress  Stress tensor
    !>
    !> @return Stress ratio tensor
    !***************************************************************************
    module function Ratio_impl(tensor) result(res)
      real(DP), intent(in), dimension(3, 3) :: tensor
      real(DP), dimension(3, 3) :: res
    endfunction Ratio_impl
    !***************************************************************************
    !> @brief Sin3theta_impl
    !>
    !> @details Calculate sin(3θ) where θ is the Lode angle
    !>
    !> @param[in]  stress  Stress tensor
    !>
    !> @return sin(3θ) value
    !***************************************************************************
    module function Get_sin3t_impl(tensor) result(val)
      real(DP), intent(in), dimension(3, 3) :: tensor
      real(DP) :: val
    endfunction Get_sin3t_impl
    !***************************************************************************
    !> @brief Shear_impl
    !>
    !> @details Calculate the shear stress (sqrt(J2))
    !>
    !> @param[in]  stress  Stress tensor
    !>
    !> @return Shear stress value
    !***************************************************************************
    module function Shear_impl(tensor) result(res)
      real(DP), dimension(3, 3), intent(in) :: tensor
      real(DP) :: res
    endfunction Shear_impl
    !***************************************************************************
    !> @brief Normalize tensor
    !>
    !> @details Normalize a tensor by dividing by its Frobenius norm,
    !> resulting in a unit tensor with the same direction.
    !>
    !> @param[in]  tensor  Input 3x3 tensor
    !>
    !> @return Normalized unit tensor
    !***************************************************************************
    module function Normalize_impl(tensor) result(res)
      real(DP), dimension(3, 3), intent(in) :: tensor
      real(DP), dimension(3, 3) :: res
    endfunction Normalize_impl
    !***************************************************************************
    !> @brief Calculate tensor norm
    !>
    !> @details Calculate the Frobenius norm (Euclidean norm) of a 3x3 tensor.
    !>
    !> @param[in]  tensor  Input 3x3 tensor
    !>
    !> @return Frobenius norm of the tensor
    !***************************************************************************
    module function Norm_impl(tensor) result(res)
      real(DP), dimension(3, 3), intent(in) :: tensor
      real(DP) :: res
    endfunction Norm_impl
    !***************************************************************************
    !> @brief Calculate cosine of angle between tensors
    !>
    !> @details Calculate the cosine of the angle between two tensors
    !> using their double dot product and norms.
    !>
    !> @param[in]  tensorA  First 3x3 tensor
    !> @param[in]  tensorB  Second 3x3 tensor
    !>
    !> @return Cosine of angle between tensors
    !***************************************************************************
    module function Get_cost_impl(tensorA, tensorB) result(val)
      real(DP), dimension(3, 3), intent(in) :: tensorA
      real(DP), dimension(3, 3), intent(in) :: tensorB
      real(DP) :: val
    endfunction Get_cost_impl
    !***************************************************************************
    !> @brief Calculate R_m parameter
    !>
    !> @details Calculate the R_m parameter used in critical state soil
    !> mechanics, related to the Lode angle and stress invariants.
    !>
    !> @param[in]  tensor  Stress tensor
    !>
    !> @return R_m parameter value
    !***************************************************************************
    module function Get_Rm_impl(tensor) result(val)
      real(DP), dimension(3, 3), intent(in) :: tensor
      real(DP) :: val
    endfunction Get_Rm_impl
    !***************************************************************************
    !> @brief Calculate unit deviatoric tensor
    !>
    !> @details Calculate the unit deviatoric tensor by normalizing the
    !> deviatoric part of a stress tensor.
    !>
    !> @param[in]  tensor  Input stress tensor
    !>
    !> @return Unit deviatoric tensor
    !***************************************************************************
    module function Get_unit_devivator_impl(tensor) result(res)
      real(DP), dimension(3, 3), intent(in) :: tensor
      real(DP), dimension(3, 3) :: res
    endfunction Get_unit_devivator_impl
  endinterface
contains

!*******************************************************************************
endmodule tensor_opt_mod
