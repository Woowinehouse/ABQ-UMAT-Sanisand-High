!*******************************************************************************
!> @brief Math module for numerical algorithms
!>
!> @details This module provides mathematical utilities including bisection method,
!>          interval checking, and other numerical algorithms used in the UMAT.
!>
!> @author wuwenhao
!> @date 2025/12/11
!*******************************************************************************
module math_mod
  use Base_config
  use Container_mod
  implicit none
  private
  !*****************************************************************************
  !> @brief Abstract interface for functions with parameters
  !>
  !> @details This abstract interface defines the signature for functions that
  !>          take shared variables, state variables, and an amplitude parameter,
  !>          returning a double precision value.
  !>
  !> @param[in] shvars    Shared variables (stress tensor, etc.)
  !> @param[in] stvars    State variables (void ratio, etc.)
  !> @param[in] amplitude Amplitude parameter (e.g., scaling factor)
  !> @return fval         Function value
  !*****************************************************************************
  abstract interface
    function func_with_param(shvars, stvars, depsln) result(fval)
      use Base_config
      use Container_mod
      implicit none
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP), dimension(3, 3), intent(in) :: depsln
      real(DP) :: fval
    endfunction func_with_param
  endinterface
  !
  type, public :: Math
  contains
    procedure, public, nopass :: elastic_update => elastic_update_impl
    procedure, public, nopass :: Intchc => intchc_impl
    procedure, public, nopass :: Onyield => Onyield_impl
    procedure, public, nopass :: Bisection_impl
    procedure, public, nopass :: drift_shvars_impl
    procedure, public, nopass :: Get_residual_impl
    procedure, public, nopass :: ftol_with_depsln
    procedure, public, nopass :: mean_with_depsln
    procedure, private, nopass :: is_monotonic
    procedure, private, nopass :: flow_direction_impl
    procedure, private, nopass :: radial_direction_impl
  endtype
  !
  interface
    module subroutine elastic_update_impl(shvars, stvars, depsln, elastiff)
      type(Share_var), intent(inout) :: shvars
      type(State_var), intent(inout) :: stvars
      real(DP), dimension(3, 3), intent(in) :: depsln
      real(DP), dimension(3, 3, 3, 3) :: elastiff
    endsubroutine elastic_update_impl
    !***************************************************************************
    !> @brief Interval checking implementation
    !>
    !> @details This subroutine performs interval checking to find the appropriate
    !>          scaling factor for the strain increment. It determines the right
    !>          boundary and the scaling factor for plastic correction.
    !>
    !> @param[in]  shvars Shared variables (stress tensor, etc.)
    !> @param[in]  stvars State variables (void ratio, etc.)
    !> @param[out] rbd    Right boundary of the scaling factor (0 <= rbd <= 1)
    !> @param[out] alout  Scaling factor for plastic correction
    !***************************************************************************
    module function intchc_impl(shvars, stvars, depsln, tol) result(alout)
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP), dimension(3, 3), intent(in) :: depsln
      real(DP), intent(in) :: tol
      real(DP) :: alout
    endfunction intchc_impl
    !***************************************************************************
    !> @brief Bisection method implementation
    !>
    !> @details This function implements the bisection method to find the root of
    !>          a function within a given interval [lbd, rbd]. The function must
    !>          have opposite signs at the boundaries. Monotonicity check can be
    !>          enabled in debug mode.
    !>
    !> @param[in] shvars   Shared variables (stress tensor, etc.)
    !> @param[in] stvars   State variables (void ratio, etc.)
    !> @param[in] func     Function to find root of (conforms to func_with_param)
    !> @param[in] lbd      Left boundary of the interval
    !> @param[in] rbd      Right boundary of the interval
    !> @param[in] condition Target value (usually 0 for root finding)
    !> @return alout       Root found within the interval
    !***************************************************************************
    module function Bisection_impl(shvars, stvars, depsln, func, lbd, rbd, condition, tol) result(alout)
      ! declration
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP), dimension(3, 3), intent(in) :: depsln
      procedure(func_with_param) :: func
      real(DP), intent(in) :: lbd
      real(DP), intent(in) :: rbd
      real(DP), intent(in) :: condition
      real(DP), intent(in) :: tol
      real(DP) :: alout
    endfunction Bisection_impl
    !***************************************************************************
    !> @brief Calculate residual between shared variable states
    !>
    !> @details This function computes the residual (difference) between three
    !>          shared variable states. It is used in convergence checking and
    !>          iterative algorithms to measure the change between successive
    !>          iterations.
    !>
    !> @param[in] shfor    First shared variable state
    !> @param[in] shsec    Second shared variable state
    !> @param[in] shtmp    Third shared variable state (reference)
    !>
    !> @return residual    Computed residual value
    !***************************************************************************
    module function Get_residual_impl(shfor, shsec, shtmp) result(residual)
      type(Share_var), intent(in) :: shfor
      type(Share_var), intent(in) :: shsec
      type(Share_var), intent(in) :: shtmp
      real(DP) :: residual
    endfunction Get_residual_impl
    !***************************************************************************
    !> @brief Update state variables when on yield surface
    !>
    !> @details This subroutine updates shared and state variables when the
    !>          material is on the yield surface. It performs plastic correction
    !>          and updates the elasticity tensor for consistent tangent modulus.
    !>          The algorithm ensures stress state remains on the yield surface
    !>          while updating hardening parameters and fabric tensors.
    !>
    !> @param[in]  shvars      Input shared variables (stress tensor, etc.)
    !> @param[in]  stvars      Input state variables (void ratio, etc.)
    !> @param[in]  depsln      Strain increment tensor
    !> @param[in]  tol         Tolerance for convergence checking
    !> @param[out] shvar_upd   Updated shared variables
    !> @param[out] stvar_upd   Updated state variables
    !> @param[out] dempx       Updated elasticity tensor (consistent tangent)
    !***************************************************************************
    module subroutine Onyield_impl(shvars, stvars, depsln, tol, dempx, NUMBER, NOEL, NPT)
      type(Share_var), intent(inout) :: shvars
      type(State_var), intent(inout) :: stvars
      real(DP), dimension(3, 3), intent(in) :: depsln
      real(DP), intent(in) :: tol
      real(DP), dimension(3, 3, 3, 3) :: dempx
      integer(I4), intent(in) :: NUMBER, NOEL, NPT
    endsubroutine Onyield_impl
    !***************************************************************************
    !> @brief Drift correction for shared variables
    !>
    !> @details This subroutine performs drift correction to ensure stress state
    !>          remains on the yield surface after plastic correction. It adjusts
    !>          shared and state variables to compensate for numerical drift that
    !>          may occur during iterative solution procedures.
    !>
    !> @param[in]  shtmp   Temporary shared variables (after plastic correction)
    !> @param[in]  sttmp   Temporary state variables (after plastic correction)
    !> @param[in]  tol     Tolerance for drift correction
    !> @param[out] dedrt   Updated elasticity tensor after drift correction
    !***************************************************************************
    module subroutine drift_shvars_impl(shtmp, sttmp, epsilon, NUMBER, NOEL, NPT)
      type(Share_var), intent(inout) :: shtmp
      type(State_var), intent(inout) :: sttmp
      real(DP), intent(in) :: epsilon
      integer(I4), intent(in) :: NUMBER, NOEL, NPT
    endsubroutine drift_shvars_impl
    !***************************************************************************
    !> @brief Yield distance with strain increment
    !>
    !> @details This function calculates the yield distance after applying a
    !>          scaled strain increment. It computes the stress increment from
    !>          the strain increment using the elasticity tensor, updates the
    !>          stress, and returns the yield distance.
    !>
    !> @param[in] shvars    Shared variables (stress tensor, etc.)
    !> @param[in] stvars    State variables (void ratio, etc.)
    !> @param[in] amplitude Scaling factor for the strain increment
    !> @return ftol         Yield distance after applying scaled strain increment
    !***************************************************************************
    module function ftol_with_depsln(shvars, stvars, depsln) result(ftol)
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP), dimension(3, 3), intent(in) :: depsln
      real(DP) :: ftol
    endfunction ftol_with_depsln
    !***************************************************************************
    !> @brief Mean stress with strain increment
    !>
    !> @details This function calculates the mean stress after applying a
    !>          scaled strain increment. It computes the stress increment from
    !>          the strain increment using the elasticity tensor, updates the
    !>          stress, and returns the mean stress (trace/3).
    !>
    !> @param[in] shvars    Shared variables (stress tensor, etc.)
    !> @param[in] stvars    State variables (void ratio, etc.)
    !> @param[in] amplitude Scaling factor for the strain increment
    !> @return res          Mean stress after applying scaled strain increment
    !***************************************************************************
    module function mean_with_depsln(shvars, stvars, depsln) result(res)
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP), dimension(3, 3), intent(in) :: depsln
      real(DP) :: res
    endfunction mean_with_depsln
    !***************************************************************************
    !> @brief Monotonicity check for a function
    !>
    !> @details This function checks whether a given function is monotonic
    !>          (either non-decreasing or non-increasing) within the interval
    !>          [lbd, rbd]. It samples the function at multiple points and
    !>          verifies monotonic behavior.
    !>
    !> @param[in] shvars Shared variables (stress tensor, etc.)
    !> @param[in] stvars State variables (void ratio, etc.)
    !> @param[in] func   Function to check (conforms to func_with_param)
    !> @param[in] lbd    Left boundary of the interval
    !> @param[in] rbd    Right boundary of the interval
    !> @return           .true. if function is monotonic, .false. otherwise
    !***************************************************************************
    module logical function is_monotonic(shvars, stvars, depsln, func, lbd, rbd)
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP), dimension(3, 3), intent(in) :: depsln
      procedure(func_with_param) :: func
      real(DP), intent(in) :: lbd
      real(DP), intent(in) :: rbd
    endfunction is_monotonic
    !***************************************************************************
    !> @brief Calculate plastic flow direction
    !>
    !> @details This function computes the plastic flow direction tensor based on
    !>          the current stress state and material properties. The flow
    !>          direction defines the direction of plastic strain increment in
    !>          stress space, which is essential for associative or non-associative
    !>          plasticity models.
    !>
    !> @param[in] shvars   Shared variables (stress tensor, etc.)
    !> @param[in] stvars   State variables (void ratio, etc.)
    !>
    !> @return res         Flow direction tensor
    !***************************************************************************
    module function flow_direction_impl(shvars, stvars) result(res)
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      type(Share_var) :: res
    endfunction flow_direction_impl
    !***************************************************************************
    !> @brief Calculate radial direction in stress space
    !>
    !> @details This function computes the radial direction tensor in stress
    !>          space, which points from the origin to the current stress state.
    !>          The radial direction is used in soil mechanics models to define
    !>          fabric evolution and anisotropic hardening directions.
    !>
    !> @param[in] shvars   Shared variables (stress tensor, etc.)
    !> @param[in] stvars   State variables (void ratio, etc.)
    !>
    !> @return res         Radial direction tensor
    !***************************************************************************
    module function radial_direction_impl(shvars) result(res)
      type(Share_var), intent(in) :: shvars
      type(Share_var) :: res
    endfunction radial_direction_impl
  endinterface

contains
!*******************************************************************************
endmodule math_mod
