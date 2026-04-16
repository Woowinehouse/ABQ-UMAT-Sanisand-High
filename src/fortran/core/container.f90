!*******************************************************************************
!> @brief Container module for UMAT state variables
!>
!> @details
!> This module defines container types for storing and managing state
!> variables in ABAQUS UMAT implementations. It provides two main types:
!> 1. Share_var: Contains hardening parameters, stress tensors, and fabric
!>    tensors with associated update operations.
!> 2. State_var: Contains void ratio and time increment parameters for
!>    state-dependent calculations.
!> The module supports arithmetic operations, assignment, and various
!> utility functions for state variable management.
!>
!> @author wuwenhao
!> @date 2025/12/07
!*******************************************************************************
module Container_mod
  use Base_config
  implicit none
  private
  ! Type definitions
  type, public :: Share_var
    private
    real(DP) :: sigma_(3, 3)
    real(DP) :: alpha_(3, 3)
    real(DP) :: P0_
    logical :: initialized_ = .false.
    logical :: is_lowstress = .false.
    logical :: is_nan_inf = .false.
  contains
    procedure, public, pass(this) :: get_sigma => get_sigma_impl
    procedure, public, pass(this) :: get_alpha => get_alpha_impl
    procedure, public, pass(this) :: get_p0 => get_p0_impl
    procedure, public, pass(this) :: update_sigma => update_sigma_impl
    procedure, public, pass(this) :: update_alpha => update_alpha_impl
    procedure, public, pass(this) :: update_p0 => update_p0_impl
    procedure, public, pass(this) :: update_shvars => update_shvars_impl
    procedure, public, pass(this) :: changed_sigma => changed_sigma_impl
    procedure, public, pass(this) :: changed_alpha => changed_alpha_impl
    procedure, public, pass(this) :: changed_p0 => changed_p0_impl
    procedure, public, pass(this) :: is_low => low_impl
    procedure, public, pass(this) :: norm => norm_impl
    procedure, public, pass(this) :: print => print_impl
    procedure, private, pass(this) :: jugde_nan_inf_impl
    procedure, pass(this) :: assign => assign_impl
    generic, public :: assignment(=) => assign
    procedure, pass(this) :: binary_add => binary_add_impl
    generic, public :: operator(+) => binary_add
    procedure, pass(this) :: binary_sub => binary_sub_impl
    generic, public :: operator(-) => binary_sub
    procedure, pass(this) :: unary_minus => unary_minus_impl
    generic, public :: operator(-) => unary_minus
    procedure, pass(this) :: unary_lhs_scalar => unary_lhs_scalar_impl
    generic, public :: operator(*) => unary_lhs_scalar
    procedure, pass(this) :: unary_rhs_scalar => unary_rhs_scalar_impl
    generic, public :: operator(*) => unary_rhs_scalar
    procedure, pass(this) :: unary_div_impl => unary_div_impl
    generic, public :: operator(/) => unary_div_impl
  endtype Share_var
  !
  type, public :: State_var
    private
    real(DP), dimension(3, 3) :: alpha_in_
    real(DP) :: voidr_
    real(DP) :: pnewdt_
    logical :: initialized_ = .false.
  contains
    procedure, public, pass(this) :: get_alpha_in => get_alpha_in_impl
    procedure, public, pass(this) :: get_voidr => get_voidr_impl
    procedure, public, pass(this) :: get_pnewdt => get_pnewdt_impl
    procedure, public, pass(this) :: update_voidr => update_voidr_impl
    procedure, public, pass(this) :: changed_voidr => changed_voidr_impl
    procedure, public, pass(this) :: change_alpha_in => change_alpha_in_impl
    procedure, public, pass(this) :: changed_pnewdt => changed_pnewdt_impl
    procedure, pass(this) :: assign => assign_impl_state
    generic, public :: assignment(=) => assign
  endtype State_var
  !
  interface Share_var
    module procedure :: share_construct_param
    module procedure :: Share_construct_zero
  endinterface
  !
  interface State_var
    module procedure :: State_construct_param
  endinterface
  !
  interface
    !***************************************************************************
    !> @brief Construct Share_var with given parameters
    !>
    !> @details Creates a Share_var object with specified stress tensor,
    !>          anisotropy tensor, initial pressure, and break pressure
    !>
    !> @param[in] sigma Stress tensor (3x3)
    !> @param[in] alpha Anisotropy tensor (3x3)
    !> @param[in] p0 Initial pressure
    !> @param[in] pb Break pressure
    !> @return this Constructed Share_var object
    !***************************************************************************
    module function share_construct_param(sigma, alpha, p0) result(this)
      real(DP), dimension(3, 3), intent(in) :: sigma
      real(DP), dimension(3, 3), intent(in) :: alpha
      real(DP), intent(in) :: p0
      type(Share_var) :: this
    endfunction share_construct_param
    !***************************************************************************
    !> @brief Construct zero-initialized Share_var
    !>
    !> @details Creates a Share_var object with all components set to zero
    !>          and low stress flag set to true
    !>
    !> @return this Zero-initialized Share_var object
    !***************************************************************************
    module function Share_construct_zero() result(this)
      type(Share_var) :: this
    endfunction Share_construct_zero
    !***************************************************************************
    !> @brief Get initial pressure (P0) from Share_var
    !>
    !> @details Returns the current initial pressure value from the Share_var object
    !>
    !> @param[in] this Share_var object
    !> @return p0 Initial pressure value
    !***************************************************************************
    module function get_p0_impl(this) result(p0)
      class(Share_var), intent(in) :: this
      real(DP) :: p0
    endfunction get_p0_impl
    !***************************************************************************
    !> @brief Get stress tensor from Share_var
    !>
    !> @details Returns the current stress tensor (3x3) from the Share_var object
    !>
    !> @param[in] this Share_var object
    !> @return sigma Stress tensor (3x3)
    !***************************************************************************
    module function get_sigma_impl(this) result(sigma)
      class(Share_var), intent(in) :: this
      real(DP), dimension(3, 3) :: sigma
    endfunction
    !***************************************************************************
    !> @brief Get anisotropy tensor (alpha) from Share_var
    !>
    !> @details Returns the current anisotropy tensor (3x3) from the Share_var object
    !>
    !> @param[in] this Share_var object
    !> @return alpha Anisotropy tensor (3x3)
    !***************************************************************************
    module function get_alpha_impl(this) result(alpha)
      class(Share_var), intent(in) :: this
      real(DP), dimension(3, 3) :: alpha
    endfunction
    !***************************************************************************
    !> @brief Check if Share_var is in low stress state
    !>
    !> @details Returns true if the mean stress is below or equal to EPS
    !>
    !> @param[in] this Share_var object
    !> @return is_true Logical flag indicating low stress state
    !***************************************************************************
    module function low_impl(this) result(is_true)
      class(Share_var), intent(in) :: this
      logical :: is_true
    endfunction low_impl
    !***************************************************************************
    !> @brief Update initial pressure (P0) by increment
    !>
    !> @details Adds the given increment to the current initial pressure
    !>          and checks for NaN/Inf values
    !>
    !> @param[inout] this Share_var object to update
    !> @param[in] dp0 Increment to add to initial pressure
    !***************************************************************************
    module subroutine update_p0_impl(this, dp0)
      class(Share_var), intent(inout) :: this
      real(DP), intent(in) :: dp0
    endsubroutine update_p0_impl
    !***************************************************************************
    !> @brief Update stress tensor by increment
    !>
    !> @details Adds the given increment to the current stress tensor,
    !>          updates low stress flag based on mean stress, and checks
    !>          for NaN/Inf values
    !>
    !> @param[inout] this Share_var object to update
    !> @param[in] dsigma Stress tensor increment (3x3)
    !***************************************************************************
    module subroutine update_sigma_impl(this, dsigma)
      class(Share_var), intent(inout) :: this
      real(DP), dimension(3, 3), intent(in) :: dsigma
    endsubroutine update_sigma_impl
    !***************************************************************************
    !> @brief Update anisotropy tensor (alpha) by increment
    !>
    !> @details Adds the given increment to the current anisotropy tensor
    !>          and checks for NaN/Inf values
    !>
    !> @param[inout] this Share_var object to update
    !> @param[in] dalpha Anisotropy tensor increment (3x3)
    !***************************************************************************
    module subroutine update_alpha_impl(this, dalpha)
      class(Share_var), intent(inout) :: this
      real(DP), dimension(3, 3), intent(in) :: dalpha
    endsubroutine update_alpha_impl
    !***************************************************************************
    !> @brief Update all shared variables simultaneously
    !>
    !> @details Updates initial pressure, break pressure, stress tensor, and anisotropy tensor
    !>          with given increments, updates low stress flag based on mean
    !>          stress, and checks for NaN/Inf values
    !>
    !> @param[in] dp0 Initial pressure increment
    !> @param[in] dpb Break pressure increment
    !> @param[in] dsigma Stress tensor increment (3x3)
    !> @param[in] dalpha Anisotropy tensor increment (3x3)
    !> @param[inout] this Share_var object to update
    !***************************************************************************
    module subroutine update_shvars_impl(dsigma, dalpha, dp0, this)
      real(DP), dimension(3, 3), intent(in) :: dsigma
      real(DP), dimension(3, 3), intent(in) :: dalpha
      real(DP), intent(in) :: dp0
      class(Share_var), intent(inout) :: this
    endsubroutine update_shvars_impl
    !***************************************************************************
    !> @brief Change initial pressure (P0) to new value
    !>
    !> @details Sets the initial pressure to the given value (not increment)
    !>          and checks for NaN/Inf values
    !>
    !> @param[inout] this Share_var object to modify
    !> @param[in] p0 New initial pressure value
    !***************************************************************************
    module subroutine changed_p0_impl(this, p0)
      class(Share_var), intent(inout) :: this
      real(DP), intent(in) :: p0
    endsubroutine changed_p0_impl
    !***************************************************************************
    !> @brief Change stress tensor to new value
    !>
    !> @details Sets the stress tensor to the given value (not increment),
    !>          updates low stress flag based on mean stress, and checks
    !>          for NaN/Inf values
    !>
    !> @param[inout] this Share_var object to modify
    !> @param[in] sigma New stress tensor value (3x3)
    !***************************************************************************
    module subroutine changed_sigma_impl(this, sigma)
      class(Share_var), intent(inout) :: this
      real(DP), dimension(3, 3), intent(in) :: sigma
    endsubroutine changed_sigma_impl
    !***************************************************************************
    !> @brief Change anisotropy tensor (alpha) to new value
    !>
    !> @details Sets the anisotropy tensor to the given value (not increment)
    !>          and checks for NaN/Inf values
    !>
    !> @param[inout] this Share_var object to modify
    !> @param[in] alpha New anisotropy tensor value (3x3)
    !***************************************************************************
    module subroutine changed_alpha_impl(this, alpha)
      class(Share_var), intent(inout) :: this
      real(DP), dimension(3, 3), intent(in) :: alpha
    endsubroutine changed_alpha_impl
    !***************************************************************************
    !> @brief Compute norms of Share_var components
    !>
    !> @details Returns a 4-element array containing:
    !>          1. Absolute value of initial pressure (P0)
    !>          2. Absolute value of break pressure (Pb)
    !>          3. Norm of stress tensor
    !>          4. Norm of anisotropy tensor
    !>
    !> @param[in] this Share_var object
    !> @return res Array of norms [|P0|, |Pb|, ||sigma||, ||alpha||]
    !***************************************************************************
    module function norm_impl(this) result(res)
      class(Share_var), intent(in) :: this
      real(DP), dimension(3) :: res
    endfunction norm_impl
    !***************************************************************************
    !> @brief Print Share_var contents to standard output
    !>
    !> @details Outputs the stress tensor, anisotropy tensor, initial pressure,
    !>          and break pressure in a formatted manner for debugging and
    !>          inspection purposes
    !>
    !> @param[in] this Share_var object to print
    !***************************************************************************
    module subroutine print_impl(this)
      class(Share_var), intent(in) :: this
    endsubroutine print_impl
    !***************************************************************************
    !> @brief Check for NaN and Inf values in Share_var
    !>
    !> @details Examines all components of the Share_var object for NaN (Not a Number)
    !>          and infinite values, updating the is_nan_inf flag accordingly
    !>
    !> @param[inout] this Share_var object to check
    !***************************************************************************
    module subroutine jugde_nan_inf_impl(this)
      class(Share_var), intent(inout) :: this
    endsubroutine jugde_nan_inf_impl
    !***************************************************************************
    !> @brief Assign one Share_var to another
    !>
    !> @details Copies all components from the source Share_var to the destination,
    !>          including internal state flags (initialized, low stress, NaN/Inf)
    !>
    !> @param[inout] this Destination Share_var object
    !> @param[in] other Source Share_var object
    !***************************************************************************
    module subroutine assign_impl(this, other)
      class(Share_var), intent(inout) :: this
      type(Share_var), intent(in) :: other
    endsubroutine assign_impl
    !***************************************************************************
    !> @brief Add two Share_var objects
    !>
    !> @details Performs element-wise addition of two Share_var objects,
    !>          adding corresponding hardening parameters, stress tensors,
    !>          and fabric tensors. The break parameter is not included
    !>          in the addition operation.
    !>
    !> @param[in] this First Share_var object
    !> @param[in] other Second Share_var object
    !> @return res Resulting Share_var object with summed components
    !***************************************************************************
    module function binary_add_impl(this, other) result(res)
      class(Share_var), intent(in) :: this
      type(Share_var), intent(in) :: other
      type(Share_var) :: res
    endfunction binary_add_impl
    !***************************************************************************
    !> @brief Subtract two Share_var objects
    !>
    !> @details Performs element-wise subtraction of two Share_var objects,
    !>          subtracting corresponding hardening parameters, stress tensors,
    !>          and fabric tensors. The break parameter is not included
    !>          in the subtraction operation.
    !>
    !> @param[in] this First Share_var object (minuend)
    !> @param[in] other Second Share_var object (subtrahend)
    !> @return res Resulting Share_var object with subtracted components
    !***************************************************************************
    module function binary_sub_impl(this, other) result(res)
      class(Share_var), intent(in) :: this
      type(Share_var), intent(in) :: other
      type(Share_var) :: res
    endfunction binary_sub_impl
    !***************************************************************************
    !> @brief Negate a Share_var object
    !>
    !> @details Performs element-wise negation of a Share_var object,
    !>          negating the hardening parameter, stress tensor, and fabric tensor.
    !>          The break parameter is not included in the negation operation.
    !>
    !> @param[in] this Share_var object to negate
    !> @return res Resulting Share_var object with negated components
    !***************************************************************************
    module function unary_minus_impl(this) result(res)
      class(Share_var), intent(in) :: this
      type(Share_var) :: res
    endfunction unary_minus_impl
    !***************************************************************************
    !> @brief Multiply scalar on left side with Share_var object
    !>
    !> @details Performs element-wise multiplication of a scalar with a Share_var object,
    !>          multiplying the scalar with the hardening parameter, stress tensor,
    !>          and fabric tensor. The break parameter is not included in the operation.
    !>
    !> @param[in] scalar Scalar multiplier
    !> @param[in] this Share_var object to multiply
    !> @return res Resulting Share_var object with scaled components
    !***************************************************************************
    module function unary_lhs_scalar_impl(scalar, this) result(res)
      real(DP), intent(in) :: scalar
      class(Share_var), intent(in) :: this
      type(Share_var) :: res
    endfunction unary_lhs_scalar_impl
    !***************************************************************************
    !> @brief Multiply Share_var object with scalar on right side
    !>
    !> @details Performs element-wise multiplication of a Share_var object with a scalar,
    !>          multiplying the hardening parameter, stress tensor, and fabric tensor
    !>          with the scalar. The break parameter is not included in the operation.
    !>
    !> @param[in] this Share_var object to multiply
    !> @param[in] scalar Scalar multiplier
    !> @return res Resulting Share_var object with scaled components
    !***************************************************************************
    module function unary_rhs_scalar_impl(this, scalar) result(res)
      class(Share_var), intent(in) :: this
      real(DP), intent(in) :: scalar
      type(Share_var) :: res
    endfunction unary_rhs_scalar_impl
    !***************************************************************************
    !> @brief Divide Share_var object by scalar
    !>
    !> @details Performs element-wise division of a Share_var object by a scalar,
    !>          dividing the hardening parameter, stress tensor, and fabric tensor
    !>          by the scalar. The break parameter is not included in the operation.
    !>
    !> @param[in] this Share_var object to divide
    !> @param[in] scalar Scalar divisor
    !> @return res Resulting Share_var object with divided components
    !***************************************************************************
    module function unary_div_impl(this, scalar) result(res)
      class(Share_var), intent(in) :: this
      real(DP), intent(in) :: scalar
      type(Share_var) :: res
    endfunction unary_div_impl
    ! end interface
  endinterface

  !
  interface
    !***************************************************************************
    !> @brief Construct State_var with given parameters
    !>
    !> @details Creates a State_var object with specified void ratio
    !>          and time increment parameter
    !>
    !> @param[in] voidr Void ratio parameter
    !> @param[in] pnewdt Time increment parameter
    !> @return this Constructed State_var object
    !***************************************************************************
    module function State_construct_param(voidr, alpha_in, pnewdt) result(this)
      real(DP), intent(in) :: voidr
      real(DP), dimension(3, 3), intent(in) :: alpha_in
      real(DP), intent(in) :: pnewdt
      type(State_var) :: this
    endfunction State_construct_param
    !***************************************************************************
    module function get_alpha_in_impl(this) result(alpha_in)
      class(State_var), intent(in) :: this
      real(DP), dimension(3, 3) :: alpha_in
    endfunction get_alpha_in_impl
    !***************************************************************************
    !> @brief Get void ratio from State_var
    !>
    !> @details Returns the current void ratio value from the State_var object
    !>
    !> @param[in] this State_var object
    !> @return voidr Void ratio value
    !***************************************************************************
    module function get_voidr_impl(this) result(voidr)
      class(State_var), intent(in) :: this
      real(DP) :: voidr
    endfunction get_voidr_impl
    !***************************************************************************
    !> @brief Get time increment parameter from State_var
    !>
    !> @details Returns the current time increment parameter value from the State_var object
    !>
    !> @param[in] this State_var object
    !> @return pnewdt Time increment parameter value
    !***************************************************************************
    module function get_pnewdt_impl(this) result(pnewdt)
      class(State_var), intent(in) :: this
      real(DP) :: pnewdt
    endfunction get_pnewdt_impl
    !***************************************************************************
    !> @brief Update void ratio by strain increment
    !>
    !> @details Updates the void ratio based on the given strain increment tensor.
    !>          The void ratio is typically updated according to volumetric strain
    !>          changes in soil mechanics calculations.
    !>
    !> @param[inout] this State_var object to update
    !> @param[in] depsln Strain increment tensor (3x3)
    !***************************************************************************
    module subroutine update_voidr_impl(this, depsln)
      class(State_var), intent(inout) :: this
      real(DP), dimension(3, 3), intent(in) :: depsln
    endsubroutine update_voidr_impl
    !***************************************************************************
    !> @brief Change void ratio to new value
    !>
    !> @details Sets the void ratio to the given value (not increment)
    !>          in the State_var object
    !>
    !> @param[inout] this State_var object to modify
    !> @param[in] voidr New void ratio value
    !***************************************************************************
    module subroutine changed_voidr_impl(this, voidr)
      class(State_var), intent(inout) :: this
      real(DP), intent(in) :: voidr
    endsubroutine changed_voidr_impl
    !***************************************************************************
    module subroutine change_alpha_in_impl(this, alpha_in)
      class(State_var), intent(inout) :: this
      real(DP), dimension(3, 3), intent(in) :: alpha_in
    endsubroutine change_alpha_in_impl
    !***************************************************************************
    !> @brief Change time increment parameter to new value
    !>
    !> @details Sets the time increment parameter (pnewdt) to the given value
    !>          (not increment) in the State_var object. This parameter is used
    !>          in ABAQUS UMAT to control the time step size for the next increment.
    !>
    !> @param[inout] this State_var object to modify
    !> @param[in] pnewdt New time increment parameter value
    !***************************************************************************
    module subroutine changed_pnewdt_impl(this, pnewdt)
      class(State_var), intent(inout) :: this
      real(DP), intent(in) :: pnewdt
    endsubroutine changed_pnewdt_impl
    !***************************************************************************
    !> @brief Assign one State_var to another
    !>
    !> @details Copies all components from the source State_var to the destination,
    !>          including void ratio, time increment parameter, and initialization flag.
    !>          This enables assignment operations between State_var objects.
    !>
    !> @param[inout] this Destination State_var object
    !> @param[in] other Source State_var object
    !***************************************************************************
    module subroutine assign_impl_state(this, other)
      class(State_var), intent(inout) :: this
      type(State_var), intent(in) :: other
    endsubroutine assign_impl_state
  endinterface
contains

endmodule Container_mod
