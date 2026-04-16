!*******************************************************************************
!> @brief share_vars_impl
!>
!> @details 模块详细描述
!>
!> @author wuwenhao
!> @date 2025/11/27
!*******************************************************************************
submodule(Container_mod) Container_impl
  use Base_config
  use tensor_opt_mod
  use exception_mod
#include "macro.h"
  implicit none
  type(Torch) torch_
contains
  !=============================================================================
  !
  ! interface Share_var
  !
  !=============================================================================
  module procedure share_construct_param
  real(DP) :: mean
  !-----------------------------------------------------------------------------
  this%sigma_(:, :) = sigma(:, :)
  this%alpha_(:, :) = alpha(:, :)
  this%P0_ = p0
  !
  this%initialized_ = .true.
  !
  mean = torch_%Trace(this%sigma_) / 3.0_DP
  this%is_lowstress = merge(.true., .false., mean <= EPS)
  call this%jugde_nan_inf_impl()
  end procedure share_construct_param
  !*****************************************************************************
  module procedure Share_construct_zero
  this%P0_ = 0.0_DP
  this%sigma_(:, :) = 0.0_DP
  this%alpha_(:, :) = 0.0_DP
  this%initialized_ = .true.
  this%is_lowstress = .true.
  this%is_nan_inf = .false.
  end procedure Share_construct_zero
  !*****************************************************************************
  module procedure get_p0_impl
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  p0 = this%P0_
  end procedure get_p0_impl
  !*****************************************************************************
  module procedure get_sigma_impl
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  sigma(:, :) = this%sigma_(:, :)
  end procedure get_sigma_impl
  !*****************************************************************************
  module procedure get_alpha_impl
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  alpha(:, :) = this%alpha_(:, :)
  end procedure get_alpha_impl
  !*****************************************************************************
  module procedure low_impl
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  is_true = this%is_lowstress
  end procedure low_impl
  !*****************************************************************************
  module procedure update_p0_impl
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  this%P0_ = this%P0_ + dp0
  call this%jugde_nan_inf_impl()
  end procedure update_p0_impl
  !*****************************************************************************
  module procedure update_sigma_impl
  real(DP) :: mean
  !-----------------------------------------------------------------------------
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  this%sigma_(:, :) = this%sigma_(:, :) + dsigma(:, :)
  mean = torch_%Trace(this%sigma_) / 3.0_DP
  this%is_lowstress = merge(.true., .false., mean <= EPS)
  call this%jugde_nan_inf_impl()
  end procedure update_sigma_impl
  !*****************************************************************************
  module procedure update_alpha_impl
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  this%alpha_(:, :) = this%alpha_(:, :) + dalpha(:, :)
  call this%jugde_nan_inf_impl()
  end procedure update_alpha_impl
  !*****************************************************************************
  module procedure update_shvars_impl
  real(DP) :: mean
  !-----------------------------------------------------------------------------
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  this%sigma_(:, :) = this%sigma_(:, :) + dsigma(:, :)
  this%alpha_(:, :) = this%alpha_(:, :) + dalpha(:, :)
  this%P0_ = this%P0_ + dp0
  mean = torch_%Trace(this%sigma_) / 3.0_DP
  this%is_lowstress = merge(.true., .false., mean <= EPS)
  call this%jugde_nan_inf_impl()
  end procedure update_shvars_impl
  !*****************************************************************************
  module procedure changed_p0_impl
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  this%P0_ = p0
  call this%jugde_nan_inf_impl()
  end procedure changed_p0_impl
  !*****************************************************************************
  module procedure changed_sigma_impl
  real(DP) :: mean
  !-----------------------------------------------------------------------------
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  this%sigma_(:, :) = sigma(:, :)
  mean = torch_%Trace(this%sigma_) / 3.0_DP
  this%is_lowstress = merge(.true., .false., mean <= EPS)
  call this%jugde_nan_inf_impl()
  end procedure changed_sigma_impl
  !*****************************************************************************
  module procedure changed_alpha_impl
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  this%alpha_(:, :) = alpha(:, :)
  call this%jugde_nan_inf_impl()
  end procedure changed_alpha_impl
  !*****************************************************************************
  module procedure norm_impl
  res(1) = torch_%Norm(this%get_sigma())
  res(2) = torch_%Norm(this%get_alpha())
  res(3) = abs(this%get_p0())
  end procedure norm_impl
  !*****************************************************************************
  module procedure print_impl
  write(6, '(A12, F6.2)') &
    "P0 = ", this%P0_
  write(6, *) "Stress Tensor : "
  call torch_%Print(this%get_sigma())
  write(6, *) "Alpha Tensor : "
  call torch_%Print(this%alpha_)
  end procedure print_impl
  !*****************************************************************************
  module procedure jugde_nan_inf_impl
  logical has_error
  has_error = (this%P0_ /= this%P0_) .or. (abs(this%P0_) > MAX_DATA)
  has_error = has_error .or. any(this%sigma_ /= this%sigma_) &
              .or. any(abs(this%sigma_) > MAX_DATA)
  has_error = has_error .or. any(this%alpha_ /= this%alpha_) &
              .or. any(abs(this%alpha_) > MAX_DATA)
  this%is_nan_inf = has_error
  end procedure jugde_nan_inf_impl
  !*****************************************************************************
  module procedure assign_impl
  CHECK_TRUE(other%initialized_, "container share_vars has not initialized")
  this%P0_ = other%P0_
  this%sigma_(:, :) = other%sigma_(:, :)
  this%alpha_(:, :) = other%alpha_(:, :)
  this%initialized_ = other%initialized_
  this%is_lowstress = other%is_lowstress
  this%is_nan_inf = other%is_nan_inf
  end procedure assign_impl
  !*****************************************************************************
  module procedure binary_add_impl
  real(DP) :: mean
  !-----------------------------------------------------------------------------
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  CHECK_TRUE(other%initialized_, "container share_vars has not initialized")
  res%P0_ = this%P0_ + other%P0_
  res%sigma_ = this%sigma_ + other%sigma_
  res%alpha_ = this%alpha_ + other%alpha_
  res%initialized_ = .true.
  mean = torch_%Trace(res%sigma_) / 3.0_DP
  res%is_lowstress = merge(.true., .false., mean <= EPS)
  call res%jugde_nan_inf_impl()
  end procedure binary_add_impl
  !*****************************************************************************
  module procedure binary_sub_impl
  real(DP) :: mean
  !-----------------------------------------------------------------------------
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  CHECK_TRUE(other%initialized_, "container share_vars has not initialized")
  res%P0_ = this%P0_ - other%P0_
  res%sigma_ = this%sigma_ - other%sigma_
  res%alpha_ = this%alpha_ - other%alpha_
  res%initialized_ = .true.
  mean = torch_%Trace(res%sigma_) / 3.0_DP
  res%is_lowstress = merge(.true., .false., mean <= EPS)
  call res%jugde_nan_inf_impl()
  end procedure binary_sub_impl
  !*****************************************************************************
  module procedure unary_minus_impl
  real(DP) :: mean
  !-----------------------------------------------------------------------------
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  res%P0_ = -this%P0_
  res%sigma_ = -this%sigma_
  res%alpha_ = -this%alpha_
  res%initialized_ = this%initialized_
  mean = torch_%Trace(res%sigma_) / 3.0_DP
  res%is_lowstress = merge(.true., .false., mean <= EPS)
  call res%jugde_nan_inf_impl()
  end procedure unary_minus_impl
  !*****************************************************************************
  module procedure unary_lhs_scalar_impl
  real(DP) :: mean
  !-----------------------------------------------------------------------------
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  res%P0_ = scalar * this%P0_
  res%sigma_ = scalar * this%sigma_
  res%alpha_ = scalar * this%alpha_
  res%initialized_ = .true.
  mean = torch_%Trace(res%sigma_) / 3.0_DP
  res%is_lowstress = merge(.true., .false., mean <= EPS)
  call res%jugde_nan_inf_impl()
  end procedure unary_lhs_scalar_impl
  !*****************************************************************************
  module procedure unary_rhs_scalar_impl
  real(DP) :: mean
  !-----------------------------------------------------------------------------
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  res%P0_ = this%P0_ * scalar
  res%sigma_ = this%sigma_ * scalar
  res%alpha_ = this%alpha_ * scalar
  res%initialized_ = .true.
  mean = torch_%Trace(res%sigma_) / 3.0_DP
  res%is_lowstress = merge(.true., .false., mean <= EPS)
  call res%jugde_nan_inf_impl()
  end procedure unary_rhs_scalar_impl
  !*****************************************************************************
  module procedure unary_div_impl
  real(DP) :: mean, scalar_
  !-----------------------------------------------------------------------------
  CHECK_TRUE(this%initialized_, "container share_vars has not initialized")
  scalar_ = scalar
  if(abs(scalar_) < EPS) scalar_ = sign(scalar_, EPS)
  res%P0_ = this%P0_ / scalar_
  res%sigma_ = this%sigma_ / scalar_
  res%alpha_ = this%alpha_ / scalar_
  res%initialized_ = .true.
  mean = torch_%Trace(res%sigma_) / 3.0_DP
  res%is_lowstress = merge(.true., .false., mean <= EPS)
  call res%jugde_nan_inf_impl()
  end procedure unary_div_impl
  !
  !=============================================================================
  !
  ! interface State_var
  !
  !=============================================================================
  !
  module procedure State_construct_param
  this%voidr_ = voidr
  this%alpha_in_ = alpha_in
  this%pnewdt_ = pnewdt
  this%initialized_ = .true.
  end procedure State_construct_param
  !*****************************************************************************
  module procedure get_alpha_in_impl
  CHECK_TRUE(this%initialized_, "container state_vars has not initialized")
  alpha_in = this%alpha_in_
  end procedure get_alpha_in_impl
  !*****************************************************************************
  module procedure get_voidr_impl
  CHECK_TRUE(this%initialized_, "container state_vars has not initialized")
  voidr = this%voidr_
  end procedure get_voidr_impl
  !*****************************************************************************
  module procedure get_pnewdt_impl
  CHECK_TRUE(this%initialized_, "container state_vars has not initialized")
  pnewdt = this%pnewdt_
  end procedure get_pnewdt_impl
  !*****************************************************************************
  module procedure update_voidr_impl
  real(DP) :: despv
  !-----------------------------------------------------------------------------
  CHECK_TRUE(this%initialized_, "container state_vars has not initialized")
  despv = torch_%Trace(depsln)
  this%voidr_ = this%voidr_ - (1.0_DP + this%voidr_) * despv
  end procedure update_voidr_impl
  !*****************************************************************************
  module procedure changed_voidr_impl
  CHECK_TRUE(this%initialized_, "container state_vars has not initialized")
  this%voidr_ = voidr
  end procedure changed_voidr_impl
  !*****************************************************************************
  module procedure change_alpha_in_impl
  CHECK_TRUE(this%initialized_, "container state_vars has not initialized")
  this%alpha_in_ = alpha_in
  end procedure change_alpha_in_impl
  !*****************************************************************************
  module procedure changed_pnewdt_impl
  CHECK_TRUE(this%initialized_, "container state_vars has not initialized")
  this%pnewdt_ = pnewdt
  end procedure changed_pnewdt_impl
  !*****************************************************************************
  module procedure assign_impl_state
  CHECK_TRUE(other%initialized_, "container state_vars has not initialized")
  this%voidr_ = other%voidr_
  this%pnewdt_ = other%pnewdt_
  this%initialized_ = other%initialized_
  end procedure assign_impl_state
endsubmodule
