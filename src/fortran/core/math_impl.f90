!*******************************************************************************
!> @brief Math module for numerical algorithms
!>
!> @details This module provides mathematical utilities including bisection method,
!>          interval checking, and other numerical algorithms used in the UMAT.
!>
!> @author wuwenhao
!> @date 2025/11/27
!*******************************************************************************
! #ifdef DEBUG
! #define ENABLE_MONOTONIC_CHECK 1
! #else
! #define ENABLE_MONOTONIC_CHECK 0
! #endif
submodule(math_mod) math_impl
  use exception_mod
  use tensor_opt_mod
  use elastic_mod
  use plastic_mod
  USE PRESOLVE_MOD
#include "macro.h"
  implicit none
  type(Elast) :: elast_
  type(Plast) :: plast_
  type(Math) :: math_
  type(Torch) :: torch_
  !
contains
  module procedure elastic_update_impl
  real(DP), dimension(3, 3) :: dsigma
  !----------------------------------------------------------------------------
  elastiff = elast_%get_stiffness(shvars, stvars)
  dsigma(:, :) = elast_%calc_dsigma(shvars, stvars, depsln)
  call shvars%update_sigma(dsigma)
  call stvars%update_voidr(depsln)
  end procedure
  !*****************************************************************************
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
  !*****************************************************************************
  module procedure intchc_impl
  real(DP) :: mean_etr, mean_cur
  real(DP), dimension(3, 3) :: pfsig, dsigma
  real(DP) :: angle, lbd
  real(DP) :: fright, fleft, ftemp
  real(DP) :: iter, rbd
  !
  mean_cur = torch_%Trace(shvars%get_sigma()) / 3.0_DP
  mean_etr = mean_with_depsln(shvars, stvars, depsln)
  if(mean_etr <= 0.0_DP .and. mean_cur >= 0.0_DP) then
    ! Ensure that there is a solution within the domain.
    rbd = Bisection_impl(shvars, stvars, depsln, mean_with_depsln, &
                         0.0_DP, 1.0_DP, 0.0_DP, tol)
  else
    rbd = 1.0_DP
  endif
  !
  fleft = elast_%Yield_distance(shvars)
  fright = ftol_with_depsln(shvars, stvars, rbd * depsln)
  !
  if(fleft * fright >= 0.0_DP) then
    pfsig(:, :) = plast_%Get_pfsig(shvars)
    dsigma(:, :) = elast_%calc_dsigma(shvars, stvars, depsln)
    angle = torch_%Get_cost(pfsig, dsigma)
    if(angle < 0.0_DP) then
      iter = 0.0_DP
      do while(iter <= 1.0_DP)
        ftemp = ftol_with_depsln(shvars, stvars, iter * depsln)
        if(ftemp <= -tol) then
          lbd = iter
          exit
        else
          iter = iter + 0.01_DP
          if(iter >= 1.0_DP) then
            alout = 0.0_DP
            return
          endif
        endif
      enddo
    else
      alout = 0.0_DP
      return
    endif
  else
    lbd = 0.0_DP
  endif
  alout = Bisection_impl(shvars, stvars, depsln, ftol_with_depsln, &
                         lbd, rbd, 0.0_DP, tol)
  !
  end procedure intchc_impl
  !*****************************************************************************
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
  !*****************************************************************************
  module procedure Bisection_impl
  real(DP) :: left, right, mid
  real(DP) :: f_left, f_right, f_mid
  real(DP) :: df_left, df_right, df_mid
  integer(I4) :: it, it_max
  !-----------------------------------------------------------------------------
  left = lbd
  right = rbd
  it_max = 100
  !check input variable
  CHECK_TRUE(left >= 0.0_DP .and. right <= 1.0_DP, "left and right should be in [0,1]")
  CHECK_TRUE(left < right, "The interval can not be emptied.")
  !
  f_left = func(shvars, stvars, left * depsln)
  f_right = func(shvars, stvars, right * depsln)
  !
  CHECK_TRUE(f_left * f_right <= 0.0_DP, " Bisection_impl: Thefunction must have different signs at the boundaries.")
  ! monotonic
  ! if(ENABLE_MONOTONIC_CHECK == 1) then
  !   monotonic = is_monotonic(shvars, stvars, depsln, func, left, right)
  !   CHECK_TRUE(monotonic, "func must be monotonic")
  ! endif
  ! iterator
  do it = 1, it_max
    mid = left + (right - left) / 2.0_DP
    f_mid = func(shvars, stvars, mid * depsln)
    df_mid = f_mid - condition
    df_left = f_left - condition
    df_right = f_right - condition
    if(abs(df_mid) <= tol) then
      alout = mid
      return
    endif
    if(df_left * df_mid >= 0.0_DP) then
      left = mid
      f_left = f_mid
    else
      right = mid
      f_right = f_mid
    endif
  enddo
  end procedure Bisection_impl
  !*****************************************************************************
  module procedure Onyield_impl
  type(Share_var) :: shtmp, Defor, Desec, shfor, shdrt
  type(State_var) :: sttmp, stfor, stdrt
  real(DP), dimension(3, 3, 3, 3) :: dempx1, dempx2, dedrt
  integer(I4) :: it, nfail
  real(DP) :: dt, t, sstol, rtol, beta, fupd
  real(DP), dimension(3, 3) :: dnorm, temp
  logical :: converged
  !-----------------------------------------------------------------------------
  ! initialize variable
  dt = 1.0_DP
  t = 0.0_DP
  dempx = 0.0_DP
  sstol = 1.0D-6
  nfail = 0
  converged = .false.
  !
  do it = 1, 1000
    call plast_%Elstop(shvars, stvars, dt * depsln, Defor, dempx1)
    shfor = shvars + Defor
    stfor = stvars
    if(shfor%is_low()) then
      if(dt <= 1.0D-6) then
        exit
      else
        dt = dt / 2.0_DP
        cycle
      endif
    endif
    call plast_%Elstop(shfor, stfor, dt * depsln, Desec, dempx2)
    shtmp = shvars + (Defor + Desec) / 2.0_DP

    sttmp = stfor
    call sttmp%update_voidr(dt * depsln)
    !
    dnorm = elast_%Get_dnorm(shtmp)
    temp = shtmp%get_alpha() - stfor%get_alpha_in()
    if(sum(temp * dnorm) < 0) then
      call sttmp%change_alpha_in(shfor%get_alpha())
    endif
    if(shtmp%is_low()) then
      if(dt <= 1.0D-6) then
        shvars = shfor
        exit
      else
        dt = dt / 2.0_DP
        cycle
      endif
    endif
    !
    rtol = Get_residual_impl(Defor, Desec, shtmp)
    beta = 0.8d0 * dsqrt(sstol / rtol)
    !
    if(rtol <= sstol) then
      fupd = elast_%Yield_distance(shtmp)
      ! revise
      call drift_shvars_impl(shtmp, sttmp, eps, NUMBER, NOEL, NPT)

      ! update time
      t = t + dt
      dempx = dempx + ((dempx1 + dempx2) / 2.0_DP) * dt
      ! update variable
      shvars = shtmp
      stvars = sttmp
      ! exit
      if(abs(1.0_DP - t) <= EPS) then
        converged = .true.
        exit
      endif
      ! the next dt
      select case(nfail)
      case(1)
        dt = min(beta * dt, dt, 1.0D0 - t)
      case(0)
        dt = min(beta * dt, 1.1d0 * dt, 1.0D0 - t)
      endselect
    else
      ! update fail
      nfail = 1
      ! the next dt
      dt = max(beta * dt, 1.0D-3, 0.1 * dt)
    endif
  enddo
  !
  if(.not. converged) then
    ! 迭代失败
    call stvars%changed_pnewdt(0.5d0)
    write(7, *) "too many attempt made for the increment of stress"
    write(7, *) "total time=", t, "current increment time=", dt, &
      "sstol=", sstol, "rtol=", rtol
    return
  endif
  ! 应力太小
  if(shvars%is_low()) then
    !
    return
  endif
  !
  return
  end procedure Onyield_impl
  !*****************************************************************************
  module procedure Get_residual_impl
  type(Share_var) :: sh_dif
  real(DP), dimension(3) :: norm_dif, norm_tmg, vartol
  !-----------------------------------------------------------------------------
  sh_dif = shfor - shsec
  norm_dif = sh_dif%norm()
  norm_tmg = shtmp%norm()
  where(norm_tmg < EPS)
    norm_tmg = EPS
  endwhere
  !
  vartol = norm_dif / 2.0_DP / norm_tmg
  residual = max(vartol(1), vartol(2), vartol(3), EPS)
  if(vartol(1) <= 1.D-8) residual = vartol(1)
  if(vartol(2) <= 1.D-8) residual = vartol(2)
  end procedure Get_residual_impl
  !*****************************************************************************
  !> @brief : drift_shvars_impl
  !
  !> @param[in] shtmp
  !> @param[in] sttmp
  !> @param[in] depsln
  !> @param[out] shdrt
  !> @param[out] stdrt
  !*****************************************************************************
  module procedure drift_shvars_impl
  type(Share_var) :: shvar_with_flow, sh_radial
  real(DP) :: df_cur, df_flow, ftol
  integer(I4) :: it
  logical :: converged
  !-----------------------------------------------------------------------------
  converged = .false.
  ! iterator
  do it = 1, 8
    df_cur = elast_%Yield_distance(shtmp)
    if(abs(df_cur) <= epsilon) then
      converged = .true.
      exit
    endif
    !
    ! @brief 塑性流动方向修正
    !
    shvar_with_flow = flow_direction_impl(shtmp, sttmp)
    df_flow = elast_%Yield_distance(shvar_with_flow)
    if(abs(df_flow) < abs(df_cur)) then
      ! 径向返回
      shtmp = shvar_with_flow
    else
      shtmp = radial_direction_impl(shvar_with_flow)
    endif
  enddo
  if(.not. converged) then
    call sttmp%changed_pnewdt(0.5d0)
  endif
  end procedure drift_shvars_impl
  !*****************************************************************************
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
  !*****************************************************************************
  module procedure ftol_with_depsln
  real(DP), dimension(3, 3, 3, 3) :: stiff
  real(DP), dimension(3, 3) :: dsigma
  type(Share_var) :: sh_temp
  stiff = elast_%Get_stiffness(shvars, stvars)
  dsigma(:, :) = stiff.ddot.depsln
  sh_temp = shvars
  call sh_temp%update_sigma(dsigma)
  ftol = elast_%Yield_distance(sh_temp)
  end procedure ftol_with_depsln
  !*****************************************************************************
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
  !*****************************************************************************
  module procedure mean_with_depsln
  real(DP), dimension(3, 3, 3, 3) :: stiff
  real(DP), dimension(3, 3) :: dsigma
  type(Share_var) :: sh_temp
  !
  stiff = elast_%Get_stiffness(shvars, stvars)
  dsigma = stiff.ddot.depsln
  sh_temp = shvars
  call sh_temp%update_sigma(dsigma(:, :))
  res = torch_%Trace(sh_temp%get_sigma()) / 3.0_DP
  end procedure mean_with_depsln
  !*****************************************************************************
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
  !*****************************************************************************
  module procedure is_monotonic
  real(DP) :: x1, x2, f1, f2
  integer(I4) :: i, n_samples
  logical :: increasing = .true.
  logical :: decreasing = .true.
  n_samples = 10  ! 采样点数
  do i = 1, n_samples
    x1 = lbd + (rbd - lbd) * (i - 1) / real(n_samples, DP)
    x2 = lbd + (rbd - lbd) * i / real(n_samples, DP)

    f1 = func(shvars, stvars, x1 * depsln)
    f2 = func(shvars, stvars, x2 * depsln)
    if(f2 < f1) increasing = .false.
    if(f2 > f1) decreasing = .false.
  enddo
  is_monotonic = increasing .or. decreasing
  end procedure is_monotonic
  !*****************************************************************************
  !> @brief : flow_direction_impl
  !
  !> @param[in] shtmp
  !> @param[in] sttmp
  !> @param[out] res
  !*****************************************************************************
  module procedure flow_direction_impl
  real(DP) :: dftol
  real(DP) :: pfsig(3, 3), xm(3, 3), Dkp, dnmetr, frnde
  real(DP) :: Ralpha(3, 3), RP0, dlamda, cbxm(3, 3), Rb
  real(DP), dimension(3, 3, 3, 3) :: stiff
  !-----------------------------------------------------------------------------
  dftol = elast_%Yield_distance(shvars)
  stiff = elast_%Get_stiffness(shvars, stvars)
  pfsig = plast_%Get_pfsig(shvars)
  xm = plast_%Get_pgsig(shvars, stvars)
  Dkp = plast_%Get_Dkp(shvars, stvars)
  cbxm = stiff.ddot.xm
  dnmetr = sum(pfsig * (stiff.ddot.xm))
  frnde = dnmetr + Dkp
  if(abs(frnde) <= EPS) frnde = sign(frnde, EPS)
  dlamda = dftol / frnde
  !
  call plast_%Get_evolution(shvars, stvars, Ralpha, RP0)
  res = shvars
  call res%update_shvars(-cbxm * dlamda, dlamda * Ralpha, dlamda * RP0)
  return
  end procedure flow_direction_impl
  !*****************************************************************************
  !> @brief : flow_direction_impl
  !
  !> @param[in] shtmp
  !> @param[in] sttmp
  !> @param[out] res
  !*****************************************************************************
  module procedure radial_direction_impl
  real(DP) :: dftol, dnme, dlamda
  real(DP), dimension(3, 3) :: pfsig
  !-----------------------------------------------------------------------------
  dftol = elast_%Yield_distance(shvars)
  pfsig = plast_%Get_pfsig(shvars)
  dnme = sum(pfsig**2)
  dlamda = dftol / dnme
  res = shvars
  call res%update_sigma(-dlamda * pfsig)
  return
  end procedure radial_direction_impl
!*******************************************************************************
endsubmodule math_impl
