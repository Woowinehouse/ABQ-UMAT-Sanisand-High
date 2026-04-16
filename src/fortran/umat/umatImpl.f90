!dir$ freeform
!*******************************************************************************
!  project name
!  copyright (c) 2025 wuwenhao 617082766@qq.com
!
!  this file is part of project name.
!
!  this program is free software; you can redistribute it and/or modify
!  it under the terms of the mit license version 3 as
!  published by the free software foundation.
!
!  you should have received a copy of the mit license
!  along with this program. if not, see <https://mit-license.org/>.
!
!  @file     umat.f90
!  @brief    abaqus umat subroutine for sanisand constitutive model
!  @details  this file implements the user material subroutine (umat) for the
!            sanisand constitutive model in abaqus. the umat calculates stress
!            updates and consistent tangent moduli for soil materials under
!            various loading conditions. it incorporates elastic predictor -
!            plastic corrector algorithms, fabric evolution, and state-dependent
!            hardening.
!
!  @author   wuwenhao
!  @email    617082766@qq.com
!  @version  1.0.0.1
!  @date     2025/11/17
!  @license  mit massachusetts institute of technology (mit)
!-------------------------------------------------------------------------------
!  remark         : a state variable array of size nstatv to be
!  updated by the umat,which includes following variables:
!  statev(1) = void_ratio : current void ratio
!  statev(2) = p0
!  statev(3) = alpha[1] : alpha f11
!  statev(4) = alpha[2] : alpha f22
!  statev(5) = alpha[3] : alpha f33
!  statev(6) = alpha[4] : alpha f12,f21
!  statev(7) = alpha[5] : alpha f13,f31
!  statev(8) = alpha[6] : alpha f23,f32
!  statev(9) = alpha_in[1] : alpha_in a11
!  statev(10)= alpha_in[2] : alpha_in a22
!  statev(11)= alpha_in[3] : alpha_in a33
!  statev(12)= alpha_in[4] : alpha_in a12,a21
!  statev(13)= alpha_in[5] : alpha_in a13,a31
!  statev(14)= alpha_in[6] : alpha_in a23,a32
!  statev(15)= confining pressure
!  statev(16)= shear stress
!  statev(17)= ratio stress
!  statev(18)= the total of volumetric strain
!  statev(19)= the increment of shear strain
!  statev(20)= dkp
!  statev(21)= dilatancy
!  statev(22)= axial stress
!-------------------------------------------------------------------------------
!  change history :
!  <date>     | <version> | <author>       | <description>
!  2025/12/24 | 1.0.0.1   | wuwenhao      | create file
!*******************************************************************************
!> @brief abaqus umat subroutine for sanisand constitutive model
!>
!> @details this subroutine implements the user material subroutine (umat) for
!>          the sanisand constitutive model in abaqus. it calculates stress
!>          updates and consistent tangent moduli for soil materials under
!>          various loading conditions. the algorithm incorporates elastic
!>          predictor - plastic corrector methodology, fabric evolution, and
!>          state-dependent hardening. the subroutine handles both elastic and
!>          plastic loading paths, including return mapping to the yield surface.
!>
!> @param[in,out] stress    cauchy stress tensor (ntens)
!> @param[in,out] statev    state variable array (nstatv)
!> @param[out]    ddsdde    jacobian matrix (ntens x ntens)
!> @param[out]    sse       specific elastic strain energy
!> @param[out]    spd       specific plastic dissipation
!> @param[out]    scd       specific creep dissipation
!> @param[out]    rpl       volumetric heat generation per unit time
!> @param[out]    ddsddt    stress variation with temperature (ntens)
!> @param[out]    drplde    variation of rpl with strain increments (ntens)
!> @param[out]    drpldt    variation of rpl with temperature
!> @param[in]     stran     total strains at beginning of increment (ntens)
!> @param[in]     dstran    strain increments (ntens)
!> @param[in]     time      time array: time(1) = step time, time(2) = total time
!> @param[in]     dtime     time increment
!> @param[in]     temp      temperature at beginning of increment
!> @param[in]     dtemp     temperature increment
!> @param[in]     predef    predefined field variables array
!> @param[in]     dpred     increments of predefined field variables
!> @param[in]     cmname    user-defined material name (character*80)
!> @param[in]     ndi       number of direct stress components
!> @param[in]     nshr      number of engineering shear stress components
!> @param[in]     ntens     size of stress/strain array (ndi + nshr)
!> @param[in]     nstatv    number of state variables
!> @param[in]     props     material properties array (nprops)
!> @param[in]     nprops    number of material properties
!> @param[in]     coords    spatial coordinates of integration point
!> @param[in]     drot      rotation increment matrix (3x3)
!> @param[in,out] pnewdt    ratio of suggested new time increment
!> @param[in]     celent    characteristic element length
!> @param[in]     dfgrd0    deformation gradient at beginning of increment (3x3)
!> @param[in]     dfgrd1    deformation gradient at end of increment (3x3)
!> @param[in]     noel      element number
!> @param[in]     npt       integration point number
!> @param[in]     layer     layer number (for composite shells and layered solids)
!> @param[in]     kspt      section point number within the current layer
!> @param[in]     kstep     step number
!> @param[in]     kinc      increment number
!>
!> @author wuwenhao
!> @date 2025/11/17
!*******************************************************************************
module Fortran_umat_impl_mod
  use iso_c_binding, only: c_double, c_int
  use base_config
  use presolve_mod
  use tensor_opt_mod
  use elastic_mod
  use container_mod
  use plastic_mod
  use math_mod
  implicit none
  interface
    ! module subroutine Fortran_umat_intf(stress, statev, ddsdde, stran, dstran, &
    !                                     ntens, nstatv, pnewdt, &
    !                                     noel, npt, NUMBER) bind(c, name="Fortran_umat_intf")
    !   !DEC$ ATTRIBUTES DLLEXPORT :: Fortran_umat_intf
    !   real(c_double) :: stress(ntens), statev(nstatv), ddsdde(ntens, ntens), &
    !                     stran(ntens), dstran(ntens)
    !   real(c_double) :: pnewdt
    !   integer(c_int), value :: ntens, nstatv, noel, npt, NUMBER
    ! endsubroutine Fortran_umat_intf
    module subroutine umat_fortran_impl(stress, statev, ddsdde, stran, dstran, &
                                        ntens, nstatv, pnewdt, &
                                        noel, npt, NUMBER) bind(C, name="umat_fortran_impl")
      !DEC$ ATTRIBUTES DLLEXPORT :: umat_fortran_impl
      real(DP) :: stress(ntens), statev(nstatv), ddsdde(ntens, ntens), &
                  stran(ntens), dstran(ntens), pnewdt
      integer, intent(in), value :: ntens, nstatv, noel, npt, NUMBER
    endsubroutine umat_fortran_impl
  endinterface

contains
  ! module procedure Fortran_umat_intf
  ! call umat_fortran_impl(stress, statev, ddsdde, stran, dstran, &
  !                        ntens, nstatv, pnewdt, &
  !                        noel, npt, NUMBER)
  ! end procedure Fortran_umat_intf
  module procedure umat_fortran_impl
! variable declarations
!
! input/output variables
!-----------------------------------------------------------------------------
  type(torch) :: torch_
  type(elast) :: elast_
  type(plast) :: plast_
  type(math) :: math_
  real(dp), dimension(3, 3) :: depsln, deplsn_ela, depsln_res
  real(dp), dimension(3, 3) :: dsigma, sigma_final
  type(share_var) :: shvars
  type(state_var) :: stvars
  real(dp) :: voidr_ini, p0_ini
  real(dp), dimension(3, 3) :: sigma_ini, alpha_ini, alpha_in_ini
  real(dp) :: mean_etr, ftol_etr, alout, ftolr, total_ev, total_eq
  real(dp), dimension(3, 3, 3, 3) :: dsigde, dsdeyl, dsdetl
!
  real(dp) :: mean_final, shear, ratio_stress, dev, deq
  real(dp) :: dkp, dpla(2)
!-----------------------------------------------------------------------------
! initialize variable
  ftolr = 1.0d-6
!
  sigma_ini = -convert_array_to_tensor(stress, 1.0_dp)
  p0_ini = statev(2)
  alpha_ini = convert_array_to_tensor(statev(3:8))
! create share_var container
  shvars = share_var(sigma_ini, alpha_ini, p0_ini)
!
  depsln = -convert_array_to_tensor(dstran, 2.0_dp)
  voidr_ini = statev(1)
  alpha_in_ini = convert_array_to_tensor(statev(9:14))
  stvars = state_var(voidr_ini, alpha_in_ini, pnewdt)
!
  if(shvars%is_low()) then
    ! too low
  else
    mean_etr = math_%mean_with_depsln(shvars, stvars, depsln)
    ftol_etr = math_%ftol_with_depsln(shvars, stvars, depsln)
    if(ftol_etr <= ftolr .and. mean_etr >= tensno) then
      call math_%elastic_update(shvars, stvars, depsln, dsdetl)
      !
    elseif(ftol_etr > ftolr .or. (mean_etr < tensno .and. ftol_etr < ftolr)) then
      !
      alout = math_%intchc(shvars, stvars, depsln, ftolr)
      ! update variables
      deplsn_ela(:, :) = alout * depsln(:, :)
      depsln_res = (1.0_dp - alout) * depsln
      call math_%elastic_update(shvars, stvars, deplsn_ela, dsigde)
      !
      call math_%onyield(shvars, stvars, depsln_res, ftolr, dsdeyl, number, noel, npt)
      dsdetl = alout * dsigde + (1.0_dp - alout) * dsdeyl
    endif
  endif
!-----------------------------------------------------------------------------
!> update variable
! update ddsdde
  ddsdde = convert_tensor4_to_tensor2(dsdetl, ntens)
! update state variables
  stress(:) = -convert_tensor_to_array(shvars%get_sigma(), ntens)
  statev(1) = stvars%get_voidr()
  statev(2) = shvars%get_p0()
  statev(3:8) = convert_tensor_to_array(shvars%get_alpha(), 6)
!
  statev(9:14) = convert_tensor_to_array(stvars%get_alpha_in(), 6)
  pnewdt = stvars%get_pnewdt()
! confining pressure
  mean_final = torch_%trace(shvars%get_sigma()) / 3.0_dp
! shear_stress
  shear = torch_%shear(shvars%get_sigma())
! ratio stress
  ratio_stress = torch_%get_rm(shvars%get_sigma())
! the total of volumetric strain
  dev = torch_%trace(depsln)
  total_ev = statev(18) + dev * 100
! the increment of shear strain
  deq = torch_%shear(depsln)
  total_eq = statev(19) + deq * 100
! dkp
  dkp = plast_%get_dkp(shvars, stvars)
! dilatancy
  dpla = plast_%get_dilatancy(shvars, stvars)
! axial stress
  sigma_final = shvars%get_sigma()
  ! statev(15:23) = [mean_final, shear, ratio_stress, total_ev, total_eq, &
  !                  dkp, dpla(1), dpla(2), sigma_final(1, 1)]
! print

!   if(noel == 1 .and. npt == 1) then
!     call shvars_final%print()
! 11  format(a14, f6.2, 2x, a21, f6.2, /, &
!            a15, f6.2, 2x, a13, f6.2/, &
!            a20, f6.2, 2x, a16, f6.2, /, &
!            a6, e15.6, 2x, a8, f10.3, 2x, a8, f10.3)
!     write(6, 11) "mean stress = ", mean_final, ' deviatoric stress = ', shear, &
!       "ratio stress = ", ratio_stress, "void ratio = ", stvar_final%get_voidr(), &
!       "volumetric strain = ", statev(12), " shear strain = ", statev(13), &
!       "dkp = ", dkp, " dpla1 = ", dpla(1), "dpla2 = ", dpla(2)
!   endif
! !
  end procedure umat_fortran_impl
endmodule Fortran_umat_impl_mod
