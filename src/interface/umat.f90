
subroutine Umat(stress, statev, ddsdde, sse, spd, scd, &
                rpl, ddsddt, drplde, drpldt, &
                stran, dstran, time, dtime, temp, dtemp, predef, dpred, &
                cmname, ndi, nshr, ntens, nstatv, props, nprops, &
                coords, drot, pnewdt, celent, dfgrd0, dfgrd1, &
                noel, npt, layer, kspt, kstep, kinc)
  use iso_c_binding, only: c_size_t
  use cpp_to_fortran_intf_mod, only: CXX_umat_Intf
  use Fortran_umat_impl_mod, only: umat_fortran_impl
  use presolve_mod, only: abaqus_debug
  ! Variable declarations
  implicit none
  !----------------------------------------------------------------------------
  character * 80 cmname
  ! Input/Output variables
  integer :: ndi, nshr, ntens, nstatv, nprops, noel, npt, &
             layer, kspt, kstep, kinc
  real(8) :: stress(ntens), statev(nstatv), &
             ddsdde(ntens, ntens), ddsddt(ntens), drplde(ntens), &
             stran(ntens), dstran(ntens), time(2), props(nprops), &
             coords(3), drot(3, 3), dfgrd0(3, 3), dfgrd1(3, 3), &
             sse, spd, scd, rpl, drpldt, dtime, temp, dtemp, &
             predef, dpred, celent, pnewdt

  !----------------------------------------------------------------------------
  real(8) :: stress_c(ntens), statev_c(nstatv), ddsdde_c(ntens, ntens)
  real(8) :: stress_f(ntens), statev_f(nstatv), ddsdde_f(ntens, ntens)
  real(8) :: stress_deom(ntens), statev_deom(nstatv), ddsdde_deom(ntens, ntens)
  logical :: stress_is_consistent, statev_is_consistent, ddsdde_is_consistent
  integer, SAVE :: NUMBER = 0 ! static variable
  integer(c_size_t) :: err
  real(8) :: alpha(6), alpha_ini(6)
  integer :: tempvar
  integer :: tid
  integer(kind=8) :: count, count_rate, t_start, t_end
  real(8) :: elapsed_time
  !----------------------------------------------------------------------------

  if(noel == 1 .and. npt == 1) then
    write(6, *) '=============================================================='
    write(6, *) 'noel = ', noel, ' npt = ', npt, ' number = ', NUMBER
    write(6, *) '=============================================================='
    call abaqus_debug(noel, npt, NUMBER, 1, 1, 31, "Umat")
    NUMBER = NUMBER + 1
  endif

  ! if(noel == 1 .and. npt == 1) then

  !   ! 初始化两套内存
  !   stress_f = stress
  !   statev_f = statev
  !   ddsdde_f = ddsdde

  !   stress_c = stress
  !   statev_c = statev
  !   ddsdde_c = ddsdde
  !   ! fortran
  !   call umat_fortran_impl(stress_f, statev_f, ddsdde_f, stran, dstran, &
  !                          ntens, nstatv, pnewdt, &
  !                          noel, npt, NUMBER)

  !   stress_deom = merge(stress_f, sign(EPS, stress_f), abs(stress_f) >= EPS)
  !   statev_deom = merge(statev_f, sign(EPS, statev_f), abs(statev_f) >= EPS)
  !   ddsdde_deom = merge(ddsdde_f, sign(EPS, ddsdde_f), abs(ddsdde_f) >= EPS)
  !   stress_is_consistent = all(abs(stress_c - stress_f) / stress_deom < 1e-6)
  !   statev_is_consistent = all(abs(statev_c - statev_f) / statev_deom < 1e-6)
  !   ddsdde_is_consistent = all(abs(ddsdde_c - ddsdde_f) / ddsdde_deom < 1e-6)
  !   if(stress_is_consistent) then
  !     write(6, '(A)') "stress is consistent between fortran and c++"
  !   else
  !     write(6, '(A)') "stress is inconsistent between fortran and c++"
  !   endif
  !   if(statev_is_consistent) then
  !     write(6, '(A)') "statev is consistent between fortran and c++"
  !   else
  !     write(6, '(A)') "statev is inconsistent between fortran and c++"
  !   endif
  !   if(ddsdde_is_consistent) then
  !     write(6, '(A)') "ddsdde is consistent between fortran and c++"
  !   else
  !     write(6, '(A)') "ddsdde is inconsistent between fortran and c++"
  !   endif
  !   if(stress_is_consistent .and. statev_is_consistent .and. ddsdde_is_consistent) then
  !     stress = stress_f
  !     statev = statev_f
  !     ddsdde = ddsdde_f
  !   else
  !     write(7, *) "Error: the results from fortran and c++ are inconsistent in the umatIntf function at: noel = ", noel, "npt = ", npt, "NUMBER = ", NUMBER
  !     call exit
  !   endif

  ! else
  !   call umat_fortran_impl(stress, statev, ddsdde, stran, dstran, &
  !                          ntens, nstatv, pnewdt, &
  !                          noel, npt, NUMBER)
  ! endif
  err = CXX_umat_Intf(stress, statev, ddsdde, stran, dstran, ntens, nstatv, pnewdt, noel, npt, &
                      kspt, kstep, kinc, NUMBER)
  if(err .ne. 0) then
    ! occur error
    select case(err)
    case(1)
      write(7, *) "NanError : there are some variables are not a number in the umatIntf function"
    case(2)
      write(7, *) "InfError : there are some variables are infinity in the umatIntf function"
    case(3)
      write(7, *) "IosError : there are some variables are negative in the umatIntf function"
    case(4) !ValueError = 4,
      write(7, *) "ValueError : there are some variables are out of bounds in the umatIntf function"
    case(5)
      write(7, *) "TypeError : there are some variables are of the wrong type in the umatIntf function"
    case(6)
      write(7, *) "GradError : there are some memory errors in the umatIntf function"
    case(7) !IterError = 7,
      write(7, *) "IterError : there are some iteration errors in the umatIntf function"
      pnewdt = 0.5d0; 
    case(8)!InputError
      write(7, *) "InputError : there are some input errors in the umatIntf function"
    case(9)
      write(7, *) "UnknownError : there are some unknown errors in the umatIntf function"
    endselect
  endif
  !----------------------------------------------------------------------------
  ! print variables
  !----------------------------------------------------------------------------
  ! print stress
  if(noel == 1 .and. npt == 1) then
    select case(ntens)
    case(3)
    case(4)
    case(6)
      write(6, '(A)') "stress tensor:"
      write(6, '(F12.8, 2x, F12.8, 2x,F12.8)', advance='yes') - stress(1), -stress(4), -stress(5)
      write(6, '(F12.8, 2x, F12.8, 2x,F12.8)', advance='yes') - stress(4), -stress(2), -stress(6)
      write(6, '(F12.8, 2x, F12.8, 2x,F12.8)', advance='yes') - stress(5), -stress(6), -stress(3)
      write(6, '(A)') "p0 tensor:"
      write(6, '(F12.8)') statev(2)
      write(6, '(A)') "alpha tensor:"
      alpha = statev(3:8)
      write(6, '(ES12.4, 2x, ES12.4, 2x,ES12.4)', advance='yes') alpha(1), alpha(4), alpha(5)
      write(6, '(ES12.4, 2x, ES12.4, 2x,ES12.4)', advance='yes') alpha(4), alpha(2), alpha(6)
      write(6, '(ES12.4, 2x, ES12.4, 2x,ES12.4)', advance='yes') alpha(5), alpha(6), alpha(3)
      alpha_ini = statev(9:14)
      write(6, '(A)') "alpha_ini tensor:"
      write(6, '(ES12.4, 2x, ES12.4, 2x,ES12.4)', advance='yes') alpha_ini(1), alpha_ini(4), alpha_ini(5)
      write(6, '(ES12.4, 2x, ES12.4, 2x,ES12.4)', advance='yes') alpha_ini(4), alpha_ini(2), alpha_ini(6)
      write(6, '(ES12.4, 2x, ES12.4, 2x,ES12.4)', advance='yes') alpha_ini(5), alpha_ini(6), alpha_ini(3)
    endselect
  endif
endsubroutine Umat

subroutine Sigini(sigma, coords, ntens, ncrds, noel, npt, layer, &
                  kspt, lrebar, names)
  use iso_c_binding, only: c_size_t, c_double, c_int, c_intptr_t
  use cpp_to_fortran_intf_mod
  implicit none
  integer, intent(in) :: ntens, ncrds, noel, npt, layer, kspt, lrebar
  real(8), intent(inout) :: sigma(ntens)
  real(8), intent(in) :: coords(ncrds)
  character(len=80), intent(in) :: names(2)
  !----------------------------------------------------------------------------
  character*256 :: JOBNAME, OUTDIR
  integer :: LENJOBNAME, LENOUTDIR
  integer(c_size_t) :: err
  logical :: firstrun = .true.
  integer tempvar
  !----------------------------------------------------------------------------
  ! if(firstrun) then
  !   write(*, *) "please input an integer"
  !   read(*, *) tempvar
  ! endif
  CALL GETJOBNAME(JOBNAME, LENJOBNAME)
  !
  CALL GETOUTDIR(OUTDIR, LENOUTDIR)
  err = siginiIntf(sigma, ntens, noel, npt, JOBNAME, LENJOBNAME, OUTDIR, LENOUTDIR)

  return
endsubroutine Sigini

SUBROUTINE Sdvini(statev, coords, nstatv, ncrds, noel, npt, layer, kspt)
  use iso_c_binding, only: c_size_t, c_double, c_int
  use cpp_to_fortran_intf_mod
  implicit none
  !----------------------------------------------------------------------------
  integer, intent(in) :: nstatv, ncrds, noel, npt, layer, kspt
  real(8), intent(in) :: coords(ncrds)
  real(8), intent(inout) :: statev(nstatv)
  integer(c_size_t) :: err
  !----------------------------------------------------------------------------
  err = sdviniIntf(statev, nstatv, noel, npt)
  RETURN
ENDSUBROUTINE Sdvini
