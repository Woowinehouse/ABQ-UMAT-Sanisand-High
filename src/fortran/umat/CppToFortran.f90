!*****************************************************************************
!> @brief 模块简要说明
!>
!> @details 模块详细描述
!>
!> @author wuwenhao
!> @date 2026/04/11
!*****************************************************************************
module cpp_to_fortran_intf_mod

  implicit none
  interface
    function CXX_umat_Intf(stress_, statev_, ddsdde_, strain_, &
                           dstrain_, ntens_, nstatv_, pnewdt_, noel_, &
                           npt_, kspt_, jstep_, kinc_, NUMBER) result(errorCode) bind(c, name="CXX_umat_Intf")
      use iso_c_binding, only: c_double, c_int, c_size_t
      implicit none
      real(c_double), dimension(*) :: stress_
      real(c_double), dimension(*) :: statev_
      real(c_double), dimension(*) :: ddsdde_
      real(c_double), dimension(*) :: strain_
      real(c_double), dimension(*) :: dstrain_
      real(c_double) :: pnewdt_
      integer(c_int), value :: ntens_, nstatv_, noel_, npt_, kspt_, jstep_, kinc_, NUMBER
      integer(c_size_t) :: errorCode
    endfunction

    function siginiIntf(sigma, ntens, noel, npt, jobname, lenjobname, outdir, lenoutdir) result(errercode) bind(c, name="siginiIntf")
      use iso_c_binding, only: c_double, c_int, c_size_t, c_char
      real(c_double), intent(inout) :: sigma(*)
      integer(c_int) :: ntens, noel, npt, lenjobname, lenoutdir
      integer(c_size_t) :: errercode
      character(c_char), dimension(*) :: jobname, outdir

    endfunction

    function sdviniIntf(statev_, nstatv_, noel_, npt_) result(errercode) bind(c, name="sdviniIntf")
      use iso_c_binding, only: c_double, c_int, c_size_t
      real(c_double), dimension(*), intent(inout) :: statev_
      integer(c_int) :: nstatv_, noel_, npt_
      integer(c_size_t) :: errercode
    endfunction sdviniIntf
  endinterface
contains

endmodule cpp_to_fortran_intf_mod
