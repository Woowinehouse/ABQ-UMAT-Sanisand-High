!DIR$ FREEFORM
!*****************************************************************************
!> @brief Preprocessing module
!>
!> @details This module provides utilities for data conversion and
!> preprocessing operations, including tensor-array conversions,
!> rotation matrix generation, and debugging utilities.
!>
!> @author wuwenhao
!> @date 2025/11/17
!*****************************************************************************
module presolve_mod
  use Base_config, only: DP
  implicit none
  public :: Convert_array_to_tensor, Convert_tensor_to_array, Print_array
  public :: Convert_tensor4_to_tensor2
  public :: abaqus_debug, Get_rotation_matrix
  ! Public interface

contains
  !**************************************************************************
  !> @brief Convert array to tensor
  !>
  !> @details Convert a 1D array to a 3x3 2D tensor representation.
  !> Commonly used for converting stress/strain vectors to tensor form.
  !>
  !> @param[in]  array Input vector (size 6 for symmetric tensor)
  !> @param[in]  scalar Optional scaling factor
  !>
  !> @return 3x3 tensor representation
  !**************************************************************************
  function Convert_array_to_tensor(array, scalar) result(tensor)
    real(DP), intent(in) :: array(:)
    real(DP), intent(in), optional :: scalar
    real(DP), dimension(3, 3) :: tensor
    real(DP) :: scalar_
    if(present(scalar)) then
      scalar_ = scalar
    else
      scalar_ = 1.0d0
    endif
    select case(size(array))
    case(3)
      tensor(1, 1) = array(1)
      tensor(2, 2) = array(2)
      tensor(1, 2) = array(3) / scalar_
      tensor(2, 1) = array(3) / scalar_
    case(4)
      tensor(1, 1) = array(1)
      tensor(2, 2) = array(2)
      tensor(3, 3) = array(3)
      tensor(1, 2) = array(4) / scalar_
      tensor(2, 1) = array(4) / scalar_
    case(6)
      tensor(1, 1) = array(1)
      tensor(2, 2) = array(2)
      tensor(3, 3) = array(3)
      tensor(1, 2) = array(4) / scalar_
      tensor(2, 1) = array(4) / scalar_
      tensor(1, 3) = array(5) / scalar_
      tensor(3, 1) = array(5) / scalar_
      tensor(2, 3) = array(6) / scalar_
      tensor(3, 2) = array(6) / scalar_
    endselect
  endfunction Convert_array_to_tensor
  !**************************************************************************
  !> @brief Convert tensor to array
  !>
  !> @details Convert a 3x3 2D tensor to a 1D array representation.
  !> For symmetric tensors, returns Voigt notation vector.
  !>
  !> @param[in]  tensor Input 3x3 tensor
  !> @param[in]  size Output array size (6 for symmetric, 9 for full)
  !> @param[in]  scalar Optional integer(I4) parameter
  !>
  !> @return 1D array representation of the tensor
  !**************************************************************************
  function Convert_tensor_to_array(tensor, size, scalar) result(array)
    real(DP), dimension(3, 3), intent(in) :: tensor
    integer, value, intent(in) :: size
    integer, intent(in), optional :: scalar
    real(DP), dimension(size) :: array
    real(DP) :: scalar_
    if(present(scalar)) then
      scalar_ = scalar
    else
      scalar_ = 1.0_DP
    endif
    select case(size)
    case(3)
      array(1) = tensor(1, 1)
      array(2) = tensor(2, 2)
      array(3) = tensor(1, 2) * scalar_
    case(4)
      array(1) = tensor(1, 1)
      array(2) = tensor(2, 2)
      array(3) = tensor(3, 3)
      array(4) = tensor(1, 2) * scalar_
    case(6)
      array(1) = tensor(1, 1)
      array(2) = tensor(2, 2)
      array(3) = tensor(3, 3)
      array(4) = tensor(1, 2) * scalar_
      array(5) = tensor(1, 3) * scalar_
      array(6) = tensor(2, 3) * scalar_
    endselect
  endfunction Convert_tensor_to_array
  !**************************************************************************
  !> @brief Convert fourth-order tensor to second-order matrix
  !>
  !> @details Convert a 3x3x3x3 fourth-order tensor to a 2D matrix
  !> representation using Voigt notation.
  !>
  !> @param[in]  tensor4 Input fourth-order tensor (3x3x3x3)
  !> @param[in]  size Output matrix size (6x6 for symmetric)
  !>
  !> @return 2D matrix representation
  !**************************************************************************
  function Convert_tensor4_to_tensor2(tensor4, martix_size) result(tensor2)
    real(DP), dimension(3, 3, 3, 3), intent(in) :: tensor4
    integer, value, intent(in) :: martix_size
    real(DP), dimension(martix_size, martix_size) :: tensor2
    integer, parameter :: FST(6) = [1, 2, 3, 1, 1, 2]
    integer, parameter :: SCD(6) = [1, 2, 3, 2, 3, 3]
    integer :: i1, i2, j1, j2, i, j
    !
    do j = 1, martix_size
      j1 = FST(j)
      j2 = SCD(j)
      do i = 1, martix_size
        i1 = FST(i)
        i2 = SCD(i)
        tensor2(i, j) = 0.25d0 * (tensor4(i1, i2, j1, j2) + tensor4(i1, i2, j2, j1) + &
                                  tensor4(i2, i1, j1, j2) + tensor4(i2, i1, j2, j1))
      enddo
    enddo
  endfunction Convert_tensor4_to_tensor2
  !**************************************************************************
  !> @brief Generate rotation matrix
  !>
  !> @details Generate a 3x3 rotation matrix for given angle and axis
!> using Rodrigues rotation formula.
  !>
  !> @param[in]  angle Rotation angle in radians
  !> @param[in]  axis  Rotation axis vector (3D)
  !>
  !> @return 3x3 rotation matrix
  !**************************************************************************
  function Get_rotation_matrix(angle, axis) result(rot_matrix)
    real(DP), intent(in) :: angle
    real(DP), intent(in), dimension(3) :: axis
    real(DP), dimension(3, 3) :: rot_matrix
    real(DP), dimension(3) :: uaxis
    real(DP) :: cos, sin, temp, ux, uy, uz
    !
    if(axis(1) == 0.0_DP .and. axis(2) == 0.0_DP .and. axis(3) == 0.0_DP) then
      write(6, *) "Error: Rotation axis cannot be zero vector."
      call exit(1)
    endif
    uaxis = axis / dsqrt(sum(axis**2))
    ux = uaxis(1)
    uy = uaxis(2)
    uz = uaxis(3)
    cos = dcos(angle)
    sin = dsin(angle)
    temp = 1.0_DP - cos
    rot_matrix(1, 1) = cos + ux * ux * temp
    rot_matrix(1, 2) = ux * uy * temp - uz * sin
    rot_matrix(1, 3) = ux * uz * temp + uy * sin
    rot_matrix(2, 1) = uy * ux * temp + uz * sin
    rot_matrix(2, 2) = cos + uy * uy * temp
    rot_matrix(2, 3) = uy * uz * temp - ux * sin
    rot_matrix(3, 1) = uz * ux * temp - uy * sin
    rot_matrix(3, 2) = uz * uy * temp + ux * sin
    rot_matrix(3, 3) = cos + uz * uz * temp
  endfunction Get_rotation_matrix
  !**************************************************************************
  !> @brief Print array contents
  !>
  !> @details Print the contents of a 1D array to standard output
  !> for debugging purposes.
  !>
  !> @param[in]  array Input array to print
  !**************************************************************************
  Subroutine Print_array(array)
    real(DP), intent(in), dimension(:) :: array
    integer :: i, n

    n = size(array)
    print *, "Array size: ", n
    print *, "Array elements:"
    do i = 1, n
      print *, "  array(", i, ") = ", array(i)
    enddo
  endSubroutine
  !**************************************************************************
  !> @brief ABAQUS debugging utility
  !>
  !> @details Print debugging information for ABAQUS UMAT simulations,
  !> including element number, integration point, and iteration count.
  !>
  !> @param[in]  noel_num  Number of elements
  !> @param[in]  npt_num   Number of integration points
  !> @param[in]  num       Additional numerical parameter
  !> @param[in]  noel      Current element number
  !> @param[in]  npt       Current integration point number
  !> @param[in]  iteration Current iteration number
  !> @param[in]  names     Descriptive name for debugging output
  !**************************************************************************
  subroutine abaqus_debug(noel, npt, number, noel_num, npt_num, iteration, names) bind(C, name="abaqus_debug")
    !DEC$ ATTRIBUTES DLLEXPORT :: abaqus_debug
    integer, intent(in) :: noel_num, npt_num, iteration
    integer, intent(in) :: noel, npt, number
    character(len=*), intent(in) :: names
    logical :: firstrun = .true.
    integer :: tempvar
    if((noel == noel) .and. (npt_num == npt) .and. number >= iteration) then
      write(*, *) "debug in ", trim(names)
      if(firstrun) then
        write(*, *) "please input an integer"
        read(*, *) tempvar
      endif
    endif
  endsubroutine abaqus_debug
endmodule presolve_mod
