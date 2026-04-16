
!*****************************************************************************
!> @brief exception_mod
!>
!> @details 模块详细描述
!>
!> @author wuwenhao
!> @date 2025/11/27
!*****************************************************************************
module exception_mod
  implicit none
  private
  public :: ASSERT_TRUE, ASSERT_EQUAL, ASSERT_FLOAT_EQUAL
  type ErrorContext
    character(len=:), allocatable :: file_
    integer :: line_
    character(len=:), allocatable :: prefix_
  contains
    procedure, public, pass(this) :: throw_error => throw_error_impl
  endtype ErrorContext

  interface ErrorContext
    module procedure create_error_context
  endinterface ErrorContext

  interface
    ! ErrorContext(std::string file, I4 line, std::string prefix);
    !***************************************************************************
    !> @brief Create an error context object
    !>
    !> @details Constructs an ErrorContext object with file, line, and optional prefix
    !>
    !> @param[in] file Name of the source file where error occurred
    !> @param[in] line Line number in the source file
    !> @param[in] prefix Optional prefix message for error identification
    !>
    !> @return ErrorContext object containing error location information
    !***************************************************************************
    module function create_error_context(file, line, prefix) result(this)
      implicit none
      character(len=*), intent(in) :: file
      integer, intent(in) :: line
      character(len=*), intent(in), optional :: prefix
      type(ErrorContext) :: this
    endfunction create_error_context
    !***************************************************************************
    !> @brief Throw an error with context information
    !>
    !> @details Outputs an error message with file, line, and prefix
    !>          information, then terminates program
    !>
    !> @param[in] msg Error message to display
    !> @param[in] this ErrorContext object containing location information
    !***************************************************************************
    module subroutine throw_error_impl(msg, this)
      implicit none
      character(len=*), intent(in) :: msg
      class(ErrorContext), intent(in) :: this
    endsubroutine throw_error_impl
    !***************************************************************************
    !> @brief Assert that a logical expression is true
    !>
    !> @details Checks if the expression is true, if not, throws an error
    !>          with the provided message and location information
    !>
    !> @param[in] expr Logical expression to evaluate
    !> @param[in] msg Error message to display if assertion fails
    !> @param[in] file Name of the source file where assertion is called
    !> @param[in] line Line number in the source file
    !***************************************************************************
    module subroutine ASSERT_TRUE(expr, msg, file, line)
      logical, intent(in) :: expr
      character(len=*), intent(in) :: msg
      character(len=*), intent(in) :: file
      character(len=:), allocatable :: validator_msg
      integer, intent(in) :: line
    endsubroutine ASSERT_TRUE
  endinterface ! end interface
contains

  !*****************************************************************************
  !> @brief Assert that two integer values are equal
  !>
  !> @details Compares two integer values, if they are not equal, throws an error
  !>          with the provided message and location information
  !>
  !> @param[in] lhs Left-hand side integer value
  !> @param[in] rhs Right-hand side integer value
  !> @param[in] msg Error message to display if assertion fails
  !> @param[in] file Name of the source file where assertion is called
  !> @param[in] line Line number in the source file
  !*****************************************************************************
  subroutine ASSERT_EQUAL(lhs, rhs, msg, file, line)
    implicit none
    ! Input/Output variables
    integer, intent(in) :: lhs, rhs, line
    character(len=*), intent(in) :: msg, file
    character(len=:), allocatable :: validator_msg
    type(ErrorContext) Error
    integer :: msg_len
    !
    ! 分配足够大的缓冲区用于整数输出
    ! I0格式：对于32位整数，最大长度约为11（包括负号）
    msg_len = 50  ! 足够容纳 "Value X/= expected Y"
    allocate(character(len=msg_len) :: validator_msg)
    write(validator_msg, '("Value ",I0, "/= expected ", I0)') &
      lhs, rhs
    if(lhs /= rhs) then
      Error = ErrorContext(file, line, msg)
      call Error%throw_error(validator_msg)
    endif
  endsubroutine ASSERT_EQUAL
  !*****************************************************************************
  !> @brief Assert that two floating-point values are equal within tolerance
  !>
  !> @details Compares two floating-point values, if the absolute difference
  !>          exceeds EPS tolerance, throws an error with the provided message
  !>          and location information
  !>
  !> @param[in] lhs Left-hand side floating-point value
  !> @param[in] rhs Right-hand side floating-point value
  !> @param[in] msg Error message to display if assertion fails
  !> @param[in] file Name of the source file where assertion is called
  !> @param[in] line Line number in the source file
  !*****************************************************************************
  subroutine ASSERT_FLOAT_EQUAL(lhs, rhs, msg, file, line)
    use Base_config, only: DP, EPS
    implicit none
    ! Input/Output variables
    real(DP), intent(in) :: lhs, rhs
    character(len=*), intent(in) :: msg, file
    integer, intent(in) :: line
    character(len=:), allocatable :: validator_msg
    type(ErrorContext) Error
    integer :: msg_len
    msg_len = 200
    allocate(character(len=msg_len) :: validator_msg)
    write(validator_msg, '("Value ", G0, "/=expected ", G0," (tolerance: ",G0,")" )') &
      lhs, rhs, EPS
    if(abs(lhs - rhs) > EPS) then
      Error = ErrorContext(file, line, msg)
      call Error%throw_error(validator_msg)
    endif
  endsubroutine ASSERT_FLOAT_EQUAL

endmodule exception_mod
