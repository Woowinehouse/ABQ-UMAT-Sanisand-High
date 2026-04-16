!*****************************************************************************
!> @brief Implementation of exception module procedures
!>
!> @details Contains the implementation of error handling and assertion
!>          procedures defined in the exception_mod interface
!>
!> @author wuwenhao
!> @date 2025/11/27
!*****************************************************************************
submodule(exception_mod) exception_impl
  implicit none
contains
  !*****************************************************************************
  !> @brief Create an error context object
  !>
  !> @details Implementation of the ErrorContext constructor that stores
  !>          file, line, and optional prefix information for error reporting
  !>
  !> @param[in] file Name of the source file where error occurred
  !> @param[in] line Line number in the source file
  !> @param[in] prefix Optional prefix message for error identification
  !>
  !> @return ErrorContext object containing error location information
  !*****************************************************************************
  module procedure create_error_context
  this%file_ = file
  this%line_ = line
  if(present(prefix)) then
    this%prefix_ = prefix
  else
    this%prefix_ = "Validation failed"
  endif
  end procedure create_error_context
  !*****************************************************************************
  !> @brief Throw an error with context information
  !>
  !> @details Implementation of error throwing procedure that formats
  !>          and outputs an error message with file, line, and prefix
  !>          information, then terminates the program
  !>
  !> @param[in] msg Error message to display
  !> @param[in] this ErrorContext object containing location information
  !*****************************************************************************
  module procedure throw_error_impl
  character(len=:), allocatable :: full_msg
  integer :: msg_len
  !
  ! 计算所需消息长度并分配缓冲区
  msg_len = len(this%prefix_) + len(msg) + len(this%file_) + 30  ! 额外空间用于格式文本和行号
  allocate(character(len=msg_len) :: full_msg)
  !
  ! 格式化错误消息
  write(full_msg, '(A,": ", A, " Error at: ",A, ": ",I0)') &
    this%prefix_, msg, this%file_, this%line_
  ! 在标准环境中使用单元0输出到标准错误
  write(0, '(A)') trim(full_msg)
  error stop trim(full_msg)
  end procedure throw_error_impl
  !*****************************************************************************
  !> @brief Assert that a logical expression is true
  !>
  !> @details Implementation of assertion procedure that checks if the
  !>          expression is true, if not, throws an error with the
  !>          provided message and location information
  !>
  !> @param[in] expr Logical expression to evaluate
  !> @param[in] msg Error message to display if assertion fails
  !> @param[in] file Name of the source file where assertion is called
  !> @param[in] line Line number in the source file
  !*****************************************************************************
  module procedure ASSERT_TRUE
  character(len=:), allocatable :: validator_msg
  type(ErrorContext) Error
  !
  validator_msg = "Condition evaluated to false"
  if(.not. expr) then
    Error = ErrorContext(file, line, msg)
    call Error%throw_error(validator_msg)
  endif
  end procedure ASSERT_TRUE
endsubmodule
