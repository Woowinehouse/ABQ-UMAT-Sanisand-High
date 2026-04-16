!*******************************************************************************
!> @brief tensor_torch_mod
!>
!> @details Module for tensor operations in continuum mechanics
!>
!> @author wuwenhao
!> @date 2025/11/27
!*******************************************************************************
submodule(tensor_opt_mod) tensor_torch_impl
  use Base_config
  implicit none
  type(Torch) :: torch_
contains
  !*****************************************************************************
  !> @brief Tensor4_ddot_tensor2
  !>
  !> @details Calculate the double dot product of a fourth-order tensor and
  !> a second-order tensor
  !> @param[in]  tensor4  Fourth-order tensor
  !> @param[in]  tensor2  Second-order tensor
  !>
  !> @return Resulting second-order tensor
  !*****************************************************************************
  module procedure Tensor4_ddot_tensor2
  integer(I4) :: i, j
  do j = 1, 3
    do i = 1, 3
      res(i, j) = sum(tensor4(i, j, :, :) * tensor2(:, :))
    enddo
  enddo
  end procedure Tensor4_ddot_tensor2
  !*****************************************************************************
  !> @brief Tensor2_ddot_tensor4
  !>
  !> @details Calculate the double dot product of a second-order tensor and
  !> a fourth-order tensor
  !> @param[in]  tensor2  Second-order tensor
  !> @param[in]  tensor4  Fourth-order tensor
  !>
  !> @return Resulting second-order tensor
  !*****************************************************************************
  module procedure Tensor2_ddot_tensor4
  integer(I4) :: i, j
  do j = 1, 3
    do i = 1, 3
      res(i, j) = sum(tensor2(:, :) * tensor4(:, :, i, j))
    enddo
  enddo
  end procedure Tensor2_ddot_tensor4
  !*****************************************************************************
  !> @brief Dyadic product implementation
  !>
  !> @details Calculate the dyadic (outer) product of two second-order
  !> tensors, resulting in a fourth-order tensor.
  !>
  !> @param[in]  tensorA  First second-order tensor (3x3)
  !> @param[in]  tensorB  Second second-order tensor (3x3)
  !>
  !> @return Fourth-order tensor (3x3x3x3)
  !*****************************************************************************
  module procedure Tensor2_dyad_tensor2
  integer(I4) :: i, j, k, l
  DO l = 1, 3
    DO k = 1, 3
      DO j = 1, 3
        DO i = 1, 3
          res(i, j, k, l) = tensorA(i, j) * tensorB(k, l)
        ENDDO
      ENDDO
    ENDDO
  ENDDO
  end procedure Tensor2_dyad_tensor2
  !*****************************************************************************
  !> @brief Print tensor implementation
  !>
  !> @details Print the components of a 3x3 tensor to standard output
  !> in formatted matrix layout for debugging and visualization.
  !>
  !> @param[in]  tensor  3x3 tensor to print
  !*****************************************************************************
  module procedure Print_impl
  integer(I4) :: i, j
  ! declaration
  do i = 1, 3
    write(6, '(A, I1, A)', advance='no') "Row ", i, ": ["
    do j = 1, 3
      write(6, '(F6.2)', advance='no') tensor(i, j)
      if(j < 3) write(6, '(A)', advance='no') ", "
    enddo
    write(6, '(A)') "]"
  enddo
  end procedure Print_impl
  !*****************************************************************************
  !> @brief Trace_Impl
  !>
  !> @details Calculate the trace of a stress tensor
  !>
  !> @param[in]  stress  Stress tensor
  !>
  !> @return Trace of the stress tensor
  !*****************************************************************************
  module procedure Trace_impl
  integer(I4) :: i
  !-----------------------------------------------------------------------------
  val = sum([(tensor(i, i), i=1, 3)])
  end procedure Trace_impl
  !*****************************************************************************
  module procedure pressure_impl
  !-----------------------------------------------------------------------------
  val = torch_%Trace(tensor) / 3.0D0
  end procedure pressure_impl
  !*****************************************************************************
  !> @brief Deviatoric_impl
  !>
  !> @details Calculate the deviatoric part of a stress tensor
  !>
  !> @param[in]  stress  Stress tensor
  !>
  !> @return Deviatoric stress tensor
  !*****************************************************************************
  module procedure Deviatoric_impl
  real(DP) :: P
  !-----------------------------------------------------------------------------
  P = torch_%Trace(tensor) / 3.0_DP
  res = tensor - P * DELTA
  end procedure Deviatoric_impl
  !*****************************************************************************
  !> @brief Sec_dev_invar_impl
  !>
  !> @details Calculate the second deviatoric invariant (J2) of a stress tensor
  !>
  !> @param[in]  stress  Stress tensor
  !>
  !> @return Second deviatoric invariant (J2)
  !*****************************************************************************
  module procedure Get_J2_impl
  real(DP), dimension(3, 3) :: S
  !-----------------------------------------------------------------------------
  S = torch_%Deviatoric(tensor)
  val = sum(S**2) / 2.0_DP
  end procedure Get_J2_impl
  !*****************************************************************************
  !> @brief Get_J3_impl
  !>
  !> @details Calculate the third deviatoric invariant (J3) of a stress tensor
  !>
  !> @param[in]  stress  Stress tensor
  !>
  !> @return Third deviatoric invariant (J3)
  !*****************************************************************************
  module procedure Get_J3_impl
  real(DP), dimension(3, 3) :: S, temp
  !-----------------------------------------------------------------------------
  S = torch_%Deviatoric(tensor)
  temp = matmul(S, matmul(S, S))
  val = torch_%Trace(temp) / 3.0_DP
  end procedure Get_J3_impl
  !*****************************************************************************
  !> @brief Ratio_impl
  !>
  !> @details Calculate the stress ratio tensor (deviatoric stress divided by
  !> mean stress)
  !> @param[in]  stress  Stress tensor
  !>
  !> @return Stress ratio tensor
  !*****************************************************************************
  module procedure Ratio_impl
  ! declaration
  real(DP), dimension(3, 3) :: S
  real(DP) :: P
  ! implementation
  P = torch_%Trace(tensor) / 3.0_DP
  P = merge(P, sign(EPS, P), abs(P) >= EPS)
  ! deviatoric stress
  S = torch_%Deviatoric(tensor)
  ! return tensor
  res = S / P
  end procedure Ratio_impl
  !*****************************************************************************
  !> @brief Get_sin3t_impl
  !>
  !> @details Calculate sin(3θ) where θ is the Lode angle
  !>
  !> @param[in]  stress  Stress tensor
  !>
  !> @return sin(3θ) value
  !*****************************************************************************
  module procedure Get_sin3t_impl
  real(DP) :: J2, J3
  !-----------------------------------------------------------------------------
  ! compute J2 and J3
  J2 = torch_%Get_J2(tensor)
  J2 = merge(J2, sign(EPS, J2), abs(J2) >= EPS)
  J3 = torch_%Get_J3(tensor)
  !
  val = -1.5_dp * DSQRT(3.0_dp) * J3 / (J2**1.5_dp)
  val = merge(val, sign(1.0_DP, val), abs(val) <= 1.0_DP)
  end procedure Get_sin3t_impl
  !*****************************************************************************
  !> @brief Shear_impl
  !>
  !> @details Calculate the shear stress (sqrt(J2))
  !>
  !> @param[in]  stress  Stress tensor
  !>
  !> @return Shear stress value
  !*****************************************************************************
  module procedure Shear_impl
  real(DP) :: J2
  !-----------------------------------------------------------------------------
  J2 = torch_%Get_J2(tensor)
  res = dsqrt(3.0_DP * J2)
  end procedure Shear_impl
  !*****************************************************************************
  !> @brief Normalize tensor implementation
  !>
  !> @details Normalize a tensor by dividing by its Frobenius norm,
  !> resulting in a unit tensor with the same direction.
  !> Handles near-zero norms with epsilon protection.
  !>
  !> @param[in]  tensor  Input 3x3 tensor
  !>
  !> @return Normalized unit tensor
  !*****************************************************************************
  module procedure Normalize_impl
  real(DP) :: norm
  !-----------------------------------------------------------------------------
  norm = sum(tensor**2)
  norm = max(dsqrt(norm), eps)
  res(:, :) = tensor(:, :) / norm
  end procedure Normalize_impl
  !*****************************************************************************
  !> @brief Calculate tensor norm implementation
  !>
  !> @details Calculate the Frobenius norm (Euclidean norm) of a 3x3 tensor.
  !> Returns zero for tensors with near-zero components.
  !>
  !> @param[in]  tensor  Input 3x3 tensor
  !>
  !> @return Frobenius norm of the tensor
  !*****************************************************************************
  module procedure Norm_impl
  real(DP) :: temp
  temp = sum(tensor**2)
  res = max(dsqrt(temp), 0.0_DP)
  end procedure Norm_impl
  !*****************************************************************************
  !> @brief Calculate cosine of angle between tensors implementation
  !>
  !> @details Calculate the cosine of the angle between two tensors
  !> using their double dot product and Frobenius norms.
  !> Includes protection against division by zero and ensures
  !> result stays within [-1, 1] range.
  !>
  !> @param[in]  tensorA  First 3x3 tensor
  !> @param[in]  tensorB  Second 3x3 tensor
  !>
  !> @return Cosine of angle between tensors (clamped to [-1, 1])
  !*****************************************************************************
  module procedure Get_cost_impl
  real(DP) :: norm_A, norm_B, dot_product
  !
  norm_A = torch_%norm(tensorA)
  norm_A = max(norm_A, EPS)
  norm_B = torch_%Norm(tensorB)
  norm_B = max(norm_B, EPS)
  dot_product = sum(tensorA * tensorB)
  !
  val = dot_product / (norm_A * norm_B)
  val = merge(val, sign(1.0_DP, val), abs(val) <= 1.0_DP)
  end procedure Get_cost_impl
  !*****************************************************************************
  !> @brief Calculate R_m parameter implementation
  !>
  !> @details Calculate the R_m parameter used in critical state soil
  !> mechanics. Computed as sqrt(3*J2) of the stress ratio tensor.
  !>
  !> @param[in]  tensor  Stress tensor
  !>
  !> @return R_m parameter value
  !*****************************************************************************
  module procedure Get_Rm_impl
  real(DP), dimension(3, 3) :: ratio
  ratio = torch_%Get_ratio(tensor)
  val = dsqrt(3.0_DP * torch_%Get_J2(ratio))
  end procedure Get_Rm_impl
  !*****************************************************************************
  !> @brief Calculate unit deviatoric tensor implementation
  !>
  !> @details Calculate the unit deviatoric tensor by first extracting
  !> the deviatoric part of the stress tensor, then normalizing it.
  !>
  !> @param[in]  tensor  Input stress tensor
  !>
  !> @return Unit deviatoric tensor
  !*****************************************************************************
  module procedure Get_unit_devivator_impl
  real(DP), dimension(3, 3) :: S
  !-----------------------------------------------------------------------------
  S = torch_%Deviatoric(tensor)
  res = torch_%Normalize(S)
  end procedure
!*******************************************************************************
endsubmodule tensor_torch_impl
