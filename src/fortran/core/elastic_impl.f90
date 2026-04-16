submodule(elastic_mod) elastic_impl
  use Base_config
  use Material_config
  use tensor_opt_mod
  implicit none
  type(Torch) torch_
  type(Elast) elast_
contains
  !*****************************************************************************
  module procedure Get_principal_impl
  real(dp) :: work(300)
  integer(I4) :: info
  !-----------------------------------------------------------------------------
  call DSYEV('V', 'U', 3, tensor, 3, res, work, 300, info)
  return
  end procedure Get_principal_impl
  !*****************************************************************************
  module procedure Get_dnorm_impl
  real(DP), dimension(3, 3) :: R
  real(DP), dimension(3) :: pris
  logical :: is_isotropic
  !-----------------------------------------------------------------------------
  pris = elast_%Get_principal(shvars%get_sigma())
  is_isotropic = (abs(pris(1) - pris(2)) < EPS) &
                 .and. (abs(pris(2) - pris(3)) <= EPS) &
                 .and. (abs(pris(1) - pris(3)) <= EPS)
  if(is_isotropic) then
    dnorm(:, :) = 0.0_DP
    dnorm(1, 1) = dsqrt(2.0_DP / 3.0_DP)
    dnorm(2, 2) = -dsqrt(2.0_DP / 3.0_DP) / 2.D0
    dnorm(3, 3) = -dsqrt(2.0_DP / 3.0_DP) / 2.D0
  else
    R(:, :) = torch_%Get_ratio(shvars%get_sigma())
    dnorm = torch_%Normalize(R(:, :) - shvars%get_alpha())
  endif
  end procedure Get_dnorm_impl
  !*****************************************************************************
  module procedure Get_cos3t_impl
  real(DP), dimension(3, 3) :: dnorm, temp
  !-----------------------------------------------------------------------------
  dnorm(:, :) = Get_dnorm_impl(shvars)
  temp = matmul(dnorm, matmul(dnorm, dnorm))
  cos3t = dsqrt(6.0D0) * torch_%trace(temp)
  end procedure Get_cos3t_impl
  !*****************************************************************************
  !> @brief 函数简要说明
  !>
  !> @details 函数详细描述
  !>
  !> @param[in]  参数名 输入参数说明
  !> @param[out] 参数名 输出参数说明
  !>
  !> @return 返回值说明
  !*****************************************************************************
  module procedure Get_gtheta_impl
  real(DP) :: cos3t, temp
  !-----------------------------------------------------------------------------
  cos3t = elast_%Get_cos3t(shvars)
  temp = PARAM%C**(1.0D0 / PARAM%W)
  gtheta = (((1.D0 + temp) + (1.D0 - temp) * cos3t) / 2.0D0)**PARAM%W
  end procedure Get_gtheta_impl
  !*****************************************************************************
  !> @brief Get_Fr_impl
  !>
  !> @details 函数详细描述
  !>
  !> @param[in]  参数名 输入参数说明
  !> @param[out] 参数名 输出参数说明
  !>
  !> @return 返回值说明
  !*****************************************************************************
  module procedure Get_lamda_impl
  real(DP) :: gtheta
  !-----------------------------------------------------------------------------
  gtheta = elast_%Get_gtheta(shvars)
  lamda = PARAM%LAMDAR + 3.0D0 / 2.0D0 * sum(shvars%get_alpha()**2) &
          / (gtheta * PARAM%ALPHAC)**2 * (PARAM%LAMDACS - PARAM%LAMDAR)
  end procedure Get_lamda_impl
  !*****************************************************************************
  module procedure Get_bulk_impl
  real(DP) :: P, BH_1, BH_2
  real(DP) :: lamda_a, temp
  !-----------------------------------------------------------------------------
  P = torch_%pressure(shvars%get_sigma())
  lamda_a = elast_%Get_lamda(shvars)
  temp = lamda_a * (P / PA)**PARAM%KSI
  BH_1 = (stvars%get_voidr() - PARAM%VOIDRL) * PARAM%K * PARAM%KSI * temp
  BH_2 = PARAM%Y * (PARAM%VOIDREF - PARAM%VOIDRL) * exp(-lamda_a * (P / PA)**PARAM%KSI) &
         * (PARAM%PE**PARAM%Y * P) / (P + PARAM%CH * PA)**(PARAM%Y + 1)
  bulk = ((1.0D0 + stvars%get_voidr()) * P) / (BH_1 + BH_2)
  end procedure Get_bulk_impl
  !*****************************************************************************
  module procedure Get_Gv_impl
  real(DP) :: p, nu
  !-----------------------------------------------------------------------------
  p = torch_%pressure(shvars%get_sigma())
  nu = PARAM%nu_min + (PARAM%nu_max - PARAM%nu_min) * exp(-PARAM%nu_v * (P / PA))
  Gv = 3.0 * (1.0D0 - 2.0D0 * nu) / (2.0D0 * (1.0D0 + nu))
  end procedure Get_Gv_impl
  !*****************************************************************************
  module procedure Get_shear_impl
  real(DP) :: bulk, Gv
  !-----------------------------------------------------------------------------
  bulk = elast_%Get_bulk(shvars, stvars)
  Gv = elast_%Get_Gv(shvars)
  ! shear modulus
  shear = Gv * bulk
  end procedure Get_shear_impl
  !*****************************************************************************
  !> @brief 函数简要说明
  !>
  !> @details 函数详细描述
  !>
  !> @param[in]  参数名 输入参数说明
  !> @param[out] 参数名 输出参数说明
  !>
  !> @return 返回值说明
  !*****************************************************************************
  module procedure Yield_distance_impl
  real(DP) :: s(3, 3), p
  !-----------------------------------------------------------------------------
  s = torch_%Deviatoric(shvars%get_sigma())
  p = torch_%pressure(shvars%get_sigma())
  ftol = 3.0D0 / 2.0D0 * sum((S - P * shvars%get_alpha())**2) - (PARAM%FM * P)**2 &
         * (1.0D0 - (P / shvars%get_p0())**PARAM%FN)
  end procedure Yield_distance_impl
  !*****************************************************************************
  !> @brief get_stiffness_impl
  !>
  !> @details 函数详细描述
  !>
  !> @param[in] shvars%get_sigma()(:,:) : current shvars%get_sigma()(:,:) tensor(3x3)
  !> @param[in] void_ratio : current void ratio(scalar)
  !> @param[out] stiffness : a stiffness tensor of size 3x3x3x3
  !>
  !> @return 返回值说明
  !*****************************************************************************
  module procedure get_stiffness_impl
  real(DP) :: shear, bulk
  integer(I4) :: i, j, k, l
  shear = elast_%Get_shear(shvars, stvars)
  bulk = elast_%Get_bulk(shvars, stvars)
  ! stiffness tensor
  do l = 1, 3
    do k = 1, 3
      do j = 1, 3
        do i = 1, 3
          stiffness(i, j, k, l) = &
            (bulk - 2.0_DP / 3.0_DP * shear) * DELTA(i, j) * DELTA(k, l) + &
            shear * (DELTA(i, k) * DELTA(j, l) + DELTA(i, l) * DELTA(j, k))
        enddo
      enddo
    enddo
  enddo
  end procedure get_stiffness_impl
  !*****************************************************************************
  module procedure calc_dsigma_impl
  real(DP), dimension(3, 3, 3, 3) :: stiff
  !-----------------------------------------------------------------------------
  stiff = get_stiffness_impl(shvars, stvars)
  dsigma = stiff.ddot.depsln
  end procedure calc_dsigma_impl
!*******************************************************************************
endsubmodule elastic_impl
