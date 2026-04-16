submodule(plastic_mod) plastic_impl
  use Base_config
  use Material_config
  use tensor_opt_mod
  use elastic_mod
  implicit none
  type(Torch) :: torch_
  type(Elast) :: elast_
  type(plast) :: plast_
contains
  !*****************************************************************************
  !> @brief Calculate partial derivative of yield function with respect to stress
  !>
  !> @details This function computes the partial derivative of the yield
  !>          function with respect to the stress tensor (∂f/∂σ). This
  !>          derivative is essential for computing the plastic flow
  !>          direction and consistent tangent modulus in plasticity
  !>          algorithms.
  !>
  !> @param[in] shvars   Shared variables containing current stress tensor
  !>
  !> @return res         Partial derivative ∂f/∂σ (3x3 tensor)
  !*****************************************************************************
  module procedure get_pfsig_impl
  real(DP), dimension(3, 3) :: pfps, s, temp
  real(DP) :: p, pfpp
  !-----------------------------------------------------------------------------
  s = torch_%Deviatoric(shvars%get_sigma())
  p = torch_%pressure(shvars%get_sigma())
  temp = s - p * shvars%get_alpha()
  pfps = 3.0D0 * temp
  pfpp = -3.0D0 * sum(shvars%get_alpha() * temp) - 2.0D0 * PARAM%FM**2 * p &
         + (2.0D0 + PARAM%FN) * PARAM%FM**2 * p * (p / shvars%get_p0())**PARAM%FN
  pfpsigma = pfps + pfpp * DELTA / 3.0D0
  end procedure get_pfsig_impl
  !*****************************************************************************
  !> @brief Calculate partial derivative of plastic potential with respect to stress
  !>
  !> @details This function computes the partial derivative of the plastic
  !>          potential function with respect to the stress tensor (∂g/∂σ).
  !>          This derivative defines the direction of plastic flow in
  !>          stress space and is essential for non-associative plasticity
  !>          models where the plastic potential differs from the yield
  !>          function.
  !>
  !> @param[in] shvars   Shared variables containing current stress tensor
  !> @param[in] stvars   State variables (void ratio, etc.)
  !>
  !> @return res         Partial derivative ∂g/∂σ (3x3 tensor)
  !*****************************************************************************
  module procedure Get_pgsig_impl
  real(DP) :: cos3t, B, C, temp, r_ef, p, Rm, temp2, Ev, temp3(3, 3), exprf, X_alpha
  real(DP), dimension(3, 3) :: dnorm, R_, Rnorm, ratio
  real(DP), dimension(2) :: dpla
  real(DP), dimension(3, 3) :: Ep, r
  !-----------------------------------------------------------------------------
  p = torch_%pressure(shvars%get_sigma())
  r = torch_%Get_ratio(shvars%get_sigma())
  cos3t = elast_%Get_cos3t(shvars)
  temp = PARAM%C**(1.D0 / PARAM%W)
  dnorm = elast_%Get_dnorm(shvars)
  B = 1.0D0 + 3.0D0 * (PARAM%W * (1.D0 - temp) * cos3t) / ((1.D0 + temp) + (1.D0 - temp) * cos3t)
  C = 9.D0 * dsqrt(2.D0 / 3.D0) * PARAM%W * (1.D0 - temp) / ((1.D0 + temp) + (1.D0 - temp) * cos3t)
  R_ = B * dnorm(:, :) - C * (matmul(dnorm, dnorm) - DELTA / 3.D0)
  Rnorm = torch_%Normalize(R_)
  dpla = plast_%Get_dilatancy(shvars, stvars)
  r_ef = dsqrt(3.0D0 / 2.D0 * sum((r - shvars%get_alpha())**2))
  exprf = exp(-PARAM%V * r_ef)
  ratio = torch_%Get_ratio(shvars%get_sigma())
  Ep = dsqrt(3.D0 / 2.D0) * Rnorm(:, :) * r_ef + 3.D0 / 2.D0 * PARAM%X * ratio(:, :) &
       * exprf
  Rm = torch_%Get_Rm(shvars%get_sigma())
  X_alpha = dpla(2) * PARAM%X * Rm
  Ev = dpla(1) * r_ef + X_alpha * exprf
  !
  pgpsigma = Ep(:, :) + Ev * DELTA / 3.D0
  temp2 = torch_%Trace(pgpsigma)
  temp3 = torch_%Deviatoric(pgpsigma)
  end procedure Get_pgsig_impl
  !*****************************************************************************
  !> @brief Calculate state parameter (ψ) for SANISAND model
  !>
  !> @details This function computes the state parameter ψ, which represents
  !>          the difference between current void ratio and critical state
  !>          void ratio at the same mean effective stress. The state
  !>          parameter is a key variable in critical state soil mechanics
  !>          that characterizes the soils density state relative to the
  !>          critical state line.
  !>
  !> @param[in] shvars   Shared variables (stress tensor, fabric tensor, etc.)
  !> @param[in] stvars   State variables (void ratio, etc.)
  !>
  !> @return res         State parameter ψ = e - e_cs
  !*****************************************************************************
  module procedure Get_psim_impl
  real(DP) :: voidc, p
  !-----------------------------------------------------------------------------
  p = torch_%pressure(shvars%get_sigma())
  voidc = (PARAM%VOIDREF - PARAM%VOIDRL) &
          * exp(-PARAM%LAMDACS * (P / PA)**PARAM%KSI) + PARAM%VOIDRL
  psim = stvars%get_voidr() - voidc
  end procedure Get_psim_impl
  !*****************************************************************************
  module procedure Get_psim_alpha_impl
  real(DP) :: p, lamda_alpha, void_alpha
  !-----------------------------------------------------------------------------
  p = torch_%pressure(shvars%get_sigma())
  lamda_alpha = elast_%Get_lamda(shvars)
  void_alpha = (PARAM%VOIDREF - PARAM%VOIDRL) * exp(-lamda_alpha * (p / PA)**PARAM%KSI) &
               * (1.0D0 + (PARAM%PE / (p + PARAM%CH * PA))**PARAM%Y) + PARAM%VOIDRL
  psim_alpha = stvars%get_voidr() - void_alpha
  end procedure Get_psim_alpha_impl
  !*****************************************************************************
  !> @brief Calculate dilatancy coefficient for SANISAND model
  !>
  !> @details This function computes the dilatancy coefficient D, which
  !>          relates the volumetric plastic strain increment to the
  !>          deviatoric plastic strain increment. The dilatancy coefficient
  !>          determines whether the soil exhibits contractive (D > 0) or
  !>          dilative (D < 0) behavior during plastic deformation.
  !>
  !> @param[in] shvars   Shared variables (stress tensor, fabric tensor, etc.)
  !> @param[in] stvars   State variables (void ratio, etc.)
  !>
  !> @return res         Dilatancy coefficient D
  !*****************************************************************************
  module procedure Get_dilatancy_impl
  real(DP) :: gtheta, C_alpha, psim_alpha, psim, p, X_alpha, Rm
  real(DP), dimension(3, 3) :: dnorm, alpha_d, temp
  !-----------------------------------------------------------------------------
  p = torch_%pressure(shvars%get_sigma())
  gtheta = elast_%Get_gtheta(shvars)
  psim = plast_%Get_psim(shvars, stvars)
  psim_alpha = plast_%Get_psim_alpha(shvars, stvars)
  C_alpha = 2.0D0 / (1.0D0 + exp(-PARAM%BETA * psim_alpha))
  !
  dnorm = elast_%Get_dnorm(shvars)
  alpha_d = dsqrt(2.0D0 / 3.0D0) * PARAM%ALPHAC * gtheta * exp(PARAM%Nd * psim) * dnorm
  temp = alpha_d - shvars%get_alpha()
  dpla(1) = dsqrt(3.0D0 / 2.0D0) * PARAM%Ad / gtheta &
            * exp(PARAM%Pd * C_alpha * dsqrt(p / PA)) * sum(temp * dnorm)
  !
  X_alpha = dsqrt(3.0D0 / 2.0D0) &
            * (dsqrt(sum(alpha_d**2)) - dsqrt(sum(shvars%get_alpha()**2))) &
            * max((1.0D0 - C_alpha), 0.0D0)**2 + C_alpha
  Rm = torch_%Get_Rm(shvars%get_sigma())
  dpla(2) = X_alpha / PARAM%X / Rm
  end procedure Get_dilatancy_impl
  !*****************************************************************************
  !> @brief Calculate fabric evolution for SANISAND model
  !>
  !> @details This subroutine computes the evolution of fabric tensor and
  !>          hardening parameters in the SANISAND model. The fabric tensor
  !>          represents the anisotropic microstructure of the soil, which
  !>          evolves with plastic deformation and affects the materials
  !>          mechanical response.
  !>
  !> @param[in]  shvars   Shared variables (stress tensor, fabric tensor, etc.)
  !> @param[in]  stvars   State variables (void ratio, etc.)
  !> @param[out] Rh       Hardening parameter evolution rate
  !> @param[out] RF       Fabric tensor evolution rate (3x3 tensor)
  !*****************************************************************************
  module procedure Get_evolution_impl
  real(DP) :: lamda_alpha, p, h_e, h_ocr, G_v, psim_alpha, C_alpha, b_0, r_ef, h
  real(DP) :: gtheta, psim, void_alpha, temp2, temp3, temp4
  real(DP), dimension(3, 3) :: temp, dnorm, alpha_b, r, s
  real(DP) :: ftol
  !-----------------------------------------------------------------------------
  ftol = elast_%Yield_distance(shvars)
  p = torch_%pressure(shvars%get_sigma())
  s = torch_%Deviatoric(shvars%get_sigma())
  r = torch_%Get_ratio(shvars%get_sigma())
  gtheta = elast_%Get_gtheta(shvars)
  psim = plast_%Get_psim(shvars, stvars)
  lamda_alpha = elast_%Get_lamda(shvars)
  h_e = log((PARAM%VOIDREF - PARAM%VOIDRL) / (stvars%get_voidr() - PARAM%VOIDRL) &
            * (1.D0 + (PARAM%PE / (p + PARAM%CH * PA))**PARAM%Y)) / lamda_alpha
  h_ocr = 1.D0 + PARAM%PR * (1.D0 - p / shvars%get_p0()) * (p / PA)**0.2
  G_v = elast_%Get_Gv(shvars)
  psim_alpha = plast_%Get_psim_alpha(shvars, stvars)
  C_alpha = 2.D0 / (1.D0 + exp(-PARAM%BETA * psim_alpha))
  b_0 = G_v * PARAM%H0 * exp(-PARAM%PS * C_alpha * dsqrt(p / PA)) * h_ocr * h_e &
        * (p / PA)**(-PARAM%KSI)
  temp = shvars%get_alpha() - stvars%get_alpha_in()
  dnorm = elast_%Get_dnorm(shvars)
  r_ef = dsqrt(3.0 / 2.D0 * sum((r - shvars%get_alpha())**2))
  temp2 = (sum(temp * dnorm) * (1.D0 - exp(-PARAM%V * r_ef)) + exp(-PARAM%V * r_ef))
  temp3 = sum(temp * dnorm) * (1.D0 - exp(-PARAM%V * r_ef))
  temp4 = exp(-PARAM%V * r_ef)
  h = b_0 / (sum(temp * dnorm) * (1.D0 - exp(-PARAM%V * r_ef)) + exp(-PARAM%V * r_ef))
  alpha_b = dsqrt(2.D0 / 3.D0) * PARAM%ALPHAC * gtheta * exp(PARAM%NB * max(-psim, 0.D0)) &
            * dnorm
  Ralpha = h * r_ef * (alpha_b - shvars%get_alpha())
  !
  void_alpha = (PARAM%VOIDREF - PARAM%VOIDRL) * exp(-lamda_alpha * (p / PA)**PARAM%KSI) &
               * (1.0D0 + (PARAM%PE / (p + PARAM%CH * PA))**PARAM%Y) + PARAM%VOIDRL
  RP0 = (1.D0 + void_alpha) * shvars%get_p0() * exp(-PARAM%V * r_ef) / C_alpha &
        / (void_alpha - PARAM%VOIDRL) / lamda_alpha / (1.D0 - PARAM%K) / PARAM%KSI &
        / (shvars%get_p0() / PA)**PARAM%KSI
  end procedure Get_evolution_impl
  !*****************************************************************************
  !>
  !> @details This function computes the plastic modulus Dkp, which relates
  !>          the plastic multiplier to the consistency condition in
  !>          plasticity theory. The plastic modulus determines the hardening
  !>          or softening behavior of the material and is essential for
  !>          computing the plastic strain increments.
  !>
  !> @param[in] shvars   Shared variables (stress tensor, fabric tensor, etc.)
  !> @param[in] stvars   State variables (void ratio, etc.)
  !>
  !> @return Dkp         Plastic modulus
  !*****************************************************************************
  module procedure Get_Dkp_impl
  real(DP), dimension(3, 3) :: pfpalpha, Ralpha, S
  real(DP) :: pfpP0, RP0, p
  !-----------------------------------------------------------------------------
  call plast_%Get_evolution(shvars, stvars, Ralpha, RP0)
  p = torch_%pressure(shvars%get_sigma())
  S = torch_%Deviatoric(shvars%get_sigma())
  pfpalpha = -3.D0 * p * (S - p * shvars%get_alpha())
  pfpP0 = -PARAM%FN / shvars%get_p0() * (PARAM%FM * p)**2 * (p / shvars%get_p0())**PARAM%FN
  Dkp = -(sum(pfpalpha * Ralpha) + pfpP0 * RP0)
  end procedure Get_Dkp_impl
  !*****************************************************************************
  !> @brief Perform elastic predictor - plastic corrector step
  !>
  !> @details This subroutine implements the elastic predictor - plastic
  !>          corrector algorithm for the SANISAND model. It computes the
  !>          updated stress state and consistent tangent modulus given an
  !>          incremental strain. The algorithm consists of two steps:
  !>          1. Elastic predictor: Compute trial stress assuming elastic
  !>             behavior.
  !>          2. Plastic corrector: If the trial stress violates the yield
  !>             condition, apply plastic correction to return to the yield
  !>             surface.
  !>
  !> @param[in]  shvars   Shared variables (stress tensor, fabric tensor, etc.)
  !> @param[in]  stvars   State variables (void ratio, etc.)
  !> @param[in]  depsln   Incremental strain tensor (3x3)
  !> @param[out] Rshvars  Updated shared variables after plastic correction
  !> @param[out] dempx    Consistent tangent modulus (3x3x3x3 fourth-order tensor)
  !*****************************************************************************
  module procedure Elstop_impl
  real(DP), dimension(3, 3) :: pfsig, xm, theta, dsigma, dalpha
  real(DP), dimension(3, 3, 3, 3) :: stiff, cplas
  real(DP) :: dnmetr, Dkp, frnde, lamda, hdl, Ralpha(3, 3), RP0, dp0
  !-----------------------------------------------------------------------------
  pfsig = plast_%Get_pfsig(shvars)
  xm = plast_%Get_pgsig(shvars, stvars)
  Dkp = plast_%Get_Dkp(shvars, stvars)
  stiff = elast_%Get_stiffness(shvars, stvars)
  dnmetr = sum(pfsig * (stiff.ddot.xm))
  frnde = dnmetr + Dkp
  if(abs(frnde) <= EPS) frnde = sign(frnde, EPS)
  !
  theta = (pfsig.ddot.stiff) / frnde
  !
  lamda = sum(theta * depsln)
  ! hdl = lamda > 0.0_DP ? 1.0_DP : 0.0_DP
  hdl = merge(1.0_DP, 0.0_DP, lamda > 0.0_DP)
  cplas = ((stiff.ddot.xm) .dyad.theta)
  dempx = stiff - hdl * cplas
  !
  call plast_%Get_evolution(shvars, stvars, Ralpha, RP0)
  dsigma = dempx.ddot.depsln
  dalpha = hdl * Ralpha * lamda
  dp0 = hdl * RP0 * lamda
  Deshvars = Share_var(dsigma, dalpha, dp0)
  end procedure Elstop_impl
  !*****************************************************************************
  !
!*******************************************************************************
endsubmodule plastic_impl
