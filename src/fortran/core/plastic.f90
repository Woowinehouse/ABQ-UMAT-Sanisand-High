!*******************************************************************************
!> @brief Plasticity module for SANISAND model
!>
!> @details This module implements the plasticity algorithms for the SANISAND
!>          constitutive model, including yield function derivatives, plastic
!>          potential derivatives, dilatancy calculation, fabric evolution,
!>          and plastic modulus computation. It provides the core plastic
!>          correction algorithms used in the UMAT implementation.
!>
!> @author wuwenhao
!> @date 2025/12/05
!*******************************************************************************
module plastic_mod
  use Base_config, only: DP
  use Container_mod
  implicit none
  private
  type, public :: plast
  contains
    procedure, public, nopass :: Get_pfsig => Get_pfsig_impl
    procedure, public, nopass :: Get_pgsig => Get_pgsig_impl
    procedure, public, nopass :: Get_psim => Get_psim_impl
    procedure, public, nopass :: Get_psim_alpha => Get_psim_alpha_impl
    procedure, public, nopass :: Get_dilatancy => Get_dilatancy_impl
    procedure, public, nopass :: Get_evolution => Get_evolution_impl
    procedure, public, nopass :: Get_Dkp => Get_Dkp_impl
    procedure, public, nopass :: Elstop => Elstop_impl
  endtype plast
  !
  interface
    !***************************************************************************
!> @brief Calculate partial derivative of yield function with respect to p
    !>
    !> @details This function computes the partial derivative of the yield
!>          function with respect to mean effective stress (p). This
    !>          derivative is essential for computing the plastic multiplier
    !>          and consistent tangent modulus in plasticity algorithms.
    !>
    !> @param[in] shvars   Shared variables (stress tensor, etc.)
    !>
!> @return res         Partial derivative ∂f/∂p (3x3 tensor)
    !***************************************************************************
    module function Get_pfpr_impl(shvars) result(res)
      type(Share_var), intent(in) :: shvars
      real(DP), dimension(3, 3) :: res
    endfunction Get_pfpr_impl
    !***************************************************************************
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
    !***************************************************************************
    module function Get_pfsig_impl(shvars) result(pfpsigma)
      type(Share_var), intent(in) :: shvars
      real(DP), dimension(3, 3) :: pfpsigma
    endfunction Get_pfsig_impl
    !***************************************************************************
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
    !***************************************************************************
    module function Get_pgsig_impl(shvars, stvars) result(pgpsigma)
      implicit none
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP), dimension(3, 3) :: pgpsigma
    endfunction Get_pgsig_impl
    !***************************************************************************
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
    !***************************************************************************
    module function Get_psim_impl(shvars, stvars) result(psim)
      ! input
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      ! output
      real(DP) :: psim
    endfunction Get_psim_impl
    !***************************************************************************
    module function Get_psim_alpha_impl(shvars, stvars) result(psim_alpha)
      ! input
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      ! output
      real(DP) :: psim_alpha
    endfunction Get_psim_alpha_impl
    !***************************************************************************
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
    !***************************************************************************
    module function Get_dilatancy_impl(shvars, stvars) result(Dpla)
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP), dimension(2) :: Dpla
    endfunction Get_dilatancy_impl
    !***************************************************************************
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
    !***************************************************************************
    module subroutine Get_evolution_impl(shvars, stvars, Ralpha, RP0)
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP), dimension(3, 3) :: Ralpha
      real(DP) :: RP0
    endsubroutine Get_evolution_impl
    !***************************************************************************
    !> @brief Calculate plastic modulus for SANISAND model
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
    !***************************************************************************
    module function Get_Dkp_impl(shvars, stvars) result(Dkp)
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP) :: Dkp
    endfunction Get_Dkp_impl
    !***************************************************************************
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
    !***************************************************************************
    module subroutine Elstop_impl(shvars, stvars, depsln, Deshvars, dempx)
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP), dimension(3, 3), intent(in) :: depsln
      type(Share_var), intent(out) :: Deshvars
      real(DP), dimension(3, 3, 3, 3), intent(out) :: dempx
    endsubroutine Elstop_impl
    !
  endinterface ! end interface
contains
!*******************************************************************************
endmodule plastic_mod
