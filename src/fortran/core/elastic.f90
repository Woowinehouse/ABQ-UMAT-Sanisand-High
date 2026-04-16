!*****************************************************************************
!> @brief Elastic constitutive model module
!>
!> @details
!> This module implements elastic constitutive relations for soil mechanics
!> within the UMAT framework. It provides functions for calculating elastic
!> properties, stiffness tensors, stress increments, and yield criteria.
!> The module includes methods for principal stress calculation, shear and
!> bulk moduli, anisotropy evaluation, and elastic stress updates.
!>
!> @author wuwenhao
!> @date 2025/12/02
!*****************************************************************************
module elastic_mod
  use Base_config
  use Container_mod
  implicit none
  private
  !-----------------------------------------------------------------------------
  !> @brief Elastic operations type
  !>
  !> @details
  !> This type encapsulates elastic constitutive operations for soil mechanics.
  !> It provides methods for calculating elastic properties, stiffness tensors,
  !> stress increments, and yield criteria evaluation. All methods are
  !> nopass procedures that operate on input parameters without requiring
  !> type instance data.
  !>
  !> @author wuwenhao
  !> @date 2025/12/02
  !-----------------------------------------------------------------------------
  type, public :: Elast
  contains
    procedure, public, nopass :: Get_principal => Get_principal_impl
    ! Get loading direction
    procedure, public, nopass :: Get_dnorm => Get_dnorm_impl
    procedure, public, nopass :: Get_cos3t => Get_cos3t_impl
    procedure, public, nopass :: Get_gtheta => Get_gtheta_impl
    procedure, public, nopass :: Get_lamda => Get_lamda_impl
    procedure, public, nopass :: Get_shear => Get_shear_impl
    procedure, public, nopass :: Get_Gv => Get_Gv_impl
    procedure, public, nopass :: Get_bulk => Get_bulk_impl
    procedure, public, nopass :: Yield_distance => yield_distance_impl
    procedure, public, nopass :: Get_stiffness => Get_stiffness_impl
    procedure, public, nopass :: calc_dsigma => calc_dsigma_impl
  endtype Elast
  !=============================================================================
  ! Abstract interface definition (implemented in the sub-module)
  !=============================================================================
  interface
    !***************************************************************************
    !> @brief Calculate principal stresses
    !>
    !> @details
    !> Compute the principal stresses (eigenvalues) of a 3x3 symmetric tensor.
    !> The eigenvalues are returned in descending order (σ₁ ≥ σ₂ ≥ σ₃).
    !>
    !> @param[in]  tensor  3x3 symmetric stress or strain tensor
    !>
    !> @return Array of principal values (σ₁, σ₂, σ₃) in descending order
    !***************************************************************************
    module function Get_principal_impl(tensor) result(res)
      real(DP), dimension(3, 3), intent(in) :: tensor
      real(DP), dimension(3) :: res
    endfunction Get_principal_impl
    !***************************************************************************
    !> @brief Calculate normalized fabric tensor
    !>
    !> @details
    !> Compute the normalized fabric tensor that represents the directional
    !> distribution of soil fabric. This tensor is normalized to have unit
    !> magnitude and characterizes the anisotropy direction without magnitude
    !> information.
    !>
    !> @param[in]  shvars  Shared variables containing fabric tensor
    !>
    !> @return 3x3 normalized fabric tensor
    !***************************************************************************
    module function Get_dnorm_impl(shvars) result(dnorm)
      type(Share_var), intent(in) :: shvars
      real(DP), dimension(3, 3) :: dnorm
    endfunction Get_dnorm_impl
    !***************************************************************************
    module function Get_cos3t_impl(shvars) result(cos3t)
      type(Share_var), intent(in) :: shvars
      real(DP) :: cos3t
    endfunction Get_cos3t_impl
    !***************************************************************************
    !> @brief Calculate g(theta) function for Lode angle dependence
    !>
    !> @details
    !> Compute the g(theta) function that describes the dependence of yield
    !> surface on the Lode angle θ. This function modulates the yield
    !> surface shape in the deviatoric plane according to the stress
    !> states Lode angle.
    !>
    !> @param[in]  shvars  Shared variables containing stress state
    !>
    !> @return Array of g(theta) values for principal stress directions
    !***************************************************************************
    module function Get_gtheta_impl(shvars) result(gtheta)
      type(Share_var), intent(in) :: shvars
      real(DP) :: gtheta
    endfunction Get_gtheta_impl
    !***************************************************************************
    !> @brief Calculate fabric ratio tensor
    !>
    !> @details
    !> Compute the fabric ratio tensor Fᵣ that characterizes the anisotropic
    !> fabric structure of the soil. This tensor relates the current fabric
    !> state to the reference isotropic configuration and influences the
    !> elastic and plastic behavior.
    !>
    !> @param[in]  shvars  Shared variables containing fabric information
    !>
    !> @return 3x3 fabric ratio tensor
    !***************************************************************************
    module function Get_lamda_impl(shvars) result(lamda)
      type(Share_var), intent(in) :: shvars
      real(DP) :: lamda
    endfunction Get_lamda_impl
    !***************************************************************************
    !> @brief Calculate shear modulus
    !>
    !> @details
    !> Compute the shear modulus (G) of the soil based on current stress
    !> state and void ratio. The shear modulus represents the materials
    !> resistance to shear deformation and is a key parameter for elastic
    !> stress-strain relationships.
    !>
    !> @param[in]  shvars  Shared variables containing stress state
    !> @param[in]  stvars  State variables containing void ratio
    !>
    !> @return Shear modulus (G)
    !***************************************************************************
    module function Get_shear_impl(shvars, stvars) result(shear)
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP) :: shear
    endfunction Get_shear_impl
    !***************************************************************************
    module function Get_Gv_impl(shvars) result(Gv)
      type(Share_var), intent(in) :: shvars
      real(DP) :: Gv
    endfunction Get_Gv_impl
    !***************************************************************************
    !> @brief Calculate bulk modulus
    !>
    !> @details
    !> Compute the bulk modulus (K) of the soil based on current stress
    !> state and void ratio. The bulk modulus represents the materials
    !> resistance to volumetric compression and is essential for calculating
    !> volumetric stress-strain responses.
    !>
    !> @param[in]  shvars  Shared variables containing stress state
    !> @param[in]  stvars  State variables containing void ratio
    !>
    !> @return Bulk modulus (K)
    !***************************************************************************
    module function Get_bulk_impl(shvars, stvars) result(bulk)
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP) :: bulk
    endfunction Get_bulk_impl
    !***************************************************************************
    !> @brief Calculate distance to yield surface
    !>
    !> @details
    !> Compute the distance (f) from the current stress state to the yield
    !> surface. This scalar value indicates how close the material is to
    !> yielding: f < 0 indicates elastic state, f = 0 indicates yielding,
    !> and f > 0 indicates stress state outside the yield surface.
    !>
    !> @param[in]  shvars  Shared variables containing stress state
    !>
    !> @return Yield function value (f)
    !***************************************************************************
    module function Yield_distance_impl(shvars) result(ftol)
      type(Share_var), intent(in) :: shvars
      real(DP) :: ftol
    endfunction Yield_distance_impl

    !***************************************************************************
    !> @brief Calculate elastic stiffness tensor
    !>
    !> @details
    !> Compute the fourth-order elastic stiffness tensor (Cᵢⱼₖₗ) based on
    !> current stress state and void ratio. This tensor relates stress
    !> increments to strain increments through the constitutive relation
    !> Δσᵢⱼ = Cᵢⱼₖₗ Δεₖₗ. For isotropic elasticity, it reduces to a function
    !> of shear and bulk moduli.
    !>
    !> @param[in]  shvars  Shared variables containing stress state
    !> @param[in]  stvars  State variables containing void ratio
    !>
    !> @return 3x3x3x3 elastic stiffness tensor
    !***************************************************************************
    module function Get_stiffness_impl(shvars, stvars) result(stiffness)
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP), dimension(3, 3, 3, 3) :: stiffness
    endfunction Get_stiffness_impl
    !***************************************************************************
    !> @brief Calculate stress increment from strain increment
    !>
    !> @details
    !> Compute the elastic stress increment (Δσ) corresponding to a given
    !> strain increment (Δε) using the elastic stiffness tensor. This
    !> implements the constitutive relation Δσᵢⱼ = Cᵢⱼₖₗ Δεₖₗ for elastic
    !> loading/unloading.
    !>
    !> @param[in]  shvars  Shared variables containing stress state
    !> @param[in]  stvars  State variables containing void ratio
    !> @param[in]  depsln  Strain increment tensor (3x3)
    !>
    !> @return Stress increment tensor (3x3)
    !***************************************************************************
    module function calc_dsigma_impl(shvars, stvars, depsln) result(dsigma)
      type(Share_var), intent(in) :: shvars
      type(State_var), intent(in) :: stvars
      real(DP), dimension(3, 3), intent(in) :: depsln
      real(DP), dimension(3, 3) :: dsigma
    endfunction calc_dsigma_impl
    ! end interface
  endinterface
contains
!*******************************************************************************
endmodule elastic_mod
