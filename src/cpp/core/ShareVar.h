#ifndef CORE_SHAREVAR_H
#define CORE_SHAREVAR_H

#include "ShareVarOptions.h"
#include "core/StateVar.h"
#include "core/impl/ShareVarImpl.h"
#include "core/impl/TensorOptions_ops.h"
#include "utils/TypeMap.h"
#include "utils/export.h"

#ifdef __GTEST__BUILD_
#include <gtest/gtest.h>
#endif
namespace umat::core {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif
/**
 * @brief Shared variables container for UMAT (User Material) constitutive model.
 *
 * This class encapsulates the stress state variables (stress, back-stress alpha,
 * p0 tensor) used in constitutive modeling. It provides methods for elastic
 * and plastic computations, state management, and tensor operations.
 *
 * The class uses intrusive pointer semantics for efficient memory management
 * and copy-on-write behavior through ShareVarImpl.
 **/
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class MYUMAT_API ShareVar final {
  // friendly
  friend class ShareVarTest;
  friend StateVar;
#ifdef __GTEST__BUILD_
  FRIEND_TEST(ShareVarTest, DefaultConstructor);
  FRIEND_TEST(ShareVarTest, Assignment);
#endif
  //
  using ShareVarImpl = impl::ShareVarImpl;
  using Tensor = torch::Tensor;
  using data_t = utils::data_t;
  using State = utils::StressState;
  using StressTensor = core::StressTensor;

  protected:
  // member
  c10::intrusive_ptr<ShareVarImpl> impl_;

  public:
  ShareVar() = default;
  explicit ShareVar(c10::intrusive_ptr<ShareVarImpl> ptr) : impl_(std::move(ptr)) {}
  ShareVar(StressTensor stress, StressTensor alpha, StressTensor p0);
  ShareVar(Tensor stress, Tensor alpha, Tensor p0, utils::StressState state);

  ///@name constructors
  ///@{
  static auto create(StressTensor stress, StressTensor alpha, StressTensor p0) -> ShareVar;
  static auto create(Tensor stress, Tensor alpha, Tensor p0, State state) -> ShareVar;

  ShareVar(const ShareVar &other) = default;
  ShareVar(ShareVar &&other) noexcept : ShareVar() { swap(*this, other); }
  /// @}
  ///@name destructor
  ~ShareVar() = default;

  ///@name operator
  /// @{
  // @brief assignment
  auto operator=(ShareVar other) -> ShareVar & {
    swap(*this, other);
    return *this;
  }
  // @brief unary operator
  // binary operator
  auto operator-() const -> ShareVar { return minus(); }
  auto operator+(const core::ShareVar &rhs) const -> ShareVar { return add(rhs); }
  auto operator-(const core::ShareVar &rhs) const -> ShareVar { return sub(rhs); }
  auto operator*(const core::ShareVar &rhs) const -> ShareVar { return mul(rhs); }
  auto operator/(const core::ShareVar &rhs) const -> ShareVar { return div(rhs); }
  auto operator/(utils::data_t scalar) const -> ShareVar { return div_scalar(scalar); }
  auto operator*(data_t scalar) const -> ShareVar {
    auto impl = c10::make_intrusive<impl::ShareVarImpl>(impl_->mul_scalar(scalar));
    return ShareVar(std::move(impl));
  }
  //
  auto operator+=(const core::ShareVar &rhs) -> ShareVar & { return add_(rhs); }
  auto operator-=(const core::ShareVar &rhs) -> ShareVar & { return sub_(rhs); }
  auto operator*=(const core::ShareVar &rhs) -> ShareVar & { return mul_(rhs); }
  auto operator/=(const core::ShareVar &rhs) -> ShareVar & { return div_(rhs); }
  /// @}
  [[nodiscard]] auto minus() const -> ShareVar;
  [[nodiscard]] auto add(const ShareVar &rhs, data_t scalar = 1.0) const -> ShareVar;
  [[nodiscard]] auto sub(const ShareVar &rhs, data_t scalar = 1.0) const -> ShareVar;
  [[nodiscard]] auto mul(const ShareVar &rhs, data_t scalar = 1.0) const -> ShareVar;
  [[nodiscard]] auto div(const ShareVar &rhs) const -> ShareVar;
  [[nodiscard]] auto div_scalar(utils::data_t scalar) const -> ShareVar;
  auto add_(const ShareVar &other, data_t scalar = 1.0) -> ShareVar &;
  auto sub_(const ShareVar &other, data_t scalar = 1.0) -> ShareVar &;
  auto mul_(const ShareVar &other, data_t scalar = 1.0) -> ShareVar &;
  auto div_(const ShareVar &other) -> ShareVar &;
  /// @name scope guard
  /// @{
  [[nodiscard]] auto backup_state() const -> ShareVar;
  auto restore_state(const ShareVar &backup) -> void;
  /// @}

  ///@name Getters
  ///@{
  [[nodiscard]] auto stress_data_ptr() const -> const void * { return impl_->stress_data_ptr(); }
  [[nodiscard]] auto alpha_data_ptr() const -> const void * { return impl_->alpha_data_ptr(); }
  [[nodiscard]] auto p0_data_ptr() const -> const void * { return impl_->p0_data_ptr(); }
  // get const data_ptr
  [[nodiscard]] auto mutable_stress_data_ptr() const -> void * {
    return impl_->mutable_stress_data_ptr();
  }
  [[nodiscard]] auto mutable_alpha_data_ptr() const -> void * {
    return impl_->mutable_alpha_data_ptr();
  }
  [[nodiscard]] auto mutable_p0_data_ptr() const -> void * { return impl_->mutable_p0_data_ptr(); }
  [[nodiscard]] auto data_ptr() const -> impl::ShareVarImpl * { return impl_.get(); }
  [[nodiscard]] auto GetShareVarImpl() const -> const impl::ShareVarImpl & { return *impl_.get(); }
  [[nodiscard]] auto unsafeGetShareVarImpl() -> impl::ShareVarImpl & { return *impl_.get(); }
  // get StressTensor
  auto get_stress() const noexcept -> const StressTensor & { return impl_->get_stress(); };
  auto get_alpha() const noexcept -> const StressTensor & { return impl_->get_alpha(); };
  auto get_p0() const noexcept -> const StressTensor & { return impl_->get_p0(); };
  auto unsafe_get_stress() -> StressTensor & { return impl_->unsafe_get_stress(); };
  auto unsafe_get_alpha() -> StressTensor & { return impl_->unsafe_get_alpha(); };
  auto unsafe_get_p0() -> StressTensor & { return impl_->unsafe_get_p0(); };
  // get tensor
  auto get_stress_tensor() const noexcept -> const Tensor & { return impl_->get_stress_tensor(); };
  auto get_alpha_tensor() const noexcept -> const Tensor & { return impl_->get_alpha_tensor(); };
  auto get_p0_tensor() const noexcept -> const Tensor & { return impl_->get_p0_tensor(); };
  auto unsafe_get_stress_tensor() noexcept -> Tensor & {
    return impl_->unsafe_get_stress_tensor();
  };
  auto unsafe_get_alpha_tensor() noexcept -> Tensor & { return impl_->unsafe_get_alpha_tensor(); };
  auto unsafe_get_p0_tensor() noexcept -> Tensor & { return impl_->unsafe_get_p0_tensor(); };

  // @brief get the member ptr
  auto stress_ptr() const noexcept -> const StressTensor * { return impl_->stress_ptr(); }
  auto alpha_ptr() const noexcept -> const StressTensor * { return impl_->alpha_ptr(); }
  auto p0_ptr() const noexcept -> const StressTensor * { return impl_->p0_ptr(); }
  auto unsafe_stress_ptr() -> StressTensor * { return impl_->unsafe_stress_ptr(); }
  auto unsafe_alpha_ptr() -> StressTensor * { return impl_->unsafe_alpha_ptr(); }
  auto unsafe_p0_ptr() -> StressTensor * { return impl_->unsafe_p0_ptr(); }

  [[nodiscard]] auto stress_tensor_ptr() const noexcept -> const Tensor * {
    return impl_->stress_tensor_ptr();
  }
  [[nodiscard]] auto alpha_tensor_ptr() const noexcept -> const Tensor * {
    return impl_->alpha_tensor_ptr();
  }
  [[nodiscard]] auto p0_tensor_ptr() const noexcept -> const Tensor * {
    return impl_->p0_tensor_ptr();
  }
  auto unsafe_stress_tensor_ptr() -> Tensor * { return impl_->unsafe_stress_tensor_ptr(); }
  auto unsafe_alpha_tensor_ptr() -> Tensor * { return impl_->unsafe_alpha_tensor_ptr(); }
  auto unsafe_p0_tensor_ptr() -> Tensor * { return impl_->unsafe_p0_tensor_ptr(); }
  // use count
  [[nodiscard]] auto get_stress_count() const -> size_t { return impl_->get_stress_count(); }
  [[nodiscard]] auto get_alpha_count() const -> size_t { return impl_->get_alpha_count(); }
  [[nodiscard]] auto get_p0_count() const -> size_t { return impl_->get_p0_count(); }
  [[nodiscard]] auto get_state() const -> State { return impl_->get_state(); }
  [[nodiscard]] auto is_lowstress() const -> bool { return impl_->is_low_stress(); }
  ///@}

  /// @name Setters
  /// @{
  auto set_stress(StressTensor new_tensor) -> void { impl_->set_stress(new_tensor); }
  auto set_alpha(StressTensor new_alpha) -> void { impl_->set_alpha(new_alpha); }
  auto set_p0(StressTensor new_p0) -> void { impl_->set_p0(new_p0); }
  auto update_stress(Tensor stress_tensor, State state) -> void {
    return update_stress(StressTensor(stress_tensor, state));
  }
  auto update_stress(const StressTensor &dstress) -> void { impl_->update_stress(dstress); }
  auto update_alpha(const StressTensor &dalpha) -> void { impl_->update_alpha(dalpha); }
  auto update_p0(const StressTensor &dp0) -> void { impl_->update_p0(dp0); }
  auto update_shareVar(const ShareVar &dshvar) -> void;
  auto update_shareVar(const StressTensor &dstress, const StressTensor &dalpha,
                       const StressTensor &dp0) -> void;
  auto update_shareVar(Tensor dstress, Tensor dalpha, Tensor dp0, utils::StressState state) -> void;
  [[nodiscard]] inline auto create_shvar_with_dstress(StressTensor dstress) const
      -> core::ShareVar {
    return ShareVar(
        c10::make_intrusive<impl::ShareVarImpl>(impl_->create_shvar_with_dstress(dstress)));
  }
  [[nodiscard]] auto create_shvar_from_new_stress(const StressTensor &new_stress) const
      -> core::ShareVar {
    return ShareVar(
        c10::make_intrusive<impl::ShareVarImpl>(impl_->create_shvar_from_new_stress(new_stress)));
  }
  /// @}

  ///@name other methods
  /// @{
  [[nodiscard]] auto clone() const -> ShareVar {
    return ShareVar(c10::make_intrusive<impl::ShareVarImpl>(impl_->clone()));
  }
  [[nodiscard]] auto lazy_clone() const -> ShareVar {
    return ShareVar(c10::make_intrusive<impl::ShareVarImpl>(impl_->lazy_clone()));
  }
  [[nodiscard]] auto use_count() const -> size_t { return impl_.use_count(); }
  [[nodiscard]] auto unique() const -> bool { return impl_.unique(); }

  auto validate() const -> void { impl_->validate(); }
  /// @}

  /// @name elastic methods
  ///@{
  template <bool is_true>
  using Scalar_type = std::conditional_t<is_true, torch::Tensor, utils::data_t>;
  /**
   * @brief Calculate mean pressure from stress tensor
   *
   * Computes the mean pressure (hydrostatic stress) from the stress tensor
   * stored in this ShareVar object. For a stress tensor σ, mean pressure
   * p = -1/3 * trace(σ). The negative sign convention follows soil mechanics
   * (compression positive).
   *
   * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
   * @param err Error code pointer for error reporting
   *
   * @return Scalar_type<retain_map> Mean pressure value
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto mean_pressure(ErrorCode *err = nullptr) -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Pressure_::call<true>(get_stress_tensor(), err);
      return res;
    } else {
      auto res = impl::Pressure_::call<false>(get_stress_tensor(), err);
      return res;
    }
  }
  /**
   * @brief Calculate deviatoric part of stress tensor
   *
   * Extracts the deviatoric (shear) component from the stress tensor
   * stored in this ShareVar object. Deviatoric stress s = σ - pI,
   * where p is mean pressure and I is identity tensor.
   *
   * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
   * @param err Error code pointer for error reporting
   *
   * @return torch::Tensor Deviatoric stress tensor
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto deviatoric(ErrorCode *err = nullptr) const -> torch::Tensor {
    if constexpr (retain_map) {
      auto res = impl::Deviatoric_::call<true>(get_stress_tensor(), err);
      return res;
    } else {
      auto res = impl::Deviatoric_::call<false>(get_stress_tensor(), err);
      return res;
    }
  }
  /**
   * @brief Calculate stress ratio tensor
   *
   * Computes the stress ratio tensor η = s/p, where s is deviatoric stress
   * and p is mean pressure. Used in plasticity models to determine yield condition.
   *
   * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
   * @param err Error code pointer for error reporting
   * @param epsilon Small value to avoid division by zero
   *
   * @return torch::Tensor Stress ratio tensor
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto StressRatio(ErrorCode *err = nullptr, data_t epsilon = 1e-12) const
      -> torch::Tensor {
    if constexpr (retain_map) {
      auto res = impl::StressRatio_::call<true>(get_stress_tensor(), err, epsilon);
      return res;
    } else {
      auto res = impl::StressRatio_::call<false>(get_stress_tensor(), err, epsilon);
      return res;
    }
  }
  /**
   * @brief Calculate Rm parameter (mean stress ratio)
   *
   * Computes Rm = q/p, where q is deviatoric stress invariant and p is mean pressure.
   * This is a key parameter in soil plasticity models for determining yield surface.
   *
   * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
   * @param err Error code pointer for error reporting
   * @param epsilon Small value to avoid division by zero
   * @param ratio Pre-computed stress ratio tensor (optional)
   *
   * @return Scalar_type<retain_map> Rm value (deviatoric stress invariant / mean pressure)
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto calc_Rm(ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                             torch::Tensor ratio = {}) -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Calc_Rm_::call<true>(get_stress_tensor(), err, epsilon, ratio);
      return res;
    } else {
      auto res = impl::Calc_Rm_::call<false>(get_stress_tensor(), err, epsilon, ratio);
      return res;
    }
  }
  /**
   * @brief Calculate loading direction tensor
   *
   * Computes the normalized loading direction tensor n = ∂f/∂σ / ||∂f/∂σ||,
   * where f is the yield function. This defines the direction of plastic flow.
   *
   * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
   * @param err Error code pointer for error reporting
   * @param epsilon Small value for numerical stability
   * @param ratio Pre-computed stress ratio tensor (optional)
   *
   * @return torch::Tensor Loading direction tensor (unit tensor)
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto loadingDirection(ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                      torch::Tensor ratio = {}) const -> torch::Tensor {
    if constexpr (retain_map) {
      auto res = impl::LoadingDirection_::call<true>(GetShareVarImpl(), err, epsilon, ratio);
      return res;
    } else {
      auto res = impl::LoadingDirection_::call<false>(GetShareVarImpl(), err, epsilon, ratio);
      return res;
    }
  }
  /**
   * @brief Calculate Lode angle cosine (cos3θ)
   *
   * Computes cos(3θ) where θ is the Lode angle, which characterizes the stress state
   * in the deviatoric plane. Important for 3D plasticity models.
   * cos3θ = (3√3/2) * J3 / (J2^(3/2)), where J2 and J3 are stress invariants.
   *
   * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
   * @param err Error code pointer for error reporting
   * @param epsilon Small value for numerical stability
   * @param norm Pre-computed norm of deviatoric stress (optional)
   *
   * @return Scalar_type<retain_map> Cosine of 3 times Lode angle
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto cos3theta(ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                               torch::Tensor norm = {}) const -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Cos3theta_::call<true>(GetShareVarImpl(), err, epsilon, norm);
      return res;
    } else {
      auto res = impl::Cos3theta_::call<false>(GetShareVarImpl(), err, epsilon, norm);
      return res;
    }
  }
  /**
   * @brief Calculate g(θ) function for yield surface
   *
   * Computes the g(θ) function that describes the shape of yield surface
   * in the deviatoric plane. Commonly used in Mohr-Coulomb and Drucker-Prager models.
   * When is_grad=true, also returns gradient ∂g/∂(cos3θ).
   *
   * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
   * @tparam is_grad If true, returns pair (g(θ), ∂g/∂(cos3θ)); otherwise returns only g(θ)
   * @param err Error code pointer for error reporting
   * @param epsilon Small value for numerical stability
   * @param cos3t Pre-computed cos3θ value (optional)
   *
   * @return If is_grad=false: g(θ) value
   *         If is_grad=true: pair of (g(θ), ∂g/∂(cos3θ))
   **/
  template <bool retain_map = false, bool is_grad = false>
  [[nodiscard]] auto calc_gtheta(ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                 Scalar_Type cos3t = {}) const
      -> std::conditional_t<is_grad, std::pair<Scalar_type<retain_map>, torch::Tensor>,
                            Scalar_type<retain_map>> {
    if constexpr (retain_map) {
      if constexpr (is_grad) {
        auto res = impl::Calc_gtheta_::call<true, true>(GetShareVarImpl(), err, epsilon, cos3t);
        return res;
      } else {
        auto res = impl::Calc_gtheta_::call<true, false>(GetShareVarImpl(), err, epsilon, cos3t);
        return res;
      }
    } else {
      if constexpr (is_grad) {
        auto res = impl::Calc_gtheta_::call<false, true>(GetShareVarImpl(), err, epsilon, cos3t);
        return res;
      } else {
        auto res = impl::Calc_gtheta_::call<false, false>(GetShareVarImpl(), err, epsilon, cos3t);
        return res;
      }
    }
  }
  /**
   * @brief Calculate plastic multiplier λ
   *
   * Computes the plastic multiplier (consistency parameter) λ for plasticity models.
   * Determines the magnitude of plastic deformation based on yield condition.
   *
   * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
   * @param err Error code pointer for error reporting
   * @param epsilon Small value for numerical stability
   * @param gtheta Pre-computed g(θ) value (optional)
   *
   * @return Scalar_type<retain_map> Plastic multiplier λ
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto calc_lamda(ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                Scalar_Type gtheta = {}) const -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Calc_lamda_::call<true>(GetShareVarImpl(), err, epsilon, gtheta);
      return res;
    } else {
      auto res = impl::Calc_lamda_::call<false>(GetShareVarImpl(), err, epsilon, gtheta);
      return res;
    }
  }
  /**
   * @brief Calculate shear modulus G and bulk modulus K
   *
   * Computes elastic shear modulus G and bulk modulus K from the stress tensor
   * stored in this ShareVar object. Used in elastic constitutive relations.
   *
   * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
   * @param err Error code pointer for error reporting
   * @param pressure Pre-computed mean pressure (optional)
   *
   * @return Scalar_type<retain_map> Pair of (shear modulus G, bulk modulus K)
   **/
  template <bool retain_map = false>
  auto calc_GV(ErrorCode *err = nullptr, Scalar_Type pressure = {}) const
      -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Calc_GV_::call<true>(get_stress_tensor(), err, pressure);
      return res;
    } else {
      auto res = impl::Calc_GV_::call<false>(get_stress_tensor(), err, pressure);
      return res;
    }
  }
  /**
   * @brief Calculate shear and bulk moduli from shared variables
   *
   * Computes elastic shear modulus G and bulk modulus K based on current
   * stress state and void ratio. These moduli are pressure-dependent in
   * hypoplastic and elastoplastic soil models.
   *
   * @tparam retain_map If true, returns tensors for gradient tracking; otherwise returns scalars
   * @param voidr Void ratio (e) of the soil
   * @param err Error code pointer for error reporting
   * @param epsilon Small value for numerical stability
   *
   * @return If retain_map=true: pair of tensors (G, K)
   *         If retain_map=false: pair of scalars (G, K)
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto calc_shear_bulk(data_t voidr, ErrorCode *err = nullptr,
                                     data_t epsilon = 1e-12) const
      -> std::conditional_t<retain_map, std::pair<Tensor, Tensor>, std::pair<data_t, data_t>> {
    if constexpr (retain_map) {
      auto res = impl::Calc_shear_bulk_::call<true>(GetShareVarImpl(), voidr, err, epsilon);
      return res;
    } else {
      auto res = impl::Calc_shear_bulk_::call<false>(GetShareVarImpl(), voidr, err, epsilon);
      return res;
    }
  }
  /**
   * @brief Calculate elastic stiffness tensor
   *
   * Computes the 4th-order elastic stiffness tensor Dᵉ based on current
   * stress state and void ratio. For isotropic elasticity:
   * Dᵉᵢⱼₖₗ = 2Gδᵢₖδⱼₗ + (K - 2G/3)δᵢⱼδₖₗ
   *
   * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
   * @param voidr Void ratio (e) of the soil
   * @param err Error code pointer for error reporting
   * @param epsilon Small value for numerical stability
   *
   * @return torch::Tensor 4th-order elastic stiffness tensor (6x6 in Voigt notation)
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto stiffness(utils::data_t voidr, ErrorCode *err = nullptr,
                               utils::data_t epsilon = 1e-12) const -> torch::Tensor {
    if constexpr (retain_map) {
      auto res = impl::Stiffness_::call<true>(GetShareVarImpl(), voidr, err, epsilon);
      return res;
    } else {
      auto res = impl::Stiffness_::call<false>(GetShareVarImpl(), voidr, err, epsilon);
      return res;
    }
  }
  /**
   * @brief Calculate yield function value
   *
   * Computes the value of yield function f(σ, α) which determines whether
   * the material is in elastic (f < 0) or plastic (f = 0) state.
   * For associated plasticity, f = g where g is plastic potential.
   *
   * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
   * @param err Error code pointer for error reporting
   *
   * @return Scalar_type<retain_map> Yield function value f(σ, α)
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto calc_yield(ErrorCode *err = nullptr) const -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Calc_yield_::call<true>(GetShareVarImpl(), err);
      return res;
    } else {
      auto res = impl::Calc_yield_::call<false>(GetShareVarImpl(), err);
      return res;
    }
  }
  /// @}

  /// @name plastic methods
  /// @{
  /**
   * @brief Calculate plastic flow direction ∂f/∂σ
   *
   * Computes the gradient of yield function with respect to stress tensor.
   * This defines the direction of plastic flow for associated plasticity.
   *
   * @param err Error code pointer for error reporting
   *
   * @return torch::Tensor Plastic flow direction tensor ∂f/∂σ
   **/
  [[nodiscard]] auto pfpsigma(ErrorCode *err = nullptr) const -> torch::Tensor {
    return impl::Pfpsigma_::call(GetShareVarImpl(), err);
  }
  /**
   * @brief Calculate plastic potential gradient ∂g/∂σ
   *
   * Computes the gradient of plastic potential function with respect to stress tensor.
   * For non-associated plasticity, g ≠ f and ∂g/∂σ defines the plastic flow direction.
   *
   * @param voidr Void ratio (e) of the soil
   * @param options Plasticity model options
   * @param err Error code pointer for error reporting
   * @param epsilon Small value for numerical stability
   *
   * @return torch::Tensor Plastic potential gradient tensor ∂g/∂σ
   **/
  [[nodiscard]] auto pgpsigma(utils::data_t voidr, const core::PlasticOptions &options = {},
                              ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12) const
      -> torch::Tensor {
    return impl::Pgpsigma_::call(GetShareVarImpl(), voidr, options, err, epsilon);
  }
  /**
   * @brief Calculate ψ_m parameter (dilatancy coefficient)
   *
   * Computes the dilatancy coefficient ψ_m which relates plastic volumetric
   * strain rate to plastic shear strain rate: dε_v^p = ψ_m dγ^p.
   * Important for modeling soil dilatancy/contractancy.
   *
   * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
   * @param voidr Void ratio (e) of the soil
   * @param err Error code pointer for error reporting
   * @param pressure Pre-computed mean pressure (optional)
   *
   * @return Scalar_type<retain_map> Dilatancy coefficient ψ_m
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto calc_psim(utils::data_t voidr, ErrorCode *err, Scalar_Type pressure = {}) const
      -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Calc_psim_::call<true>(get_stress_tensor(), voidr, err, pressure);
      return res;
    } else {
      auto res = impl::Calc_psim_::call<false>(get_stress_tensor(), voidr, err, pressure);
      return res;
    }
  }
  /**
   * @brief Calculate ψ_m_α parameter (dilatancy coefficient with back-stress)
   *
   * Computes the dilatancy coefficient considering back-stress α (kinematic hardening).
   * ψ_m_α = ψ_m(σ - α) where α is back-stress tensor.
   *
   * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
   * @param voidr Void ratio (e) of the soil
   * @param err Error code pointer for error reporting
   * @param epsilon Small value for numerical stability
   * @param pressure Pre-computed mean pressure (optional)
   * @param lamda_alpha Pre-computed plastic multiplier for back-stress evolution (optional)
   *
   * @return Scalar_type<retain_map> Dilatancy coefficient with back-stress ψ_m_α
   **/
  template <bool retain_map = false>
  auto calc_psim_alpha(utils::data_t voidr, ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                       Scalar_Type pressure = {}, Scalar_Type lamda_alpha = {}) const
      -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Calc_psim_alpha_::call<true>(GetShareVarImpl(), voidr, err, epsilon,
                                                    pressure, lamda_alpha);
      return res;
    } else {
      auto res = impl::Calc_psim_alpha_::call<false>(GetShareVarImpl(), voidr, err, epsilon,
                                                     pressure, lamda_alpha);
      return res;
    }
  }

  /**
   * @brief Calculate dilatancy parameters
   *
   * Computes dilatancy-related parameters including plastic multiplier increment
   * and direction of plastic flow. Used in plasticity integration algorithms.
   *
   * @tparam retain_map If true, returns tensors for gradient tracking; otherwise returns scalars
   * @param voidr Void ratio (e) of the soil
   * @param options Plasticity model options
   * @param err Error code pointer for error reporting
   * @param epsilon Small value for numerical stability
   *
   * @return If retain_map=true: pair of tensors (dilatancy parameters)
   *         If retain_map=false: pair of scalars (dilatancy parameters)
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto dilatancy(utils::data_t voidr, const core::PlasticOptions &options = {},
                               ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12) const
      -> std::conditional_t<retain_map, std::pair<torch::Tensor, torch::Tensor>,
                            std::pair<utils::data_t, utils::data_t>> {
    auto &shvarImpl = GetShareVarImpl();
    if constexpr (retain_map) {
      auto res = impl::Dilatancy_::call<true>(shvarImpl, voidr, options, err, epsilon);
      return res;
    } else {
      auto res = impl::Dilatancy_::call<false>(shvarImpl, voidr, options, err, epsilon);
      return res;
    }
  }
  /**
   * @brief Calculate plastic modulus evolution
   *
   * Computes the evolution of plastic modulus K_p and related parameters
   * during plastic loading. Important for hardening/softening behavior.
   *
   * @tparam retain_map If true, returns tensors for gradient tracking; otherwise returns scalars
   * @param stvars State variables container (internal variables)
   * @param options Plasticity model options
   * @param err Error code pointer for error reporting
   * @param epsilon Small value for numerical stability
   *
   * @return Tuple containing:
   *         - Plastic modulus increment ΔK_p
   *         - Direction tensor for modulus evolution
   *         - Scalar parameter for hardening/softening
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto evolution_Kp(const StateVar &stvars, const PlasticOptions &options = {},
                                  ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12) const
      -> std::tuple<torch::Tensor, torch::Tensor, Scalar_type<retain_map>> {
    auto &shvarImpl = GetShareVarImpl();
    auto &stvarImpl = stvars.GetShareVarImpl();
    if constexpr (retain_map) {
      auto res = impl::Evolution_Dkp_::call<true>(shvarImpl, stvarImpl, options, err, epsilon);
      return res;
    } else {
      auto res = impl::Evolution_Dkp_::call<false>(shvarImpl, stvarImpl, options, err, epsilon);
      return res;
    }
  }
  /// @}
  ///@name check methods
  /// @{
  [[nodiscard]] auto check_invariant() const -> bool { return impl_->check_invariant(); }
  [[nodiscard]] auto isnan() const -> bool { return impl_->is_nan(); }
  [[nodiscard]] auto isinf() const -> bool { return impl_->is_inf(); }
  auto detach_() -> void { impl_->detach_(); }
  /// @}
  /// @name friend function
  friend auto swap(ShareVar &lhs, ShareVar &rhs) noexcept -> void { lhs.impl_.swap(rhs.impl_); }
  friend auto operator<<(std::ostream &os, const ShareVar &self) -> std::ostream & {
    return os << *(self.impl_);
  }
  friend auto operator*(data_t scalar, const ShareVar &self) -> ShareVar { return self * scalar; }
  friend auto allclose(const ShareVar &lhs, const ShareVar &rhs) -> bool {
    return allclose(lhs.GetShareVarImpl(), rhs.GetShareVarImpl());
  }
}; // class ShareVar

// 完美转发
template <typename... Args>
inline auto make_ShareVar(Args &&...args) -> ShareVar {
  return ShareVar::create(std::forward<Args>(args)...);
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

} // namespace umat::core

#endif // CORE_SHAREVAR_H
