#ifndef CORE_TENSOROPTIONS_OPS_H
#define CORE_TENSOROPTIONS_OPS_H
#include "ShareVarImpl.h"
#include "core/ShareVarOptions.h"
#include "core/auxiliary.h"
#include "torch/torch.h"
#include "utils/TypeMap.h"
#include "utils/base_config.h"
#include "utils/concept.h"
#include "utils/export.h"
#include "utils/material_config.h"
#include <span>

namespace umat::core::impl {

/**
 * @brief Base type definitions for tensor operations
 *
 * Provides common type aliases used throughout the tensor operations framework.
 * Distinguishes between tensor types (for gradient tracking) and scalar types
 * (for numerical computations) using the retain_map template parameter.
 */
struct base_type {
  template <bool is_true>
  using Scalar_type = std::conditional_t<is_true, torch::Tensor, utils::data_t>;
  using Tensor = torch::Tensor;
  using data_t = utils::data_t;
};
/**
 * @brief Safe division operation with protection against division by zero
 *
 * Performs division numerator/denominator with protection against division by zero.
 * If |denominator| < epsilon, returns 0 (or zero tensor) to avoid numerical issues.
 * Supports both scalar and tensor inputs with automatic type deduction.
 */
struct MYUMAT_API Safe_divide_ {

  template <utils::Scalartype T, utils::Scalartype U>
  [[nodiscard]] static auto call(T numerator, U denominator, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12)
      -> std::conditional_t<(std::is_same_v<std::decay_t<T>, utils::data_t> &&
                             std::is_same_v<std::decay_t<U>, utils::data_t>),
                            utils::data_t, torch::Tensor>;
};
/**
 * @brief Check if tensor is isotropic
 *
 * Determines whether a stress tensor is isotropic (hydrostatic).
 * A tensor is isotropic if its deviatoric part is zero (within tolerance).
 */
struct Is_isotropic_ {
  [[nodiscard]] static auto call(const torch::Tensor &self) -> bool;
};
/**
 * @brief Check for NaN or infinite values in tensor
 *
 * Performs element-wise check for NaN (Not a Number) and infinite values
 * in the input tensor. Returns true if any element is NaN or infinite.
 */
struct MYUMAT_API Is_nan_inf_ {
  [[nodiscard]] static auto call(const torch::Tensor &tensor) -> bool;
};
/**
 * @brief Calculate norm of shared variables (stress tensor)
 *
 * Computes the Frobenius norm of the stress tensor from ShareVarImpl object.
 * This is used to measure the magnitude of stress state.
 */
struct MYUMAT_API Calc_shvar_norm_ {
  [[nodiscard]] static auto call(const ShareVarImpl &shvarsImpl, ErrorCode *err = nullptr)
      -> torch::Tensor;
};
/**
 * @brief Calculate cosine of angle between two tensors
 *
 * Computes cos(θ) = (lhs·rhs) / (||lhs||·||rhs||) where · denotes inner product.
 * Used to measure alignment between stress or strain tensors.
 */
struct MYUMAT_API Calc_Cosine_angle_ {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &lhs, const torch::Tensor &rhs,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
      -> std::conditional_t<retain_map, torch::Tensor, utils::data_t>;
};
/**
 * @brief Calculate mean pressure from stress tensor
 *
 * Computes the mean pressure (hydrostatic stress) from a stress tensor.
 * For a stress tensor σ, mean pressure p = -1/3 * trace(σ).
 * The negative sign convention follows soil mechanics (compression positive).
 */
struct MYUMAT_API Pressure_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &self, ErrorCode *err = nullptr)
      -> Scalar_type<retain_map>;
};

/**
 * @brief Calculate deviatoric part of stress tensor
 *
 * Extracts the deviatoric (shear) component from a stress tensor.
 * Deviatoric stress s = σ - pI, where p is mean pressure and I is identity tensor.
 */
struct MYUMAT_API Deviatoric_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &self, ErrorCode *err = nullptr)
      -> torch::Tensor;
};
/**
 * @brief Calculate stress ratio tensor
 *
 * Computes the stress ratio tensor η = s/p, where s is deviatoric stress
 * and p is mean pressure. Used in plasticity models to determine yield condition.
 */
struct MYUMAT_API StressRatio_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &self, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12) -> torch::Tensor;
};
/**
 * @brief Calculate Rm parameter (mean stress ratio)
 *
 * Computes Rm = q/p, where q is deviatoric stress invariant and p is mean pressure.
 * This is a key parameter in soil plasticity models for determining yield surface.
 */
struct MYUMAT_API Calc_Rm_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &self, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12, torch::Tensor ratio = {})
      -> Scalar_type<retain_map>;
};
/**
 * @brief Calculate loading direction tensor
 *
 * Computes the normalized loading direction tensor n = ∂f/∂σ / ||∂f/∂σ||,
 * where f is the yield function. This defines the direction of plastic flow.
 */
struct MYUMAT_API LoadingDirection_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                 torch::Tensor ratio = {}) -> torch::Tensor;
};
/**
 * @brief Calculate Lode angle cosine (cos3θ)
 *
 * Computes cos(3θ) where θ is the Lode angle, which characterizes the stress state
 * in the deviatoric plane. Important for 3D plasticity models.
 * cos3θ = (3√3/2) * J3 / (J2^(3/2)), where J2 and J3 are stress invariants.
 */
struct MYUMAT_API Cos3theta_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                 torch::Tensor norm = {}) -> Scalar_type<retain_map>;
};
/**
 * @brief Calculate g(θ) function for yield surface
 *
 * Computes the g(θ) function that describes the shape of yield surface
 * in the deviatoric plane. Commonly used in Mohr-Coulomb and Drucker-Prager models.
 * When is_grad=true, also returns gradient ∂g/∂(cos3θ).
 */
struct MYUMAT_API Calc_gtheta_ : base_type {
  template <bool retain_map = false, bool is_grad = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                 Scalar_Type cos3t = {})
      -> std::conditional_t<is_grad, std::pair<Scalar_type<retain_map>, torch::Tensor>,
                            Scalar_type<retain_map>>;
};
/**
 * @brief Calculate plastic multiplier λ
 *
 * Computes the plastic multiplier (consistency parameter) λ for plasticity models.
 * Determines the magnitude of plastic deformation based on yield condition.
 */
struct MYUMAT_API Calc_lamda_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                 Scalar_Type gtheta = {}) -> Scalar_type<retain_map>;
};
/**
 * @brief Calculate shear modulus G and bulk modulus K
 *
 * Computes elastic shear modulus G and bulk modulus K from stress tensor
 * and void ratio. Used in elastic constitutive relations.
 */
struct MYUMAT_API Calc_GV_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &self, ErrorCode *err = nullptr,
                                 Scalar_Type pressure = {}) -> Scalar_type<retain_map>;
};

/**
 * @brief Calculate shear and bulk moduli from shared variables
 *
 * Computes elastic shear modulus G and bulk modulus K based on current
 * stress state and void ratio. These moduli are pressure-dependent in
 * hypoplastic and elastoplastic soil models.
 */
struct MYUMAT_API Calc_shear_bulk_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
      -> std::conditional_t<retain_map, std::pair<Tensor, Tensor>, std::pair<data_t, data_t>>;
};

/**
 * @brief Calculate elastic stiffness tensor
 *
 * Computes the 4th-order elastic stiffness tensor Dᵉ based on current
 * stress state and void ratio. For isotropic elasticity:
 * Dᵉᵢⱼₖₗ = 2Gδᵢₖδⱼₗ + (K - 2G/3)δᵢⱼδₖₗ
 */
struct MYUMAT_API Stiffness_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
      -> torch::Tensor;
};

/**
 * @brief Calculate yield function value
 *
 * Computes the value of yield function f(σ, α) which determines whether
 * the material is in elastic (f < 0) or plastic (f = 0) state.
 * For associated plasticity, f = g where g is plastic potential.
 */
struct MYUMAT_API Calc_yield_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 ErrorCode *err = nullptr) -> Scalar_type<retain_map>;
};
/**
 * @brief Calculate the gradient of yield function
 *
 **/
struct MYUMAT_API Pfpsigma_ {
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 ErrorCode *err = nullptr) -> torch::Tensor;
};
/**
 * @brief Calculate plastic potential gradient ∂g/∂σ
 *
 * Computes the gradient of plastic potential function with respect to stress tensor.
 * For non-associated plasticity, g ≠ f and ∂g/∂σ defines the plastic flow direction.
 */

struct MYUMAT_API Pgpsigma_ {
  template <bool return_pair = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                                 const core::PlasticOptions &options = {}, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12)
      -> std::conditional_t<return_pair, torch::Tensor, pair_tensor>;
};
/**
 * @brief Calculate ψ_m parameter (dilatancy coefficient)
 *
 * Computes the dilatancy coefficient ψ_m which relates plastic volumetric
 * strain rate to plastic shear strain rate: dε_v^p = ψ_m dγ^p.
 * Important for modeling soil dilatancy/contractancy.
 */
struct MYUMAT_API Calc_psim_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &self, utils::data_t voidr, ErrorCode *err,
                                 Scalar_Type pressure = {}) -> Scalar_type<retain_map>;
};
/**
 * @brief Calculate ψ_m_α parameter (dilatancy coefficient with back-stress)
 *
 * Computes the dilatancy coefficient considering back-stress α (kinematic hardening).
 * ψ_m_α = ψ_m(σ - α) where α is back-stress tensor.
 */
struct MYUMAT_API Calc_psim_alpha_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                 Scalar_Type pressure = {}, Scalar_Type lamda_alpha = {})
      -> Scalar_type<retain_map>;
};
/**
 * @brief Calculate dilatancy parameters
 *
 * Computes dilatancy-related parameters including plastic multiplier increment
 * and direction of plastic flow. Used in plasticity integration algorithms.
 */
struct MYUMAT_API Dilatancy_ {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                                 const core::PlasticOptions &options = {}, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12)
      -> std::conditional_t<retain_map, std::pair<torch::Tensor, torch::Tensor>,
                            std::pair<utils::data_t, utils::data_t>>;
};

/**
 * @brief Calculate plastic modulus evolution
 *
 * Computes the evolution of plastic modulus K_p and related parameters
 * during plastic loading. Important for hardening/softening behavior.
 */
struct MYUMAT_API Evolution_Dkp_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 const core::impl::StateVarImpl &stvarImpl,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                 const core::PlasticOptions &options = {})
      -> std::tuple<torch::Tensor, torch::Tensor, Scalar_type<retain_map>>;
};

// 实例化函数
// Safe_divide_
extern template MYUMAT_API auto
Safe_divide_::call<torch::Tensor, torch::Tensor>(torch::Tensor, torch::Tensor, ErrorCode *err,
                                                 utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto
Safe_divide_::call<torch::Tensor, utils::data_t>(torch::Tensor, utils::data_t, ErrorCode *err,
                                                 utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto
Safe_divide_::call<utils::data_t, torch::Tensor>(utils::data_t, torch::Tensor, ErrorCode *err,
                                                 utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto
Safe_divide_::call<utils::data_t, utils::data_t>(utils::data_t, utils::data_t, ErrorCode *err,
                                                 utils::data_t epsilon) -> utils::data_t;
// Calc_Cosine_angle_
extern template MYUMAT_API auto
Calc_Cosine_angle_::call<true>(const torch::Tensor &lhs, const torch::Tensor &rhs, ErrorCode *err,
                               utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto
Calc_Cosine_angle_::call<false>(const torch::Tensor &lhs, const torch::Tensor &rhs, ErrorCode *err,
                                utils::data_t epsilon) -> utils::data_t;
// Pressure_
extern template MYUMAT_API auto Pressure_::call<true>(const torch::Tensor &self, ErrorCode *err)
    -> torch::Tensor;
extern template MYUMAT_API auto Pressure_::call<false>(const torch::Tensor &self, ErrorCode *err)
    -> utils::data_t;
extern template MYUMAT_API auto Deviatoric_::call<true>(const torch::Tensor &self, ErrorCode *err)
    -> torch::Tensor;
extern template MYUMAT_API auto Deviatoric_::call<false>(const torch::Tensor &self, ErrorCode *err)
    -> torch::Tensor;
extern template MYUMAT_API auto StressRatio_::call<true>(const torch::Tensor &self, ErrorCode *err,
                                                         utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto StressRatio_::call<false>(const torch::Tensor &self, ErrorCode *err,
                                                          utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto Calc_Rm_::call<true>(const torch::Tensor &self, ErrorCode *err,
                                                     utils::data_t epsilon, torch::Tensor)
    -> torch::Tensor;
extern template MYUMAT_API auto Calc_Rm_::call<false>(const torch::Tensor &self, ErrorCode *err,
                                                      utils::data_t epsilon, torch::Tensor)
    -> utils::data_t;
extern template MYUMAT_API auto
LoadingDirection_::call<true>(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                              utils::data_t epsilon, torch::Tensor ratio) -> torch::Tensor;
extern template MYUMAT_API auto
LoadingDirection_::call<false>(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                               utils::data_t epsilon, torch::Tensor ratio) -> torch::Tensor;
extern template MYUMAT_API auto Cos3theta_::call<true>(const core::impl::ShareVarImpl &shvarImpl,
                                                       ErrorCode *err, utils::data_t epsilon,
                                                       torch::Tensor norm) -> torch::Tensor;
extern template MYUMAT_API auto Cos3theta_::call<false>(const core::impl::ShareVarImpl &shvarImpl,
                                                        ErrorCode *err, utils::data_t epsilon,
                                                        torch::Tensor norm) -> utils::data_t;
extern template MYUMAT_API auto
Calc_gtheta_::call<true, true>(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                               utils::data_t epsilon, Scalar_Type cos3t)
    -> std::pair<torch::Tensor, torch::Tensor>;
extern template MYUMAT_API auto
Calc_gtheta_::call<true, false>(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                                utils::data_t epsilon, Scalar_Type cos3t) -> torch::Tensor;

extern template MYUMAT_API auto
Calc_gtheta_::call<false, true>(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                                utils::data_t epsilon, Scalar_Type cos3t)
    -> std::pair<utils::data_t, torch::Tensor>;

extern template MYUMAT_API auto
Calc_gtheta_::call<false, false>(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                                 utils::data_t epsilon, Scalar_Type cos3t) -> utils::data_t;

extern template MYUMAT_API auto Calc_lamda_::call<true>(const core::impl::ShareVarImpl &shvarImpl,
                                                        ErrorCode *err, utils::data_t epsilon,
                                                        Scalar_Type) -> torch::Tensor;
extern template MYUMAT_API auto Calc_lamda_::call<false>(const core::impl::ShareVarImpl &shvarImpl,
                                                         ErrorCode *err, utils::data_t epsilon,
                                                         Scalar_Type gtheta) -> utils::data_t;
extern template MYUMAT_API auto Calc_GV_::call<true>(const torch::Tensor &self, ErrorCode *err,
                                                     Scalar_Type) -> torch::Tensor;
extern template MYUMAT_API auto Calc_GV_::call<false>(const torch::Tensor &self, ErrorCode *err,
                                                      Scalar_Type pressure) -> utils::data_t;
extern template MYUMAT_API auto
Calc_shear_bulk_::call<true>(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                             ErrorCode *err, utils::data_t epsilon) -> std::pair<Tensor, Tensor>;
extern template MYUMAT_API auto
Calc_shear_bulk_::call<false>(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                              ErrorCode *err, utils::data_t epsilon) -> std::pair<data_t, data_t>;
extern template MYUMAT_API auto Stiffness_::call<true>(const impl::ShareVarImpl &shvarImpl,
                                                       utils::data_t voidr, ErrorCode *err,
                                                       utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto Stiffness_::call<false>(const impl::ShareVarImpl &shvarImpl,
                                                        utils::data_t voidr, ErrorCode *err,
                                                        utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto Calc_yield_::call<true>(const core::impl::ShareVarImpl &shvarImpl,
                                                        ErrorCode *err) -> torch::Tensor;
extern template MYUMAT_API auto Calc_yield_::call<false>(const core::impl::ShareVarImpl &shvarImpl,
                                                         ErrorCode *err) -> utils::data_t;
extern template MYUMAT_API auto Calc_psim_::call<true>(const torch::Tensor &self,
                                                       utils::data_t voidr, ErrorCode *err,
                                                       Scalar_Type pressure) -> torch::Tensor;
extern template MYUMAT_API auto Calc_psim_::call<false>(const torch::Tensor &self,
                                                        utils::data_t voidr, ErrorCode *err,
                                                        Scalar_Type pressure) -> utils::data_t;
extern template MYUMAT_API auto Calc_psim_alpha_::call<true>(const impl::ShareVarImpl &shvarImpl,
                                                             utils::data_t voidr, ErrorCode *err,
                                                             utils::data_t epsilon, Scalar_Type,
                                                             Scalar_Type) -> torch::Tensor;
extern template MYUMAT_API auto Calc_psim_alpha_::call<false>(const impl::ShareVarImpl &shvarImpl,
                                                              utils::data_t voidr, ErrorCode *err,
                                                              utils::data_t epsilon, Scalar_Type,
                                                              Scalar_Type) -> utils::data_t;
extern template MYUMAT_API auto
Dilatancy_::call<true>(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                       const core::PlasticOptions &options, ErrorCode *err, utils::data_t epsilon)
    -> std::pair<torch::Tensor, torch::Tensor>;
extern template MYUMAT_API auto
Dilatancy_::call<false>(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                        const core::PlasticOptions &options, ErrorCode *err, utils::data_t epsilon)
    -> std::pair<utils::data_t, utils::data_t>;
extern template MYUMAT_API auto
Evolution_Dkp_::call<true>(const core::impl::ShareVarImpl &shvarImpl,
                           const core::impl::StateVarImpl &stvarsImpl, ErrorCode *err,
                           utils::data_t epsilon, const core::PlasticOptions &options)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
extern template MYUMAT_API auto
Evolution_Dkp_::call<false>(const core::impl::ShareVarImpl &shvarImpl,
                            const core::impl::StateVarImpl &stvarsImpl, ErrorCode *err,
                            utils::data_t epsilon, const core::PlasticOptions &options)
    -> std::tuple<torch::Tensor, torch::Tensor, utils::data_t>;
} // namespace umat::core::impl

#endif // CORE_TENSOROPTIONS_OPS_H
