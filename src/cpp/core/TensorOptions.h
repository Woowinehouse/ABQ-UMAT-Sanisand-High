#ifndef CORE_TENSOROPTIONS_H
#define CORE_TENSOROPTIONS_H
#include "core/ShareVar.h"
#include "core/StateVar.h"
#include "core/impl/TensorOptions_ops.h"
#include "torch/torch.h"
#include "utils/TypeMap.h"
#include "utils/base_config.h"
#include "utils/export.h"

namespace umat::core {
template <bool is_true>
using Scalar_type = std::conditional_t<is_true, torch::Tensor, utils::data_t>;
/**
 * @brief Calculate the norm of shared variables (stress tensor)
 *
 * Computes the Frobenius norm of the stress tensor from ShareVar object.
 * This is used to measure the magnitude of stress state.
 *
 * @param shvar Shared variables container containing stress state
 * @param err Error code pointer for error reporting
 *
 * @return torch::Tensor Norm value as a scalar tensor
 **/
[[nodiscard]] MYUMAT_API inline auto calc_shvar_norm(const core::ShareVar &shvar, ErrorCode *err)
    -> torch::Tensor {
  return impl::Calc_shvar_norm_::call(shvar.GetShareVarImpl(), err);
}
/**
 * @brief Calculate cosine of angle between two tensors
 *
 * Computes cos(θ) = (lhs·rhs) / (||lhs||·||rhs||) where · denotes inner product.
 * Used to measure alignment between stress or strain tensors.
 *
 * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
 * @param lhs First input tensor
 * @param rhs Second input tensor
 * @param err Error code pointer for error reporting
 * @param epsilon Small value to avoid division by zero
 *
 * @return Cosine of angle between lhs and rhs tensors
 **/
template <bool retain_map = false>
[[nodiscard]] inline auto calc_Cosine_angle(const torch::Tensor &lhs, const torch::Tensor &rhs,
                                            ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
    -> std::conditional_t<retain_map, torch::Tensor, utils::data_t> {
  if constexpr (retain_map) {
    auto res = impl::Calc_Cosine_angle_::call<true>(lhs, rhs, err, epsilon);
    return res;
  } else {
    auto res = impl::Calc_Cosine_angle_::call<false>(lhs, rhs, err, epsilon);
    return res;
  }
}
/**
 * @brief Safe division with protection against division by zero
 *
 * Performs division numerator/denominator with protection against division by zero.
 * If |denominator| < epsilon, returns 0 (or zero tensor) to avoid numerical issues.
 * Supports both scalar and tensor inputs with automatic type deduction.
 *
 * @tparam T Type of numerator (scalar or tensor)
 * @tparam U Type of denominator (scalar or tensor)
 *
 * @param numerator Dividend
 * @param denominator Divisor
 * @param err Error code pointer for error reporting
 * @param epsilon Threshold for treating denominator as zero
 *
 * @return Result of safe division. Returns scalar if both inputs are scalars,
 *         otherwise returns tensor.
 **/
template <utils::Scalartype T, utils::Scalartype U>
[[nodiscard]] inline auto safe_divide(T numerator, U denominator, ErrorCode *err = nullptr,
                                      utils::data_t epsilon = 1e-12)
    -> std::conditional_t<(std::is_same_v<std::decay_t<T>, utils::data_t> &&
                           std::is_same_v<std::decay_t<U>, utils::data_t>),
                          utils::data_t, torch::Tensor> {
  return impl::Safe_divide_::call(numerator, denominator, err, epsilon);
}

/**
 * @brief Check if tensor contains NaN or infinite values
 *
 * Performs element-wise check for NaN (Not a Number) and infinite values
 * in the input tensor. Returns true if any element is NaN or infinite.
 *
 * @param self Input tensor to check
 *
 * @return bool True if tensor contains NaN or infinite values, false otherwise
 **/
[[nodiscard]] MYUMAT_API inline auto is_nan_inf(const torch::Tensor &self) -> bool {
  return impl::Is_nan_inf_::call(self);
}
/**
 * @brief Calculate mean pressure from stress tensor
 *
 * Computes the mean pressure (hydrostatic stress) from a stress tensor.
 * For a stress tensor σ, mean pressure p = -1/3 * trace(σ).
 * The negative sign convention follows soil mechanics (compression positive).
 *
 * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
 * @param self Input stress tensor
 * @param err Error code pointer for error reporting
 *
 * @return Scalar_type<retain_map> Mean pressure value
 **/
template <bool retain_map = false>
[[nodiscard]] inline auto mean_pressure(const torch::Tensor &self, ErrorCode *err = nullptr)
    -> Scalar_type<retain_map> {
  if constexpr (retain_map) {
    auto res = impl::Pressure_::call<true>(self, err);
    return res;
  } else {
    auto res = impl::Pressure_::call<false>(self, err);
    return res;
  }
}
/**
 * @brief Calculate deviatoric part of stress tensor
 *
 * Extracts the deviatoric (shear) component from a stress tensor.
 * Deviatoric stress s = σ - pI, where p is mean pressure and I is identity tensor.
 *
 * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
 * @param self Input stress tensor
 * @param err Error code pointer for error reporting
 *
 * @return torch::Tensor Deviatoric stress tensor
 **/
template <bool retain_map = false>
[[nodiscard]] inline auto deviatoric(const torch::Tensor &self, ErrorCode *err = nullptr)
    -> torch::Tensor {
  if constexpr (retain_map) {
    auto res = impl::Deviatoric_::call<true>(self, err);
    return res;
  } else {
    auto res = impl::Deviatoric_::call<false>(self, err);
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
 * @param self Input stress tensor
 * @param err Error code pointer for error reporting
 * @param epsilon Small value to avoid division by zero
 *
 * @return torch::Tensor Stress ratio tensor
 **/
template <bool retain_map = false>
[[nodiscard]] inline auto stressRatio(const torch::Tensor &self, ErrorCode *err = nullptr,
                                      utils::data_t epsilon = 1e-12) -> torch::Tensor {
  if constexpr (retain_map) {
    auto res = impl::StressRatio_::call<true>(self, err, epsilon);
    return res;
  } else {
    auto res = impl::StressRatio_::call<false>(self, err, epsilon);
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
 * @param self Input stress tensor
 * @param err Error code pointer for error reporting
 * @param epsilon Small value to avoid division by zero
 * @param ratio Pre-computed stress ratio tensor (optional)
 *
 * @return Scalar_type<retain_map> Rm value (deviatoric stress invariant / mean pressure)
 **/
template <bool retain_map = false>
[[nodiscard]] inline auto Calc_Rm(const torch::Tensor &self, ErrorCode *err = nullptr,
                                  utils::data_t epsilon = 1e-12, torch::Tensor ratio = {})
    -> Scalar_type<retain_map> {
  if constexpr (retain_map) {
    auto res = impl::Calc_Rm_::call<true>(self, err, epsilon, ratio);
    return res;
  } else {
    auto res = impl::Calc_Rm_::call<false>(self, err, epsilon, ratio);
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
 * @param shvar Shared variables container
 * @param err Error code pointer for error reporting
 * @param epsilon Small value for numerical stability
 * @param ratio Pre-computed stress ratio tensor (optional)
 *
 * @return torch::Tensor Loading direction tensor (unit tensor)
 **/
template <bool retain_map = false>
[[nodiscard]] inline auto loadingDirection(const core::ShareVar &shvar, ErrorCode *err = nullptr,
                                           utils::data_t epsilon = 1e-12, torch::Tensor ratio = {})
    -> torch::Tensor {
  auto &shvarImpl = shvar.GetShareVarImpl();
  if constexpr (retain_map) {
    auto res = impl::LoadingDirection_::call<true>(shvarImpl, err, epsilon, ratio);
    return res;
  } else {
    auto res = impl::LoadingDirection_::call<false>(shvarImpl, err, epsilon, ratio);
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
 * @param shvar Shared variables container
 * @param err Error code pointer for error reporting
 * @param epsilon Small value for numerical stability
 * @param norm Pre-computed norm of deviatoric stress (optional)
 *
 * @return Scalar_type<retain_map> Cosine of 3 times Lode angle
 **/
template <bool retain_map = false>
[[nodiscard]] inline auto cos3theta(const core::ShareVar &shvar, ErrorCode *err = nullptr,
                                    utils::data_t epsilon = 1e-12, torch::Tensor norm = {})
    -> Scalar_type<retain_map> {
  auto &shvarImpl = shvar.GetShareVarImpl();
  if constexpr (retain_map) {
    auto res = impl::Cos3theta_::call<true>(shvarImpl, err, epsilon, norm);
    return res;
  } else {
    auto res = impl::Cos3theta_::call<false>(shvarImpl, err, epsilon, norm);
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
 * @param shvar Shared variables container
 * @param err Error code pointer for error reporting
 * @param epsilon Small value for numerical stability
 * @param cos3t Pre-computed cos3θ value (optional)
 *
 * @return If is_grad=false: g(θ) value
 *         If is_grad=true: pair of (g(θ), ∂g/∂(cos3θ))
 **/
template <bool retain_map = false, bool is_grad = false>
[[nodiscard]] inline auto calc_gtheta(const core::ShareVar &shvar, ErrorCode *err = nullptr,
                                      utils::data_t epsilon = 1e-12, Scalar_Type cos3t = {})
    -> std::conditional_t<is_grad, std::pair<Scalar_type<retain_map>, torch::Tensor>,
                          Scalar_type<retain_map>> {
  auto &shvarImpl = shvar.GetShareVarImpl();
  if constexpr (retain_map) {
    if constexpr (is_grad) {
      auto res = impl::Calc_gtheta_::call<true, true>(shvarImpl, err, epsilon, cos3t);
      return res;
    } else {
      auto res = impl::Calc_gtheta_::call<true, false>(shvarImpl, err, epsilon, cos3t);
      return res;
    }
  } else {
    if constexpr (is_grad) {
      auto res = impl::Calc_gtheta_::call<false, true>(shvarImpl, err, epsilon, cos3t);
      return res;
    } else {
      auto res = impl::Calc_gtheta_::call<false, false>(shvarImpl, err, epsilon, cos3t);
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
 * @param shvar Shared variables container
 * @param err Error code pointer for error reporting
 * @param epsilon Small value for numerical stability
 * @param gtheta Pre-computed g(θ) value (optional)
 *
 * @return Scalar_type<retain_map> Plastic multiplier λ
 **/
template <bool retain_map = false>
[[nodiscard]] static auto calc_lamda(const core::ShareVar &shvar, ErrorCode *err = nullptr,
                                     utils::data_t epsilon = 1e-12, Scalar_Type gtheta = {})
    -> Scalar_type<retain_map> {
  auto &shvarImpl = shvar.GetShareVarImpl();
  if constexpr (retain_map) {
    auto res = impl::Calc_lamda_::call<true>(shvarImpl, err, epsilon, gtheta);
    return res;
  } else {
    auto res = impl::Calc_lamda_::call<false>(shvarImpl, err, epsilon, gtheta);
    return res;
  }
}
/**
 * @brief Calculate shear modulus G and bulk modulus K
 *
 * Computes elastic shear modulus G and bulk modulus K from stress tensor
 * and void ratio. Used in elastic constitutive relations.
 *
 * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
 * @param self Input stress tensor
 * @param err Error code pointer for error reporting
 * @param pressure Pre-computed mean pressure (optional)
 *
 * @return Scalar_type<retain_map> Pair of (shear modulus G, bulk modulus K)
 **/
template <bool retain_map = false>
[[nodiscard]] static auto calc_GV(const torch::Tensor &self, ErrorCode *err = nullptr,
                                  Scalar_Type pressure = {}) -> Scalar_type<retain_map> {
  if constexpr (retain_map) {
    auto res = impl::Calc_GV_::call<true>(self, err, pressure);
    return res;
  } else {
    auto res = impl::Calc_GV_::call<false>(self, err, pressure);
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
 * @param shvar Shared variables container
 * @param voidr Void ratio (e) of the soil
 * @param err Error code pointer for error reporting
 * @param epsilon Small value for numerical stability
 *
 * @return If retain_map=true: pair of tensors (G, K)
 *         If retain_map=false: pair of scalars (G, K)
 **/
template <bool retain_map = false>
[[nodiscard]] inline auto calc_shear_bulk(const core::ShareVar &shvar, utils::data_t voidr,
                                          ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
    -> std::conditional_t<retain_map, pair_tensor, pair_data> {
  auto &shvarImpl = shvar.GetShareVarImpl();
  if constexpr (retain_map) {
    auto res = impl::Calc_shear_bulk_::call<true>(shvarImpl, voidr, err, epsilon);
    return res;
  } else {
    auto res = impl::Calc_shear_bulk_::call<false>(shvarImpl, voidr, err, epsilon);
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
 * @param shvar Shared variables container
 * @param voidr Void ratio (e) of the soil
 * @param err Error code pointer for error reporting
 * @param epsilon Small value for numerical stability
 *
 * @return torch::Tensor 4th-order elastic stiffness tensor (6x6 in Voigt notation)
 **/
template <bool retain_map = false>
[[nodiscard]] inline auto stiffness(const core::ShareVar &shvar, utils::data_t voidr,
                                    ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
    -> torch::Tensor {
  auto &shvarImpl = shvar.GetShareVarImpl();
  if constexpr (retain_map) {
    auto res = impl::Stiffness_::call<true>(shvarImpl, voidr, err, epsilon);
    return res;
  } else {
    auto res = impl::Stiffness_::call<false>(shvarImpl, voidr, err, epsilon);
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
 * @param shvar Shared variables container
 * @param err Error code pointer for error reporting
 *
 * @return Scalar_type<retain_map> Yield function value f(σ, α)
 **/
template <bool retain_map = false>
[[nodiscard]] inline auto calc_yield(const core::ShareVar &shvar, ErrorCode *err = nullptr)
    -> Scalar_type<retain_map> {
  auto &shvarImpl = shvar.GetShareVarImpl();
  if constexpr (retain_map) {
    auto res = impl::Calc_yield_::call<true>(shvarImpl, err);
    return res;
  } else {
    auto res = impl::Calc_yield_::call<false>(shvarImpl, err);
    return res;
  }
}
/**
 * @brief Calculate plastic flow direction ∂f/∂σ
 *
 * Computes the gradient of yield function with respect to stress tensor.
 * This defines the direction of plastic flow for associated plasticity.
 *
 * @param shvar Shared variables container
 * @param err Error code pointer for error reporting
 *
 * @return torch::Tensor Plastic flow direction tensor ∂f/∂σ
 **/
[[nodiscard]] MYUMAT_API inline auto pfpsigma(const core::ShareVar &shvar, ErrorCode *err = nullptr)
    -> torch::Tensor {
  auto &shvarImpl = shvar.GetShareVarImpl();
  return impl::Pfpsigma_::call(shvarImpl, err);
}
/**
 * @brief Calculate plastic potential gradient ∂g/∂σ
 *
 * Computes the gradient of plastic potential function with respect to stress tensor.
 * For non-associated plasticity, g ≠ f and ∂g/∂σ defines the plastic flow direction.
 *
 * @param shvar Shared variables container
 * @param voidr Void ratio (e) of the soil
 * @param options Plasticity model options
 * @param err Error code pointer for error reporting
 * @param epsilon Small value for numerical stability
 *
 * @return torch::Tensor Plastic potential gradient tensor ∂g/∂σ
 **/
template <bool return_pair = false>
[[nodiscard]] inline auto pgpsigma(const core::ShareVar &shvar, utils::data_t voidr,
                                   const core::PlasticOptions &options = {},
                                   ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
    -> std::conditional_t<return_pair, torch::Tensor, pair_tensor> {
  auto &shvarImpl = shvar.GetShareVarImpl();
  if (return_pair) {
    return impl::Pgpsigma_::call<true>(shvarImpl, voidr, options, err, epsilon);
  } else {
    return impl::Pgpsigma_::call<false>(shvarImpl, voidr, options, err, epsilon);
  }
}
/**
 * @brief Calculate ψ_m parameter (dilatancy coefficient)
 *
 * Computes the dilatancy coefficient ψ_m which relates plastic volumetric
 * strain rate to plastic shear strain rate: dε_v^p = ψ_m dγ^p.
 * Important for modeling soil dilatancy/contractancy.
 *
 * @tparam retain_map If true, returns tensor for gradient tracking; otherwise returns scalar
 * @param self Input stress tensor
 * @param voidr Void ratio (e) of the soil
 * @param err Error code pointer for error reporting
 * @param pressure Pre-computed mean pressure (optional)
 *
 * @return Scalar_type<retain_map> Dilatancy coefficient ψ_m
 **/
template <bool retain_map = false>
[[nodiscard]] inline auto calc_psim(const torch::Tensor &self, utils::data_t voidr, ErrorCode *err,
                                    Scalar_Type pressure = {}) -> Scalar_type<retain_map> {
  if constexpr (retain_map) {
    auto res = impl::Calc_psim_::call<true>(self, voidr, err, pressure);
    return res;
  } else {
    auto res = impl::Calc_psim_::call<false>(self, voidr, err, pressure);
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
 * @param shvar Shared variables container
 * @param voidr Void ratio (e) of the soil
 * @param err Error code pointer for error reporting
 * @param epsilon Small value for numerical stability
 * @param pressure Pre-computed mean pressure (optional)
 * @param lamda_alpha Pre-computed plastic multiplier for back-stress evolution (optional)
 *
 * @return Scalar_type<retain_map> Dilatancy coefficient with back-stress ψ_m_α
 **/
template <bool retain_map = false>
[[nodiscard]] inline auto calc_psim_alpha(const core::ShareVar &shvar, utils::data_t voidr,
                                          ErrorCode *err, utils::data_t epsilon = 1e-12,
                                          Scalar_Type pressure = {}, Scalar_Type lamda_alpha = {})
    -> Scalar_type<retain_map> {
  auto &shvarImpl = shvar.GetShareVarImpl();
  if constexpr (retain_map) {
    auto res =
        impl::Calc_psim_alpha_::call<true>(shvarImpl, voidr, err, epsilon, pressure, lamda_alpha);
    return res;
  } else {
    auto res =
        impl::Calc_psim_alpha_::call<false>(shvarImpl, voidr, err, epsilon, pressure, lamda_alpha);
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
 * @param shvar Shared variables container
 * @param voidr Void ratio (e) of the soil
 * @param options Plasticity model options
 * @param err Error code pointer for error reporting
 * @param epsilon Small value for numerical stability
 *
 * @return If retain_map=true: pair of tensors (dilatancy parameters)
 *         If retain_map=false: pair of scalars (dilatancy parameters)
 **/
template <bool retain_map = false>
[[nodiscard]] inline auto dilatancy(const core::ShareVar &shvar, utils::data_t voidr,
                                    const core::PlasticOptions &options = {},
                                    ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
    -> std::conditional_t<retain_map, std::pair<torch::Tensor, torch::Tensor>,
                          std::pair<utils::data_t, utils::data_t>> {
  auto &shvarImpl = shvar.GetShareVarImpl();
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
 * @param shvar Shared variables container (stress state)
 * @param stvar State variables container (internal variables)
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
[[nodiscard]] inline auto evolution_Kp(const core::ShareVar &shvar, const core::StateVar &stvar,
                                       const core::PlasticOptions &options = {},
                                       ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
    -> std::tuple<torch::Tensor, torch::Tensor, Scalar_type<retain_map>> {
  auto &shvarImpl = shvar.GetShareVarImpl();
  auto &stvarImpl = stvar.GetShareVarImpl();
  if constexpr (retain_map) {
    auto res = impl::Evolution_Dkp_::call<true>(shvarImpl, stvarImpl, options, err, epsilon);
    return res;
  } else {
    auto res = impl::Evolution_Dkp_::call<false>(shvarImpl, stvarImpl, options, err, epsilon);
    return res;
  }
}

} // namespace umat::core

#endif // CORE_TENSOROPTIONS_H
