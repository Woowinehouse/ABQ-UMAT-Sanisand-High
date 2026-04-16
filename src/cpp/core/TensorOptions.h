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
 * @brief
 * @param  shvar
 *
 * @return Tensor_type
 **/
[[nodiscard]] MYUMAT_API inline auto calc_shvar_norm(const core::ShareVar &shvar, ErrorCode *err)
    -> torch::Tensor {
  return impl::Calc_shvar_norm_::call(shvar.GetShareVarImpl(), err);
}
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
 * @brief
 * @tparam T
 * @tparam U
 *
 * @param  numerator
 * @param  denominator
 * @param  err
 * @param  epsilon
 *
 * @return std::conditional_t<(std::is_same_v<std::decay_t<T>, utils::data_t> &&
 * std::is_same_v<std::decay_t<U>, utils::data_t>),
 * utils::data_t, torch::Tensor>
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
 * @brief
 * @param  self
 *
 * @return bool
 **/
[[nodiscard]] MYUMAT_API inline auto is_nan_inf(const torch::Tensor &self) -> bool {
  return impl::Is_nan_inf_::call(self);
}
/**
 * @brief
 * @param  self
 * @param  retain_map
 *
 * @return Scalar_Type
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
 * @brief
 * @tparam retain_map
 * @param  self
 * @param  err
 *
 * @return torch::Tensor
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
 * @brief
 * @tparam retain_map
 * @param  self
 * @param  err
 * @param  epsilon
 *
 * @return torch::Tensor
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
 * @brief
 * @tparam retain_map
 * @param  self : input tensor
 * @param  err
 * @param  epsilon
 *
 * @return Scalar_type<retain_map>
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
 * @brief
 * @tparam retain_map
 * @param  shvar
 * @param  err
 * @param  epsilon
 *
 * @return torch::Tensor
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
 * @brief
 * @tparam retain_map
 * @param  shvarImpl
 * @param  err
 * @param  epsilon
 * @param  norm
 *
 * @return Scalar_type<retain_map>
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
 * @brief
 * @tparam retain_map
 * @tparam is_grad
 * @tparam Dtype
 * @param  shvar
 * @param  err
 * @param  epsilon
 * @param  cos3t
 *
 * @return Scalar_Type
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
 * @brief
 * @tparam retain_map
 * @tparam Dtype
 * @param  shvar
 * @param  err
 * @param  epsilon
 * @param  gtheta
 *
 * @return Scalar_type<retain_map>
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
 * @brief
 * @tparam retain_map
 * @tparam Dtype
 * @param  self
 * @param  err
 * @param  pressure
 *
 * @return Scalar_type<retain_map>
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
 * @brief
 * @tparam retain_map
 * @param  shvar
 * @param  voidr
 * @param  err
 * @param  epsilon
 *
 * @return std::conditional_t<retain_map, std::pair<torch::Tensor, torch::Tensor>,
 * std::pair<utils::data_t, utils::data_t>>
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
 * @brief
 * @tparam retain_map
 * @param  shvar
 * @param  voidr
 * @param  err
 * @param  epsilon
 *
 * @return Tensor_type
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
 * @brief
 * @tparam retain_map
 * @param  shvar
 * @param  err
 *
 * @return Scalar_type<retain_map>
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
 * @brief
 * @param  shvar
 * @param  err
 *
 * @return torch::Tensor
 **/
[[nodiscard]] MYUMAT_API inline auto pfpsigma(const core::ShareVar &shvar, ErrorCode *err = nullptr)
    -> torch::Tensor {
  auto &shvarImpl = shvar.GetShareVarImpl();
  return impl::Pfpsigma_::call(shvarImpl, err);
}
/**
 * @brief
 * @param  shvar
 * @param  voidr
 * @param  options
 * @param  retain_map
 *
 * @return Tensor_type
 **/
[[nodiscard]] MYUMAT_API inline auto pgpsigma(const core::ShareVar &shvar, utils::data_t voidr,
                                              const core::PlasticOptions &options = {},
                                              ErrorCode *err = nullptr,
                                              utils::data_t epsilon = 1e-12) -> torch::Tensor {
  auto &shvarImpl = shvar.GetShareVarImpl();
  return impl::Pgpsigma_::call(shvarImpl, voidr, options, err, epsilon);
}
/**
 * @brief
 * @tparam retain_map
 * @tparam Dtype
 * @param  self
 * @param  voidr
 * @param  err
 * @param  pressure
 *
 * @return Scalar_type<retain_map>
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
 * @brief
 * @tparam retain_map
 * @param  shvar
 * @param  voidr
 * @param  err
 * @param  epsilon
 *
 * @return Scalar_type<retain_map>
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
 * @brief
 * @tparam retain_map
 * @param  shvar
 * @param  voidr
 * @param  options
 * @param  err
 * @param  epsilon
 *
 * @return std::conditional_t<retain_map, std::pair<torch::Tensor, torch::Tensor>,
 * std::pair<utils::data_t, utils::data_t>>
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
 * @brief
 * @tparam retain_map
 * @param  shvar
 * @param  stvars
 * @param  options
 * @param  err
 * @param  epsilon
 *
 * @return std::tuple<torch::Tensor, torch::Tensor, Scalar_type<retain_map>>
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