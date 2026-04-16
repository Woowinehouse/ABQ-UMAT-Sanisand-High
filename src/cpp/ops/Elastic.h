#ifndef OPS_ELASTIC_H
#define OPS_ELASTIC_H
#include "core/ShareVar.h"
#include "core/StateVar.h"
#include "ops/Elastic_ops.h"
#include "torch/torch.h"
#include "utils/TypeMap.h"
#include "utils/export.h"

namespace umat::ops {
template <typename Dtype>
concept StressTypes =
    std::is_same_v<Dtype, torch::Tensor> || std::is_same_v<Dtype, core::StressTensor> ||
    std::is_same_v<Dtype, core::ShareVar>;
template <typename Dtype>
concept VoidrTypes = std::is_same_v<Dtype, utils::data_t> || std::is_same_v<Dtype, core::StateVar>;
namespace detail {
template <StressTypes Dtype>
auto get_stress(Dtype value) -> torch::Tensor {
  if constexpr (std::is_same_v<Dtype, torch::Tensor>) {
    return value;
  } else if constexpr (std::is_same_v<Dtype, core::StressTensor>) {
    return value.get_tensor();
  } else {
    return value.get_stress_tensor();
  }
}
template <VoidrTypes Dtype>
auto get_voidr(Dtype value) -> utils::data_t {
  if constexpr (std::is_same_v<Dtype, utils::data_t>) {
    return value;
  } else {
    return value.get_voidr();
  }
}
} // namespace detail
/**
 * @brief
 * @param  shvar
 * @param  err
 *
 * @return Tensor_type
 **/
MYUMAT_API inline auto calc_shvar_norm(const core::ShareVar &shvar, ErrorCode *err)
    -> torch::Tensor {
  return core::calc_shvar_norm(shvar, err);
}

template <bool retain_map = false>
inline auto calc_Cosine_angle(const torch::Tensor &lhs, const torch::Tensor &rhs,
                              ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
    -> std::conditional_t<retain_map, torch::Tensor, utils::data_t> {
  if constexpr (retain_map) {
    auto res = core::calc_Cosine_angle<true>(lhs, rhs, err, epsilon);
    return res;
  } else {
    auto res = core::calc_Cosine_angle<false>(lhs, rhs, err, epsilon);
    return res;
  }
}
template <utils::Scalartype T, utils::Scalartype U>
inline auto safe_divide(T numerator, U denominator, ErrorCode *err = nullptr,
                        utils::data_t epsilon = 1e-12)
    -> std::conditional_t<(std::is_same_v<std::decay_t<T>, utils::data_t> &&
                           std::is_same_v<std::decay_t<U>, utils::data_t>),
                          utils::data_t, torch::Tensor> {
  return core::safe_divide(numerator, denominator, err, epsilon);
}
template <bool retain_map = false, StressTypes Dtype>
inline auto mean_pressure(Dtype stress, ErrorCode *err = nullptr)
    -> std::conditional_t<retain_map, torch::Tensor, utils::data_t> {
  torch::Tensor self = detail::get_stress(stress);
  if constexpr (retain_map) {
    auto res = core::mean_pressure<true>(self, err);
    return res;
  } else {
    auto res = core::mean_pressure<false>(self, err);
    return res;
  }
}
template <bool retain_map = false, StressTypes Dtype>
inline auto deviatoric(Dtype stress, ErrorCode *err = nullptr) -> torch::Tensor {
  torch::Tensor self = detail::get_stress(stress);
  if constexpr (retain_map) {
    auto res = core::deviatoric<true>(self, err);
    return res;
  } else {
    auto res = core::deviatoric<false>(self, err);
    return res;
  }
}
/**
 * @brief
 * @tparam retain_map
 * @tparam Dtype
 * @param  stress
 * @param  err
 * @param  epsilon
 *
 * @return torch::Tensor
 **/
template <bool retain_map = false, StressTypes Dtype>
inline auto stressRatio(Dtype stress, ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
    -> torch::Tensor {
  torch::Tensor self = detail::get_stress(stress);
  if constexpr (retain_map) {
    auto res = core::stressRatio<true>(self, err, epsilon);
    return res;
  } else {
    auto res = core::stressRatio<false>(self, err, epsilon);
    return res;
  }
}
template <bool retain_map = false, StressTypes Dtype>
inline auto Calc_Rm(Dtype stress, ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                    std::conditional_t<retain_map, std::monostate, torch::Tensor> ratio = {})
    -> std::conditional_t<retain_map, torch::Tensor, utils::data_t> {
  torch::Tensor self = detail::get_stress(stress);
  if constexpr (retain_map) {
    auto res = core::Calc_Rm<true>(self, err, epsilon);
    return res;
  } else {
    auto res = core::Calc_Rm<false>(self, err, epsilon, ratio);
    return res;
  }
}
template <bool retain_map = false>
inline auto
loadingDirection(const core::ShareVar &shvar, ErrorCode *err = nullptr,
                 utils::data_t epsilon = 1e-12,
                 std::conditional_t<retain_map, std::monostate, torch::Tensor> ratio = {})
    -> torch::Tensor {
  if constexpr (retain_map) {
    auto res = core::loadingDirection<true>(shvar, err, epsilon);
    return res;
  } else {
    auto res = core::loadingDirection<false>(shvar, err, epsilon, ratio);
    return res;
  }
}
template <bool retain_map = false>
inline auto cos3theta(const core::ShareVar &shvar, ErrorCode *err = nullptr,
                      utils::data_t epsilon = 1e-12,
                      std::conditional_t<retain_map, std::monostate, torch::Tensor> norm = {})
    -> std::conditional_t<retain_map, torch::Tensor, utils::data_t> {
  if constexpr (retain_map) {
    auto res = core::cos3theta<true>(shvar, err, epsilon);
    return res;
  } else {
    auto res = core::cos3theta<false>(shvar, err, epsilon, norm);
    return res;
  }
}
template <bool retain_map = false, bool is_grad = false>
inline auto calc_gtheta(const core::ShareVar &shvar, ErrorCode *err = nullptr,
                        utils::data_t epsilon = 1e-12, Scalar_Type cos3t = {})
    -> std::conditional_t<
        is_grad,
        std::pair<std::conditional_t<retain_map, torch::Tensor, utils::data_t>, torch::Tensor>,
        std::conditional_t<retain_map, torch::Tensor, utils::data_t>> {
  if constexpr (retain_map) {
    if constexpr (is_grad) {
      auto res = core::calc_gtheta<true, true>(shvar, err, epsilon, cos3t);
      return res;
    } else {
      auto res = core::calc_gtheta<true, false>(shvar, err, epsilon, cos3t);
      return res;
    }
  } else {
    if constexpr (is_grad) {
      auto res = core::calc_gtheta<false, true>(shvar, err, epsilon, cos3t);
      return res;
    } else {
      auto res = core::calc_gtheta<false, false>(shvar, err, epsilon, cos3t);
      return res;
    }
  }
}
template <bool retain_map = false>
auto calc_lamda(const core::ShareVar &shvar, ErrorCode *err = nullptr,
                utils::data_t epsilon = 1e-12, Scalar_Type gtheta = {})
    -> std::conditional_t<retain_map, torch::Tensor, utils::data_t> {
  if constexpr (retain_map) {
    auto res = core::calc_lamda<true>(shvar, err, epsilon, gtheta);
    return res;
  } else {
    utils::data_t res = core::calc_lamda<false>(shvar, err, epsilon, gtheta);
    return res;
  }
}
template <bool retain_map = false, StressTypes Dtype>
inline auto calc_GV(Dtype stress, ErrorCode *err = nullptr, Scalar_Type pressure = {})
    -> std::conditional_t<retain_map, torch::Tensor, utils::data_t> {
  torch::Tensor self = detail::get_stress(stress);
  if constexpr (retain_map) {
    auto res = calc_GV<true>(self, err, pressure);
    return res;
  } else {
    auto res = calc_GV<false>(self, err, pressure);
    return res;
  }
}
template <bool retain_map = false, VoidrTypes Dtype>
inline auto calc_shear_bulk(const core::ShareVar &shvar, Dtype voidrs, ErrorCode *err = nullptr,
                            utils::data_t epsilon = 1e-12)
    -> std::conditional_t<retain_map, pair_tensor, pair_data> {
  utils::data_t voidr = get_voidr(voidrs);
  if constexpr (retain_map) {
    auto res = core::calc_shear_bulk<true>(shvar, voidr, err, epsilon);
    return res;
  } else {
    auto res = core::calc_shear_bulk<false>(shvar, voidr, err, epsilon);
    return res;
  }
}
template <bool retain_map = false, VoidrTypes Dtype>
inline auto stiffness(const core::ShareVar &shvar, Dtype voidrs, ErrorCode *err = nullptr,
                      utils::data_t epsilon = 1e-12) -> torch::Tensor {
  utils::data_t voidr = detail::get_voidr(voidrs);
  if constexpr (retain_map) {
    auto res = core::stiffness<true>(shvar, voidr, err, epsilon);
    return res;
  } else {
    auto res = core::stiffness<false>(shvar, voidr, err, epsilon);
    return res;
  }
}
template <bool retain_map = false>
inline auto calc_yield(const core::ShareVar &shvar, ErrorCode *err = nullptr)
    -> std::conditional_t<retain_map, torch::Tensor, utils::data_t> {
  if constexpr (retain_map) {
    auto res = core::calc_yield<true>(shvar, err);
    return res;
  } else {
    auto res = core::calc_yield<false>(shvar, err);
    return res;
  }
}
MYUMAT_API inline auto pfpsigma(const core::ShareVar &shvar, ErrorCode *err = nullptr)
    -> torch::Tensor {
  auto res = core::pfpsigma(shvar, err);
  return res;
}
template <VoidrTypes Dtype>
inline auto pgpsimga(const core::ShareVar &shvar, Dtype voidrs,
                     const core::PlasticOptions &options = {}, ErrorCode *err = nullptr,
                     utils::data_t epsilon = 1e-12) -> torch::Tensor {
  utils::data_t voidr = detail::get_voidr(voidrs);
  auto res = core::pgpsigma(shvar, voidr, options, err, epsilon);
  return res;
}
template <bool retain_map = false, StressTypes T, VoidrTypes U>
inline auto calc_psim(T stress, U voidrs, ErrorCode *err, Scalar_Type pressure = {})
    -> std::conditional_t<retain_map, torch::Tensor, utils::data_t> {
  torch::Tensor self = detail::get_stress(stress);
  utils::data_t voidr = detail::get_voidr(voidrs);
  if constexpr (retain_map) {
    auto res = core::calc_psim<true>(self, voidr, err, pressure);
    return res;
  } else {
    auto res = core::calc_psim<false>(self, voidr, err, pressure);
    return res;
  }
}
template <bool retain_map = false, VoidrTypes Dtype>
inline auto calc_psim_alpha(const core::ShareVar &shvar, Dtype voidrs, ErrorCode *err,
                            utils::data_t epsilon = 1e-12, Scalar_Type pressure = {},
                            Scalar_Type lamda_alpha = {})
    -> std::conditional_t<retain_map, torch::Tensor, utils::data_t> {
  utils::data_t voidr = detail::get_voidr(voidrs);
  if constexpr (retain_map) {
    auto res = core::calc_psim_alpha<true>(shvar, voidr, err, epsilon, pressure, lamda_alpha);
    return res;
  } else {
    auto res = core::calc_psim_alpha<false>(shvar, voidr, err, epsilon, pressure, lamda_alpha);
    return res;
  }
}
template <bool retain_map = false, VoidrTypes Dtype>
MYUMAT_API inline auto dilatancy(const core::ShareVar &shvar, Dtype voidrs,
                                 const core::PlasticOptions &options = {}, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12)
    -> std::conditional_t<retain_map, pair_tensor, pair_data> {
  utils::data_t voidr = get_voidr(voidrs);
  if constexpr (retain_map) {
    auto res = core::dilatancy<true>(shvar, voidr, options, err, epsilon);
    return res;
  } else {
    auto res = core::dilatancy<false>(shvar, voidr, options, err, epsilon);
    return res;
  }
}
template <bool retain_map = false>
inline auto evolution_Kp(const core::ShareVar &shvar, const core::StateVar &stvar,
                         const core::PlasticOptions &options = {}, ErrorCode *err = nullptr,
                         utils::data_t epsilon = 1e-12)
    -> std::tuple<torch::Tensor, torch::Tensor,
                  std::conditional_t<retain_map, torch::Tensor, utils::data_t>> {
  if constexpr (retain_map) {
    auto res = core::evolution_Kp<true>(shvar, stvar, options, err, epsilon);
    return res;
  } else {
    auto res = core::evolution_Kp<false>(shvar, stvar, options, err, epsilon);
    return res;
  }
}
template <bool return_pair = false, VoidrTypes Dtype>
inline auto calc_dsigma(const core::ShareVar &shvar, Dtype voidrs, const core::StressTensor &depsln,
                        ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
    -> std::conditional_t<return_pair, std::pair<torch::Tensor, torch::Tensor>, torch::Tensor> {
  utils::data_t voidr = detail::get_voidr(voidrs);
  if constexpr (return_pair) {
    auto res = _ops::Calc_dsigma_impl::call<true>(shvar, voidr, depsln, err, epsilon);
    return res;
  } else {
    auto res = _ops::Calc_dsigma_impl::call<false>(shvar, voidr, depsln, err, epsilon);
    return res;
  }
}

MYUMAT_API inline auto pressure_with_depsln(const core::ShareVar &shvar,
                                            const core::StateVar &stvar,
                                            const core::StressTensor &depsln,
                                            ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
    -> utils::data_t {
  auto voidr = stvar.get_voidr();
  auto res = _ops::Pressure_with_depsln_impl::call<false>(shvar, voidr, depsln, err, epsilon);
  return res;
}
MYUMAT_API inline auto ftol_with_depsln(const core::ShareVar &shvar, const core::StateVar &stvar,
                                        const core::StressTensor &depsln, ErrorCode *err = nullptr,
                                        utils::data_t epsilon = 1e-12) -> utils::data_t {
  auto voidr = stvar.get_voidr();
  auto res = _ops::Ftol_with_depsln_impl::call<false>(shvar, voidr, depsln, err, epsilon);
  return res;
}
} // namespace umat::ops

#endif // OPS_MEANPRESSURE_H