#include "ops/Elastic_ops.h"
#include "core/StressTensor.h"
#include "core/TensorOptions.h"
#include "torch/headeronly/util/Exception.h"
#include "torch/optim/optimizer.h"
#include "utils/TypeMap.h"
#include "utils/base_config.h"
#include <type_traits>
#include <utility>

namespace umat::ops::_ops {
using namespace torch;
using namespace utils;
using namespace core;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif
/**
 * @brief
 * @tparam return_pair
 * @param  shvar
 * @param  voidr
 * @param  depsln
 * @param  err
 * @param  epsilon
 *
 * @return std::conditional_t<return_pair, std::pair<torch::Tensor, torch::Tensor>, torch::Tensor>
 **/
template <bool return_pair>
auto Calc_dsigma_impl::call(const core::ShareVar &shvar, utils::data_t voidr,
                            const core::StressTensor &depsln, ErrorCode *err, utils::data_t epsilon)
    -> std::conditional_t<return_pair, std::pair<torch::Tensor, torch::Tensor>, torch::Tensor> {
  auto elastiff = core::stiffness<false>(shvar, voidr, err, epsilon);

  auto elastiff_2d = elastiff.view({-1, elastiff.size(2) * elastiff.size(3)});
  // 2. 维度变换：2维 [K,L] → 2维 [K*L, 1]
  auto depsln_2d = depsln->view({-1, 1});
  // 3. 矩阵乘法 + reshape 回目标形状
  auto dsigma = elastiff_2d.mm(depsln_2d).view({elastiff.size(0), elastiff.size(1)});
  if constexpr (return_pair) {
    return std::make_pair(dsigma, elastiff);
  } else {
    return dsigma;
  }
}
/**
 * @brief
 * @tparam retain_map
 * @param  shvar
 * @param  voidr
 * @param  depsln
 * @param  err
 * @param  epsilon
 *
 * @return std::conditional_t<retain_map, torch::Tensor, utils::data_t>
 **/
template <bool retain_map>
auto Pressure_with_depsln_impl::call(const core::ShareVar &shvar, utils::data_t voidr,
                                     const core::StressTensor &depsln, ErrorCode *err,
                                     utils::data_t epsilon)
    -> std::conditional_t<retain_map, torch::Tensor, utils::data_t> {
  auto dsigma = Calc_dsigma_impl::call<false>(shvar, voidr, depsln, err, epsilon);
#ifdef STRICT_CHECK_ENABLED
  if (err)
    if (*err != ErrorCode::Success) {
      STD_TORCH_CHECK(false,
                      "Pressure_with_depsln_impl::call output variable has occured some error");
    }
#endif
  const auto &stress = shvar.get_stress_tensor();
  auto upd_stress = stress + dsigma;
  if constexpr (retain_map) {
    return core::mean_pressure<true>(upd_stress, err);
  } else {
    return core::mean_pressure<false>(upd_stress, err);
  }
}
/**
 * @brief
 * @tparam retain_map
 * @param  shvar
 * @param  voidr
 * @param  depsln
 * @param  err
 * @param  epsilon
 *
 * @return std::conditional_t<retain_map, torch::Tensor, utils::data_t>
 **/
template <bool retain_map>
auto Ftol_with_depsln_impl::call(const core::ShareVar &shvar, utils::data_t voidr,
                                 const core::StressTensor &depsln, ErrorCode *err,
                                 utils::data_t epsilon)
    -> std::conditional_t<retain_map, torch::Tensor, utils::data_t> {
  auto dsigma = Calc_dsigma_impl::call(shvar, voidr, depsln, err, epsilon);

  auto state = shvar.get_state();
  const auto &stress = shvar.get_stress_tensor();
  auto upd_stress = stress + dsigma;
  auto new_shvars = shvar.create_shvar_from_new_stress(StressTensor(upd_stress, state));
  if constexpr (retain_map) {
    return core::calc_yield<true>(new_shvars, err);
  } else {
    return core::calc_yield<false>(new_shvars, err);
  }
}

// 模板实例化
template auto Calc_dsigma_impl::call<true>(const core::ShareVar &shvar, utils::data_t voidr,
                                           const core::StressTensor &depsln, ErrorCode *err,
                                           utils::data_t epsilon)
    -> std::pair<torch::Tensor, torch::Tensor>;

template auto Calc_dsigma_impl::call<false>(const core::ShareVar &shvar, utils::data_t voidr,
                                            const core::StressTensor &depsln, ErrorCode *err,
                                            utils::data_t epsilon) -> torch::Tensor;

template auto Pressure_with_depsln_impl::call<true>(const core::ShareVar &shvar,
                                                    utils::data_t voidr,
                                                    const core::StressTensor &depsln,
                                                    ErrorCode *err, utils::data_t epsilon)
    -> torch::Tensor;

template auto Pressure_with_depsln_impl::call<false>(const core::ShareVar &shvar,
                                                     utils::data_t voidr,
                                                     const core::StressTensor &depsln,
                                                     ErrorCode *err, utils::data_t epsilon)
    -> utils::data_t;

template auto Ftol_with_depsln_impl::call<true>(const core::ShareVar &shvar, utils::data_t voidr,
                                                const core::StressTensor &depsln, ErrorCode *err,
                                                utils::data_t epsilon) -> torch::Tensor;

template auto Ftol_with_depsln_impl::call<false>(const core::ShareVar &shvar, utils::data_t voidr,
                                                 const core::StressTensor &depsln, ErrorCode *err,
                                                 utils::data_t epsilon) -> utils::data_t;

} // namespace umat::ops::_ops
