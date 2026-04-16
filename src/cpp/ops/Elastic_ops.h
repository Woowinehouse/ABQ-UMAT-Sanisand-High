#ifndef OPS_ELASTIC_OPS_H
#define OPS_ELASTIC_OPS_H

#include "core/ShareVar.h"
#include "core/StateVar.h"
#include "core/TensorOptions.h"
#include "torch/torch.h"
#include "utils/TypeMap.h"
#include "utils/base_config.h"
#include "utils/export.h"

// namespace core

namespace umat::ops::_ops {
struct MYUMAT_API Calc_dsigma_impl {
  template <bool return_pair = false>
  [[nodiscard]] static auto call(const core::ShareVar &shvar, utils::data_t voidr,
                                 const core::StressTensor &depsln, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12)
      -> std::conditional_t<return_pair, std::pair<torch::Tensor, torch::Tensor>, torch::Tensor>;
};

struct MYUMAT_API Pressure_with_depsln_impl {

  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::ShareVar &shvar, utils::data_t voidr,
                                 const core::StressTensor &depsln, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12)
      -> std::conditional_t<retain_map, torch::Tensor, utils::data_t>;
};
struct MYUMAT_API Ftol_with_depsln_impl {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::ShareVar &shvar, utils::data_t voidr,
                                 const core::StressTensor &depsln, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12)
      -> std::conditional_t<retain_map, torch::Tensor, utils::data_t>;
};
// 模板实例化
extern template auto Calc_dsigma_impl::call<true>(const core::ShareVar &shvar, utils::data_t voidr,
                                                  const core::StressTensor &depsln, ErrorCode *err,
                                                  utils::data_t epsilon)
    -> std::pair<torch::Tensor, torch::Tensor>;

extern template auto Calc_dsigma_impl::call<false>(const core::ShareVar &shvar, utils::data_t voidr,
                                                   const core::StressTensor &depsln, ErrorCode *err,
                                                   utils::data_t epsilon) -> torch::Tensor;

extern template auto Pressure_with_depsln_impl::call<true>(const core::ShareVar &shvar,
                                                           utils::data_t voidr,
                                                           const core::StressTensor &depsln,
                                                           ErrorCode *err, utils::data_t epsilon)
    -> torch::Tensor;

extern template auto Pressure_with_depsln_impl::call<false>(const core::ShareVar &shvar,
                                                            utils::data_t voidr,
                                                            const core::StressTensor &depsln,
                                                            ErrorCode *err, utils::data_t epsilon)
    -> utils::data_t;

extern template auto Ftol_with_depsln_impl::call<true>(const core::ShareVar &shvar,
                                                       utils::data_t voidr,
                                                       const core::StressTensor &depsln,
                                                       ErrorCode *err, utils::data_t epsilon)
    -> torch::Tensor;

extern template auto Ftol_with_depsln_impl::call<false>(const core::ShareVar &shvar,
                                                        utils::data_t voidr,
                                                        const core::StressTensor &depsln,
                                                        ErrorCode *err, utils::data_t epsilon)
    -> utils::data_t;
} // namespace umat::ops::_ops

#endif // OPS_ELASTIC_H
