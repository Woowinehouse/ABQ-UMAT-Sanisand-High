#ifndef OPS_DRIFT_H
#define OPS_DRIFT_H
#include "core/ShareVar.h"
#include "core/StateVar.h"
#include "utils/base_config.h"
#include "utils/export.h"
namespace umat::ops {
struct MYUMAT_API drift {
  using Tensor = torch::Tensor;
  using data_t = utils::data_t;
  using State = utils::StressState;
  using StressTensor = core::StressTensor;
  using ShareVar = core::ShareVar;
  using StateVar = core::StateVar;
  [[nodiscard]] static auto along_flow_direction(const ShareVar &shvar, const StateVar &stvar,
                                                 ErrorCode *err = nullptr,
                                                 utils::data_t epsilon = 1e-12)
      -> std::pair<core::ShareVar, utils::data_t>;
  [[nodiscard]] static auto along_radial_direction(const ShareVar &shvar, ErrorCode *err = nullptr,
                                                   utils::data_t epsilon = 1e-12)
      -> std::pair<core::ShareVar, utils::data_t>;
  [[nodiscard]] static auto drift_shareVar_impl(core::ShareVar &shvar, core::StateVar &stvar,
                                                int noel, int npt, ErrorCode *err = nullptr,
                                                utils::data_t epsilon = 1e-12,
                                                utils::index_t iter_max = 8)
      -> std::optional<utils::data_t>;
};
template <typename... Args>
[[nodiscard]] auto drift_shareVar(Args &&...args) -> std::optional<utils::data_t> {
  return drift::drift_shareVar_impl(std::forward<Args>(args)...);
}
} // namespace umat::ops
#endif // OPS_DRIFT_H