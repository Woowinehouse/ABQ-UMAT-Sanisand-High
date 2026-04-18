#ifndef UMAT_MATH_H
#define UMAT_MATH_H
#include "core/ShareVar.h"
#include "core/StateVar.h"
#include "ops/Elastic.h"
#include "utils/export.h"

namespace umat {
using bisection_func =
    std::function<utils::data_t(const core::ShareVar &, const core::StateVar &,
                                const core::StressTensor &, ErrorCode *, utils::data_t)>;
class MYUMAT_API math {
  using Tensor = torch::Tensor;
  using data_t = utils::data_t;
  using State = utils::StressState;
  using StressTensor = core::StressTensor;
  using ShareVar = core::ShareVar;
  using StateVar = core::StateVar;

  public:
  [[nodiscard]] static auto elastic_update(ShareVar &shvar, StateVar &stvar,
                                           const StressTensor &depsln, ErrorCode *err = nullptr,
                                           utils::data_t epsilon = 1e-12) -> torch::Tensor;
  static auto intchc(const ShareVar &shvar, const StateVar &stvar, const core::StressTensor &depsln,
                     ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12) -> utils::data_t;
  /**
   * @brief
   * @param  func
   * @param  shvars
   * @param  stvars
   * @param  depsln
   * @param  conditions
   * @param  left
   * @param  right
   * @param  return_tensor
   * @param  eplison
   * @param  iter
   *
   * @return Scalar_Type
   **/
  static auto bisection_method(bisection_func func, const ShareVar &shvar, const StateVar &stvar,
                               const core::StressTensor &depsln, utils::data_t conditions,
                               utils::data_t left = 0.0, utils::data_t right = 1.0,
                               ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                               utils::index_t max_iter = 100) -> utils::data_t;
  using pair_elstop = std::pair<core::ShareVar, torch::Tensor>;
  using pair_elstop_Type = std::variant<std::monostate, pair_elstop, ErrorCode>;
  /**
   * @brief
   * @param  shvars
   * @param  stvars
   * @param  depsln
   *
   * @return pair_elstop_Type
   **/
  [[nodiscard]] static auto elstop(const core::ShareVar &shvar, const core::StateVar &stvar,
                                   const core::StressTensor &depsln, ErrorCode *err = nullptr,
                                   utils::data_t epsilon = 1e-12) -> pair_elstop;
  [[nodiscard]] static auto onyield(ShareVar &shvars, core::StateVar &stvars,
                                    const core::StressTensor &depsln, int noel, int npt,
                                    ErrorCode *err = nullptr, data_t epsilon = 1e-12,
                                    utils::index_t max_iter = 200) -> torch::Tensor;
  [[nodiscard]] static auto calc_residual(const core::ShareVar &shfor, const core::ShareVar &shsec,
                                          const core::ShareVar &shvar_avg, ErrorCode *err = nullptr,
                                          utils::data_t epsilon = 1e-12) -> utils::data_t;
  static auto drift_shareVar(ShareVar &shvar, const core::StateVar &stvar, int noel, int npt,
                             ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                             utils::index_t iter = 8) -> bool;
  [[nodiscard]] static auto drift_along_flow_direction(const ShareVar &shvar, const StateVar &stvar,
                                                       ErrorCode *err = nullptr,
                                                       utils::data_t epsilon = 1e-12)
      -> core::ShareVar;
  [[nodiscard]] static auto drift_along_radial_direction(const ShareVar &shvar,
                                                         ErrorCode *err = nullptr,
                                                         utils::data_t epsilon = 1e-12)
      -> core::ShareVar;
  [[nodiscard]] static auto Consistent_stiffness_matrix(const ShareVar &shvar,
                                                        const StateVar &stvar, utils::data_t lamda,
                                                        ErrorCode *err = nullptr,
                                                        utils::data_t epsilon = 1e-12) -> Tensor;
}; // class math

} // namespace umat

#endif // UMAT_MATH_H