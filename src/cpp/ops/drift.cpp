#include "ops/drift.h"
#include "ATen/ops/einsum.h"
#include "ATen/ops/norm.h"
#include "core/ShareVar.h"
#include "core/TensorOptions.h"
#include "ops/Elastic.h"
#include "torch/headeronly/util/Exception.h"
#include "utils/TypeMap.h"
#include "utils/base_config.h"
#include <cstdlib>
#include <optional>
#include <span>
#include <utility>

namespace umat::ops {
using namespace torch;
using namespace utils;
using namespace std;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif
auto drift::along_flow_direction(const ShareVar &shvar, const StateVar &stvar, ErrorCode *err,
                                 utils::data_t epsilon)
    -> std::pair<core::ShareVar, utils::data_t> {
  auto dftol = ops::calc_yield(shvar, err);

  auto stiff = ops::stiffness<false>(shvar, stvar, err, epsilon);
  auto pfsig = ops::pfpsigma(shvar, err);
  auto pgsig = ops::pgpsimga(shvar, stvar);
  auto [Ralpha, Rp0, Kp] = ops::evolution_Kp<false>(shvar, stvar);
  auto state = shvar.get_state();
#ifdef DEBUG_SPAN_ENABLED
  auto pfsig_view_ = std::span<data_t>(pfsig.data_ptr<data_t>(), pfsig.numel());
  auto stiff_view_ = std::span<data_t>(stiff.data_ptr<data_t>(), stiff.numel());
  auto pgsig_view_ = std::span<data_t>(pgsig.data_ptr<data_t>(), pgsig.numel());
  auto Ralpha_view_ = std::span<data_t>(Ralpha.data_ptr<data_t>(), Ralpha.numel());
  auto Rp0_view_ = std::span<data_t>(Rp0.data_ptr<data_t>(), Rp0.numel());
#endif

  auto stiff_pgsig = at::einsum("ijkl,kl->ij", {stiff, pgsig});
  auto denominator_left = at::einsum("ij,ijkl,kl->", {pfsig, stiff, pgsig}).item<data_t>();
  auto denominator = denominator_left + Kp;
  auto dlamda = core::safe_divide(dftol, denominator, err, epsilon);
#ifdef DEBUG_SPAN_ENABLED
  auto stiff_pgsig_view = std::span<data_t>(stiff_pgsig.data_ptr<data_t>(), stiff_pgsig.numel());
#endif
  auto Rshvar = core::make_ShareVar(-dlamda * stiff_pgsig, dlamda * Ralpha, dlamda * Rp0, state);
  auto Shvar_corr = Rshvar + shvar;
  return std::make_pair(Shvar_corr, dlamda);
}
auto drift::along_radial_direction(const ShareVar &shvar, ErrorCode *err, utils::data_t epsilon)
    -> std::pair<core::ShareVar, utils::data_t> {
  auto dftol = ops::calc_yield(shvar);
  auto pfsig = ops::pfpsigma(shvar);
#ifdef DEBUG_SPAN_ENABLED
  auto pfsig_view = std::span<data_t>(pfsig.data_ptr<data_t>(), pfsig.numel());
#endif
  auto denominator = norm(pfsig, 2).item<data_t>();
  auto dlamda = core::safe_divide(dftol, denominator, err, epsilon);
  auto state = shvar.get_state();
  auto dsigma = StressTensor(-dlamda * pfsig, state);
  auto Shvar_corr = shvar.create_shvar_with_dstress(dsigma);
  return std::make_pair(Shvar_corr, dlamda);
}
auto drift::drift_shareVar_impl(core::ShareVar &shvar, core::StateVar &stvar, int noel, int npt,
                                ErrorCode *err, utils::data_t epsilon, utils::index_t iter_max)
    -> std::optional<utils::data_t> {
  auto dlmada_tol = 0.0;
  // 开始迭代
#ifdef STRICT_CHECK_ENABLED
  if (err) {
    if (*err != ErrorCode::Success) {
      STD_TORCH_CHECK(false,
                      "there has some error ocurred before the function of drift ShareVar at noel  "
                      "= %d , npt = %d ",
                      noel, npt);
    }
  }
#endif
  for (index_t i = 0; i < iter_max; i++) {
    // 当前屈服面函数值
    const auto df_cur = ops::calc_yield<false>(shvar, err);
    if (err && *err != ErrorCode::Success)
      return false;
    // check the df result whether has any err
    if (std::abs(df_cur) <= FTOLR)
      return dlmada_tol;
    // 沿流动方向漂移
    const auto [shvar_flow, dlmada] = along_flow_direction(shvar, stvar, err, epsilon);
    if (err && *err != ErrorCode::Success)
      return false;
    const auto df_flow = ops::calc_yield(shvar_flow, err);
    if (err && *err != ErrorCode::Success)
      return false;

    // 选择更优方向
    if (std::abs(df_flow) < std::abs(df_cur)) {
      shvar = std::move(shvar_flow);
    } else {
      const auto [shvar_flow, dlmada] = along_radial_direction(shvar, err, epsilon);
      shvar = shvar_flow;
      if (err && *err != ErrorCode::Success)
        return false;
    }
    dlmada_tol += dlmada;
  }
  if (err) {
    *err = ErrorCode::IterError;
  }
  stvar.set_pnewdt(0.5);
  return false;
}
} // namespace umat::ops