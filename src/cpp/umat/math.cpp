#include "umat/math.h"
#include "ATen/core/interned_strings.h"
#include "ATen/ops/dot.h"
#include "ATen/ops/einsum.h"
#include "ATen/ops/inverse.h"
#include "ATen/ops/norm.h"
#include "ATen/ops/tensordot.h"
#include "core/ShareVar.h"
#include "core/ShareVarOptions.h"
#include "core/StateVar.h"
#include "core/StressTensor.h"
#include "core/TensorOptions.h"
#include "ops/Elastic.h"
#include "ops/drift.h"
#include "utils/base_config.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <span>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/optim/optimizer.h>
#include <torch/torch.h>
#include <utility>
#include <utils/TypeMap.h>

namespace umat {
using namespace torch;
using namespace utils;
using namespace std;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif
/**
 * @brief
 * @param  shvar
 * @param  stvar
 * @param  depsln
 *
 * @return Tensor_type
 **/
auto math::elastic_update(ShareVar &shvar, StateVar &stvar, const StressTensor &depsln,
                          ErrorCode *err, utils::data_t epsilon) -> Tensor {
  auto [dsigma, stiff] = ops::calc_dsigma<true>(shvar, stvar, depsln, err, epsilon);
#ifdef DEBUG_SPAN_ENABLED
  auto dsigma_view = std::span<data_t>(dsigma.data_ptr<data_t>(), dsigma.numel());
  auto stiff_view = std::span<data_t>(stiff.data_ptr<data_t>(), stiff.numel());
#endif
  const auto state = depsln.get_state();
  shvar.update_stress(dsigma, state);
  stvar.update_voidr(depsln);
  return stiff;
}

/**
 * @brief
 * @param  shvars
 * @param  stvars
 * @param  depsln
 *
 * @return Scalar_Type
 **/
auto math::intchc(const ShareVar &shvar, const StateVar &stvar, const core::StressTensor &depsln,
                  ErrorCode *err, utils::data_t epsilon) -> utils::data_t {
  auto mean_cur = ops::mean_pressure(shvar, err);
  auto mean_etr = ops::pressure_with_depsln(shvar, stvar, depsln, err);

  auto rbd = (mean_cur * mean_etr <= 0.0) ? math::bisection_method(ops::pressure_with_depsln, shvar,
                                                                   stvar, depsln, 0.0, 0.0, 1.0)
                                          : 1.0;

  auto f_left = ops::calc_yield(shvar, err);
  auto f_right = ops::ftol_with_depsln(shvar, stvar, depsln, err, epsilon);
  auto lbd = 0.0;
  if (f_left * f_right >= 0.0) {
    /**
     * @brief 获取屈服面方向与塑性应变方向的余弦值
     *
     **/
    auto pfsig = ops::pfpsigma(shvar);
    auto dsigma = ops::calc_dsigma(shvar, stvar, depsln);
    auto angle = ops::calc_Cosine_angle(pfsig, dsigma, err, epsilon);
    /**
     * @brief : 获取左边界
     *
     **/
    if (angle < 0.0) {
      data_t iter = 0.0;
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
      do {
        auto f_temp = ops::ftol_with_depsln(shvar, stvar, iter * depsln, err, epsilon);
        if (f_temp < -epsilon) {
          lbd = iter;
        } else {
          if (iter >= 1.0) {
            return 0.0;
          }
          iter += 0.01;
        }
      } while (iter <= 1.0);
    } else {
      return 0.0;
    }
  } else {
    auto alout = math::bisection_method(ops::ftol_with_depsln, shvar, stvar, depsln, 0.0, lbd, rbd);
    return alout;
  }
  return std::nan("1");
}
/**
 * @brief
 * @param  func
 * @param  shvars
 * @param  stvars
 * @param  depsln
 * @param  cond
 * @param  left
 * @param  right
 * @param  return_tensor
 * @param  epslion
 * @param  max_iter
 *
 * @return Scalar_Type
 **/
auto math::bisection_method(bisection_func func, const ShareVar &shvar, const StateVar &stvar,
                            const core::StressTensor &depsln, utils::data_t conditions,
                            utils::data_t left, utils::data_t right, ErrorCode *err,
                            utils::data_t epsilon, utils::index_t max_iter) -> utils::data_t {
  STD_TORCH_CHECK(left >= 0.0 && right <= 1.0, " left and right should be in [0,1]");
  STD_TORCH_CHECK(left < right, "The interval can not be emptied ");
  auto f_left = func(shvar, stvar, left * depsln, err, epsilon);
  auto f_right = func(shvar, stvar, right * depsln, err, epsilon);
  auto df_left = f_left - conditions;
  auto df_right = f_right - conditions;
  STD_TORCH_CHECK(df_left * df_right < 0.0,
                  "The function values at the endpoints must have the same sign");
  for (index_t i = 0; i < max_iter; i++) {
    auto mid = left + (right - left) / 2.0;
    auto f_mid = func(shvar, stvar, mid * depsln, err, epsilon);
    auto df_mid = f_mid - conditions;
    if (abs(df_mid) <= FTOLR) {
      return mid;
    }
    if (df_left * df_mid >= 0.0) {
      left = mid;
      df_left = df_mid;
    } else {
      right = mid;
      df_right = df_mid;
    }
  }
  if (err) {
    *err = ErrorCode::IterError;
    return std::nan("1");
  }
  return std::nan("1");
}

/**
 * @brief
 * @param  shvars
 * @param  stvars
 * @param  depsln
 *
 * @return std::variant<std::monostate, pair_elstop, ErrorCode>
 **/
auto math::elstop(const core::ShareVar &shvar, const core::StateVar &stvar,
                  const core::StressTensor &depsln, ErrorCode *err, utils::data_t epsilon)
    -> pair_elstop {
  auto state = depsln.get_state();
  auto norm = ops::loadingDirection<false>(shvar, err, epsilon);
  auto cos3t = ops::cos3theta<false>(shvar, err, epsilon, norm);
  auto gtheta = ops::calc_gtheta<false>(shvar, err, epsilon, cos3t);
  auto psim = ops::calc_psim<false>(shvar, stvar, err);
  auto psim_alpha = ops::calc_psim_alpha<false>(shvar, stvar, err, epsilon);
  auto options = core::make_ShareVarOptions(norm, cos3t, gtheta, psim, psim_alpha);
  //
  const auto pfsig = ops::pfpsigma(shvar, err);
  const auto elastiff = ops::stiffness(shvar, stvar, err, epsilon);
  const auto pgsig = ops::pgpsimga(shvar, stvar, options, err, epsilon);
  const auto [Ralpha, Rp0, Kp] = ops::evolution_Kp(shvar, stvar, options, err, epsilon);
#ifdef DEBUG_SPAN_ENABLED
  auto pfsig_view = std::span<data_t>(pfsig.data_ptr<data_t>(), pfsig.numel());
  auto elastiff_view = std::span<data_t>(elastiff.data_ptr<data_t>(), elastiff.numel());
  auto pgsig_view = std::span<data_t>(pgsig.data_ptr<data_t>(), pgsig.numel());
  auto Ralpha_view = std::span<data_t>(Ralpha.data_ptr<data_t>(), Ralpha.numel());
  auto Rp0_view = std::span<data_t>(Rp0.data_ptr<data_t>(), Rp0.numel());
#endif

  // Implementation for handling the case where all inputs are tensors
  auto pfsig_stiff = at::tensordot(pfsig, elastiff, {0, 1}, {0, 1});
  auto &numerator = pfsig_stiff;
  auto denominator_left = at::dot(pfsig_stiff.flatten(), pgsig.flatten()).item<data_t>();
  auto denominator = denominator_left + Kp;
#ifdef DEBUG_SPAN_ENABLED
  auto numerator_view = std::span<data_t>(numerator.data_ptr<data_t>(), numerator.numel());
#endif
  auto theta = core::safe_divide(numerator, denominator);
  auto dlamda = at::dot(theta.flatten(), depsln->flatten()).item<data_t>();
  auto cplas = at::einsum("ijab,ab,kl->ijkl", {elastiff, pgsig, theta});
  auto depmx = dlamda > 0.0 ? elastiff - cplas : elastiff;
  auto dsigma = at::tensordot(depmx, *depsln, {2, 3}, {0, 1});
  auto Rshvar = core::make_ShareVar(dsigma, Ralpha * dlamda, Rp0 * dlamda, state);
#ifdef DEBUG_SPAN_ENABLED
  auto theta_view = std::span<data_t>(theta.data_ptr<data_t>(), theta.numel());
  auto cplas_view = std::span<data_t>(cplas.data_ptr<data_t>(), cplas.numel());
  auto depmx_view = std::span<data_t>(depmx.data_ptr<data_t>(), depmx.numel());
  auto dsigma_view = std::span<data_t>(dsigma.data_ptr<data_t>(), dsigma.numel());
#endif
  //
  return make_pair(Rshvar, depmx);
}
/**
 * @brief
 * @param  shfor
 * @param  shsec
 * @param  shvar_avg
 * @param  err
 * @param  epsilon
 *
 * @return utils::data_t
 **/
[[nodiscard]] auto math::calc_residual(const core::ShareVar &shfor, const core::ShareVar &shsec,
                                       const core::ShareVar &shvar_avg, ErrorCode *err,
                                       utils::data_t epsilon) -> utils::data_t {
  const auto shdif = shfor - shsec;
  const auto dif_norm = ops::calc_shvar_norm(shdif, err);
  if (err && *err != ErrorCode::Success)
    return epsilon;

  const auto tmp_norm = ops::calc_shvar_norm(shvar_avg, err);
  if (err && *err != ErrorCode::Success)
    return epsilon;

  const auto vartol = core::safe_divide(dif_norm, tmp_norm, err, epsilon);
  if (err && *err != ErrorCode::Success)
    return epsilon;

  return std::max(0.5 * vartol.min().item<utils::data_t>(), epsilon);
}
/**
 * @brief
 * @param  shvar
 * @param  stvar
 * @param  depsln
 * @param  noel
 * @param  npt
 * @param  err
 * @param  epsilon
 * @param  max_iter
 *
 * @return torch::Tensor
 **/
auto math::onyield(core::ShareVar &shvar, core::StateVar &stvar, const core::StressTensor &depsln,
                   int noel, int npt, ErrorCode *err, utils::data_t epsilon,
                   utils::index_t max_iter) -> torch::Tensor {
  constexpr double SSTOL = 1e-6;
  constexpr double BETA_COEFF = 0.8;
  constexpr double MIN_DT = 1e-6;
  constexpr double DT_GROWTH = 1.1;
  constexpr double DT_SHRINK_FACTOR = 0.1;
  constexpr double MIN_DT_ON_FAIL = 1e-3;
  double t = 0.0;
  double dt = 1.0;
  bool nfail = false;
  auto dempx = torch::empty({3, 3, 3, 3}, depsln->options());
  /**
   * @brief
   *
   **/
  dempx.zero_();
  for (index_t iter = 0; iter < max_iter; iter++) {
    const auto dt_deps = dt * depsln;
    auto [Defor, dempx1] = math::elstop(shvar, stvar, dt_deps, err, epsilon);
    if (err && *err != ErrorCode::Success)
      return torch::empty({});
    auto shfor = shvar + Defor;
    auto &stfor = stvar;
    if (shfor.is_lowstress()) {
      if (dt < MIN_DT) {
        break;
      } else {
        dt *= 0.5;
        continue;
      }
    }
    auto [Desec, dempx2] = math::elstop(shfor, stfor, dt_deps, err, epsilon);
    if (err && *err != ErrorCode::Success)
      return torch::empty({});
    auto shvar_avg = shvar + 0.5 * (Defor + Desec);
    auto stvar_avg = stfor.update_stvar(shvar_avg, dt_deps);
    if (shvar_avg.is_lowstress()) {
      if (dt < MIN_DT) {
        shvar = shfor;
        break;
      } else {
        dt *= 0.5;
        continue;
      }
    }
    const auto rtol = calc_residual(Defor, Desec, shvar_avg, err, epsilon);
    auto beta = BETA_COEFF * sqrt(SSTOL / rtol);
    if (rtol <= FTOLR) {
      // drift
      auto dlamda = ops::drift_shareVar(shvar_avg, stvar_avg, noel, npt, err, epsilon, 8);

      if (dlamda) {
        shvar = std::move(shvar_avg);
        stvar = std::move(stvar_avg);

        dempx.add_(dempx1, 0.5 * dt);
        dempx.add_(dempx2, 0.5 * dt);
        t += dt;
        if (abs(DATA_ONE - t) <= epsilon) {
          // 跳出循环
          break;
        }
        if (err && *err != ErrorCode::Success)
          return torch::empty({});
        dt = nfail ? std::min({beta * dt, dt, DATA_ONE - t})
                   : std::min({beta * dt, DT_GROWTH * dt, DATA_ONE - t});
        nfail = false;
      } else {
        return torch::empty({});
      }
      //
    } else {
      nfail = true;
      // update fail
      dt = std::max({beta * dt, MIN_DT_ON_FAIL, DT_SHRINK_FACTOR * dt});
    }
  }
  return dempx;
}
auto math::drift_shareVar(ShareVar &shvar, const core::StateVar &stvar, int noel, int npt,
                          ErrorCode *err, utils::data_t epsilon, utils::index_t iter_max) -> bool {
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
      return true;
    // 沿流动方向漂移
    auto shvar_flow = math::drift_along_flow_direction(shvar, stvar, err, epsilon);
    if (err && *err != ErrorCode::Success)
      return false;
    const auto df_flow = ops::calc_yield(shvar_flow, err);
    if (err && *err != ErrorCode::Success)
      return false;

    // 选择更优方向
    if (std::abs(df_flow) < std::abs(df_cur)) {
      shvar = std::move(shvar_flow);
    } else {
      shvar = math::drift_along_radial_direction(shvar, err, epsilon);
      if (err && *err != ErrorCode::Success)
        return false;
    }
  }
  if (err) {
    *err = ErrorCode::IterError;
  }
  stvar.set_pnewdt(0.5);
  return false;
}
/**
 * @brief
 * @param  shvars
 * @param  stvars
 * @param  epsilon
 *
 * @return Share_Type
 **/
auto math::drift_along_flow_direction(const ShareVar &shvar, const StateVar &stvar, ErrorCode *err,
                                      utils::data_t epsilon) -> core::ShareVar {
  auto dftol = calc_yield(shvar, err);

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
  return Rshvar + shvar;
}

auto math::drift_along_radial_direction(const ShareVar &shvar, ErrorCode *err,
                                        utils::data_t epsilon) -> core::ShareVar {
  auto dftol = ops::calc_yield(shvar);
  auto pfsig = ops::pfpsigma(shvar);
#ifdef DEBUG_SPAN_ENABLED
  auto pfsig_view = std::span<data_t>(pfsig.data_ptr<data_t>(), pfsig.numel());
#endif
  auto denominator = norm(pfsig, 2).item<data_t>();
  auto dlamda = core::safe_divide(dftol, denominator, err, epsilon);
  auto state = shvar.get_state();
  auto dsigma = StressTensor(-dlamda * pfsig, state);
  return shvar.create_shvar_with_dstress(dsigma);
}
auto math::Consistent_stiffness_matrix(const ShareVar &shvar_aft, const StateVar &stvar_aft,
                                       utils::data_t lamda, torch::Tensor stiff_for, ErrorCode *err,
                                       utils::data_t epsilon) -> Tensor {
  const auto voidr = stvar_aft.get_voidr();
  auto pfsig = ops::pfpsigma(shvar_aft, err);
  auto [pgsigma1_aft, pgsigma2_aft] = core::pgpsigma(shvar_aft, voidr);
  auto delta = torch::eye(3);
  auto delta_ik = delta.view({3, 1, 3, 1});
  auto delta_jl = delta.view({1, 3, 1, 3});
  auto delta_ijkl = delta_ik * delta_jl;
  auto sitff_pgsig2 = tensordot(stiff_for, pgsigma2_aft, {2, 3}, {0, 1});
  auto a = delta_ijkl + lamda * sitff_pgsig2;
  auto E_star = aten::linalg_tensorsolve(a, stiff_for);
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif
} // namespace umat
