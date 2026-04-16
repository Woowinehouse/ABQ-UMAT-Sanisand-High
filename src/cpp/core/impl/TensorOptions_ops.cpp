#include "TensorOptions_ops.h"
#include "ATen/ops/allclose.h"
#include "ATen/ops/clamp.h"
#include "ATen/ops/exp.h"
#include "ATen/ops/isinf.h"
#include "ATen/ops/isnan.h"
#include "ATen/ops/linalg_eigh.h"
#include "ATen/ops/log.h"
#include "ATen/ops/matmul.h"
#include "ATen/ops/max.h"
#include "ATen/ops/mm.h"
#include "ATen/ops/norm.h"
#include "ATen/ops/pow.h"
#include "ATen/ops/sqrt.h"
#include "ATen/ops/stack.h"
#include "ATen/ops/sum.h"
#include "ATen/ops/tensor.h"
#include "ATen/ops/trace.h"
#include "ATen/ops/where.h"
#include "ShareVarImpl.h"
#include "StateVarImpl.h"
#include "c10/util/Exception.h"
#include "core/ShareVarOptions.h"
#include "core/auxiliary.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/headeronly/util/Exception.h"
#include "torch/optim/optimizer.h"
#include "utils/TypeMap.h"
#include "utils/Visit.hpp"
#include "utils/base_config.h"
#include "utils/concept.h"
#include "utils/config.h"
#include "utils/export.h"
#include "utils/material_config.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <optional>

#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#ifdef DEBUG_SPAN_ENABLED
#include <span>
#endif

namespace umat::core::impl {
using namespace torch;
using namespace utils;
using namespace std;
// #ifdef __clang__
// #pragma clang diagnostic push
// #pragma clang diagnostic ignored "-Winconsistent-dllimport"
// #endif
auto Calc_shvar_norm_::call(const ShareVarImpl &shvars, ErrorCode *err) -> Tensor {
#ifdef STRICT_CHECK_ENABLED
  if (shvars.is_nan()) {
    if (err) {
      *err = ErrorCode::NanError;
      return torch::empty({});
    } else {
      STD_TORCH_CHECK(false, "Calc_shvar_norm_ input variable has exist nan");
    }
  }
  if (shvars.is_inf()) {
    if (err) {
      *err = ErrorCode::InfError;
      return torch::empty({});
    } else {
      STD_TORCH_CHECK(false, "Calc_shvar_norm_ input variable has exist inf");
    }
  }
#endif
  const auto &stress = shvars.get_stress();
  const auto &alpha = shvars.get_alpha();
  const auto &p0 = shvars.get_p0();

  auto stress_norm = norm(*stress, 2);
  auto alpha_norm = norm(*alpha, 2);
  auto p0_norm = norm(*p0, 2);
#ifdef DEBUG_SPAN_ENABLED
  auto stressNorm_view = std::span<data_t>(stress_norm.data_ptr<data_t>(), stress_norm.numel());
  auto alphaNorm_view = std::span<data_t>(alpha_norm.data_ptr<data_t>(), alpha_norm.numel());
  auto p0Norm_view = std::span<data_t>(p0_norm.data_ptr<data_t>(), p0_norm.numel());
#endif
#ifdef STRICT_CHECK_ENABLED
  auto checkerr = core::has_any_nan_inf(stress_norm, alpha_norm, p0_norm);

  if (checkerr != ErrorCode::Success) {
    if (err) {
      *err = checkerr;
      return torch::empty({});
    } else {
      STD_TORCH_CHECK(false, "The calculation result contains NaN/Inf.");
    }
  }
#endif
  return torch::stack({stress_norm, alpha_norm, p0_norm});
}
template <bool retain_map>
auto Calc_Cosine_angle_::call(const torch::Tensor &lhs, const torch::Tensor &rhs, ErrorCode *err,
                              utils::data_t epsilon)
    -> std::conditional_t<retain_map, torch::Tensor, utils::data_t> {
  auto lhs_norm = norm(lhs, 2);
  auto rhs_norm = norm(rhs, 2);
  auto numerator = sum(lhs_norm * rhs_norm);
  auto angle = Safe_divide_::call(numerator, lhs_norm * rhs_norm, err, epsilon);
  if constexpr (retain_map) {
    return angle;
  } else {
    return angle.item<data_t>();
  }
}
template <Scalartype T, Scalartype U>
auto Safe_divide_::call(T numerator, U denominator, ErrorCode *err, utils::data_t epsilon)
    -> std::conditional_t<(std::is_same_v<std::decay_t<T>, utils::data_t> &&
                           std::is_same_v<std::decay_t<U>, utils::data_t>),
                          utils::data_t, torch::Tensor> {

#ifdef STRICT_CHECK_ENABLED
  if constexpr (std::is_same_v<std::decay_t<T>, torch::Tensor>) {
    auto numerator_view =
        std::span<utils::data_t>(numerator.template data_ptr<utils::data_t>(), numerator.numel());
  }
  if constexpr (std::is_same_v<std::decay_t<U>, torch::Tensor>) {
    auto denominator_view = std::span<utils::data_t>(denominator.template data_ptr<utils::data_t>(),
                                                     denominator.numel());
  }
#endif
  if constexpr (std::is_same_v<std::decay_t<U>, utils::data_t>) {
    utils::data_t sign = denominator >= 0.0 ? 1.0 : -1.0;
    auto safe_deom = abs(denominator) < epsilon ? sign * epsilon : denominator;
    return numerator / safe_deom;
  } else {
    torch::Tensor sign = where(denominator >= 0.0, 1.0, -1.0);
    auto safe_deom = where(abs(denominator) < epsilon, sign * epsilon, denominator);
    return numerator / safe_deom;
  }
}
auto Is_isotropic_::call(const torch::Tensor &self) -> bool {

  auto [values, vectors] = torch::linalg_eigh(self, "L");
  auto diff12 = torch::allclose(values[0], values[1]);
  auto diff23 = torch::allclose(values[1], values[2]);
  return diff12 && diff23;
}
auto Is_nan_inf_::call(const torch::Tensor &tensor) -> bool {
  return torch::isnan(tensor).any().item<bool>() || torch::isinf(tensor).any().item<bool>();
}
/**
 * @brief
 * @tparam retain_map
 * @param  self
 * @param  err
 *
 * @return Scalar_type<retain_map>
 **/
template <bool retain_map>
auto Pressure_::call(const torch::Tensor &self, ErrorCode *err) -> Scalar_type<retain_map> {
#ifdef DEBUG_SPAN_ENABLED
  auto self_view = std::span<data_t>(self.data_ptr<data_t>(), self.numel());
#endif
  // #ifdef STRICT_CHECK_ENABLED
  //   auto inputerr = has_any_nan_inf(self);
  //   if (inputerr != ErrorCode::Success) {
  //     std::ofstream msg_file(Initialize::get_msgfile_path(), ios::app);
  //     msg_file << std::fixed << std::setprecision(6);
  //     // 写入日志标题
  //     msg_file << "\n==============================================================" <<
  //     std::endl; msg_file << "UMAT Pressure Calculation Log" << std::endl; msg_file <<
  //     "==============================================================" << std::endl;
  //   }
  // #endif
  auto pressure = self.trace() / 3.0;
#ifdef DEBUG_SPAN_ENABLED

  auto pressure_view = std::span<data_t>(pressure.data_ptr<data_t>(), pressure.numel());
#endif
#ifdef STRICT_CHECK_ENABLED
  auto checkerr = has_any_nan_inf(pressure);
  if (checkerr != ErrorCode::Success) {
    if (err) {
      *err = checkerr;
    } else {
      TORCH_WARN(false, "Pressure_::call input variable has exist nan or inf");
    }
  }
#endif
  if constexpr (retain_map) {
    return pressure;
  } else {
    return pressure.item<data_t>();
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
template <bool retain_map>
auto Deviatoric_::call(const torch::Tensor &self, ErrorCode *err) -> torch::Tensor {
  const auto delta = torch::eye(3);
  auto s_dev = [&]() -> Tensor {
    if constexpr (retain_map) {
      auto pressure = Pressure_::call<true>(self, err);
      return self - pressure * delta;
    } else {
      auto pressure = Pressure_::call<false>(self, err);
      return self - pressure * delta;
    }
  }();
#ifdef DEBUG_SPAN_ENABLED
  auto self_view = std::span<data_t>(self.data_ptr<data_t>(), self.numel());
  auto delta_view = std::span<data_t>(delta.data_ptr<data_t>(), delta.numel());
#endif
#ifdef STRICT_CHECK_ENABLED
  auto checkerr = has_any_nan_inf(self, s_dev);
  if (checkerr != ErrorCode::Success) {
    if (err) {
      *err = checkerr;
      return torch::empty({}); // true → 返回张量
    } else {
      STD_TORCH_CHECK(false, "Deviatoric_::call input variable has exist nan or inf");
    }
  }
#endif
  return s_dev;
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
template <bool retain_map>
auto StressRatio_::call(const torch::Tensor &self, ErrorCode *err, utils::data_t epsilon)
    -> torch::Tensor {
  Tensor ratio = [&]() -> Tensor {
    if constexpr (retain_map) {
      auto pressure = Pressure_::call<true>(self, err);
      auto s_dev = Deviatoric_::call<true>(self, err);
      return Safe_divide_::call(s_dev, pressure, err, epsilon);
    } else {
      auto pressure = Pressure_::call<false>(self, err);
      auto s_dev = Deviatoric_::call<false>(self, err);
      return Safe_divide_::call(s_dev, torch::tensor(pressure), err, epsilon);
    }
  }();
#ifdef DEBUG_SPAN_ENABLED
  auto self_view = std::span<data_t>(self.data_ptr<data_t>(), self.numel());
  auto ratio_view = std::span<data_t>(ratio.data_ptr<data_t>(), ratio.numel());
#endif
  return ratio;
}
/**
 * @brief
 * @tparam retain_map
 * @param  self
 * @param  err
 * @param  epsilon
 *
 * @return Scalar_type<retain_map>
 **/
template <bool retain_map>
auto Calc_Rm_::call(const torch::Tensor &self, ErrorCode *err, utils::data_t epsilon,
                    torch::Tensor ratio) -> Scalar_type<retain_map> {
  torch::Tensor ratio_t = [&]() -> torch::Tensor {
    if constexpr (retain_map) {
      return StressRatio_::call<true>(self, err, epsilon);
    } else {
      if (ratio.defined()) {
        return ratio;
      } else {
        return StressRatio_::call<false>(self, err, epsilon);
      }
    }
  }();
  auto ratio_norm = norm(ratio_t);
  auto Rm = utils::RAD32 * sqrt(ratio_norm);
#ifdef DEBUG_SPAN_ENABLED
  auto self_view = std::span<data_t>(self.data_ptr<data_t>(), self.numel());
  auto Rm_view = std::span<data_t>(Rm.data_ptr<data_t>(), Rm.numel());
#endif
#ifdef STRICT_CHECK_ENABLED
  auto checkerr = has_any_nan_inf(self, Rm);
  if (checkerr != ErrorCode::Success) {
    if (err) {
      *err = checkerr;
    } else {
      TORCH_WARN(false, "Calc_Rm_::call input variable has exist nan or inf");
    }
  }
#endif
  if constexpr (retain_map) {
    return Rm;
  } else {
    return Rm.item<data_t>();
  }
}
/**
 * @brief
 * @tparam retain_map
 * @param  shvarImpl
 * @param  err
 * @param  epsilon
 *
 * @return torch::Tensor
 **/
template <bool retain_map>
auto LoadingDirection_::call(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                             utils::data_t epsilon, torch::Tensor ratio) -> torch::Tensor {
  const auto &alpha = shvarImpl.get_alpha_tensor();
  const auto &stress = shvarImpl.get_stress();
  torch::Tensor ratio_t = [&]() -> torch::Tensor {
    if constexpr (retain_map) {
      return StressRatio_::call<true>(*stress, err, epsilon);
    } else {
      if (ratio.defined()) {
        return ratio;
      } else {
        return StressRatio_::call<false>(*stress, err, epsilon);
      }
    }
  }();

  auto temp = ratio_t - alpha;
  auto temp_norm = torch::norm(temp, 2);
  auto norm = Safe_divide_::call(temp, temp_norm, err, epsilon);
#ifdef DEBUG_SPAN_ENABLED
  auto norm_view = std::span<data_t>(norm.data_ptr<data_t>(), norm.numel());
#endif
#ifdef STRICT_CHECK_ENABLED
  auto checkerr = has_nan_inf(norm);
  if (checkerr != ErrorCode::Success) {
    if (err) {
      *err = checkerr;
    } else {
      STD_TORCH_CHECK(false, "Deviatoric_::call input variable has exist nan or inf");
    }
  }
#endif
  return norm;
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
template <bool retain_map>
auto Cos3theta_::call(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                      utils::data_t epsilon, torch::Tensor norm) -> Scalar_type<retain_map> {
  Tensor cos3t = [&]() -> Tensor {
    if constexpr (retain_map) {
      // 保留计算图，不能使用传入norm
      auto norm_map = LoadingDirection_::call<true>(shvarImpl, err, epsilon);
      auto cos3t = utils::RAD6 * trace(mm(mm(norm_map, norm_map), norm_map));
      auto cos3t_clamp = torch::clamp(cos3t, static_cast<data_t>(-1.0), static_cast<data_t>(1.0));
      return cos3t_clamp;
    } else {
      if (!norm.defined()) {
        norm = LoadingDirection_::call<false>(shvarImpl, err, epsilon);
      }
      auto cos3t = utils::RAD6 * trace(mm(mm(norm, norm), norm));
      auto cos3t_clamp = torch::clamp(cos3t, static_cast<data_t>(-1.0), static_cast<data_t>(1.0));
      return cos3t_clamp;
    }
  }();
#ifdef DEBUG_SPAN_ENABLED
  auto cos3t_view = std::span<data_t>(cos3t.data_ptr<data_t>(), cos3t.numel());
#endif
  if constexpr (retain_map) {
    return cos3t;
  } else {
    return cos3t.item<data_t>();
  }
}
/**
 * @brief
 * @tparam retain_map
 * @tparam is_grad
 * @tparam Dtype
 * @param  shvarImpl
 * @param  err
 * @param  epsilon
 * @param  cos3t
 *
 * @return std::conditional_t<is_grad, std::pair<Scalar_type<retain_map>, torch::Tensor>,
 * Scalar_type<retain_map>>
 **/
template <bool retain_map, bool is_grad>
auto Calc_gtheta_::call(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                        utils::data_t epsilon, Scalar_Type cos3t)
    -> std::conditional_t<is_grad, std::pair<Scalar_type<retain_map>, torch::Tensor>,
                          Scalar_type<retain_map>> {
  using namespace utils;
  auto inv_w = 1.0 / mat::w;
  auto cw = pow(mat::c, inv_w);
  /**
   * @brief retain_map = true
   *
   **/
  if constexpr (retain_map) {
    auto norm = LoadingDirection_::call<false>(shvarImpl, err, epsilon);
    auto cos3t_t = Cos3theta_::call<true>(shvarImpl, err, epsilon, norm);
    if constexpr (is_grad) {
      cos3t_t.detach_();
      cos3t_t.requires_grad_(true);
    }
    auto base = 0.5 * ((1.0 + cw) + (1.0 - cw) * cos3t_t);
    auto gtheta = pow(base, mat::w);
    if constexpr (is_grad) {
      auto pgcos3t = [&]() -> Tensor {
        if (cos3t_t.grad().defined()) {
          cos3t_t.grad().zero_();
        }
        gtheta.backward({}, true);
        return cos3t_t.grad().clone();
      }();
      return make_pair(gtheta, pgcos3t);
    } else {
      return gtheta;
    }
  }
  /**
   * @brief retain_map =false
   *
   **/
  else {
    if constexpr (is_grad) { // retain_map =false, is_grad = true
      Tensor cos3t_t = std::visit(
          overloaded{[](const Tensor &cos3t) -> Tensor { return cos3t; },
                     [](data_t cos3t_d) -> Tensor { return tensor({cos3t_d}); },
                     [&](std::monostate) -> Tensor {
                       auto norm = LoadingDirection_::call<true>(shvarImpl, err, epsilon);
                       auto cos3t_t = Cos3theta_::call<true>(shvarImpl, err, epsilon, norm);
                       return cos3t_t;
                     }},
          cos3t);
      cos3t_t.detach_();
      cos3t_t.requires_grad_(true);
      auto base = 0.5 * ((1.0 + cw) + (1.0 - cw) * cos3t_t);
      auto gtheta = pow(base, mat::w);
      auto pgcos3t = [&cos3t_t, &gtheta]() -> Tensor {
        if (cos3t_t.grad().defined()) {
          cos3t_t.grad().zero_();
        }
        gtheta.backward();
        return cos3t_t.grad().clone();
      }();
      return make_pair(gtheta.item<data_t>(), pgcos3t);
    } else { // retain_map =false, is_grad = false
      data_t cos3t_d = std::visit(
          overloaded{[](const Tensor &cos3t_t) -> data_t { return cos3t_t.item<data_t>(); },
                     [](data_t cos3t_d) -> data_t { return cos3t_d; },
                     [&](std::monostate) -> data_t {
                       auto norm = LoadingDirection_::call<false>(shvarImpl, err, epsilon);
                       auto cos3t_d = Cos3theta_::call<false>(shvarImpl, err, epsilon, norm);
                       return cos3t_d;
                     }},
          cos3t);
      auto base = 0.5 * ((1.0 + cw) + (1.0 - cw) * cos3t_d);
      auto gtheta = pow(base, mat::w);
      return gtheta;
    }
  }
}

/**
 * @brief
 * @tparam retain_map
 * @tparam Dtype
 * @param  shvarImpl
 * @param  err
 * @param  epsilon
 * @param  gtheta
 *
 * @return Scalar_type<retain_map>
 **/
template <bool retain_map>
auto Calc_lamda_::call(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                       utils::data_t epsilon, Scalar_Type gtheta) -> Scalar_type<retain_map> {
  using namespace utils;
  auto alpha = shvarImpl.get_alpha_tensor();
  auto dot_alpha = sum(alpha * alpha);
  /**
   * @brief retain_map =ture
   *
   **/
  if constexpr (retain_map) {
    auto gtheta_t = Calc_gtheta_::call<true>(shvarImpl, err, epsilon);
    auto deom = (gtheta_t * mat::alphac) * (gtheta_t * mat::alphac);
    auto numerator = 1.5 * dot_alpha * (mat::lamdacs - mat::lamdar);
    auto temp = Safe_divide_::call(numerator, deom, err, epsilon);
    return mat::lamdar + temp;
  }
  /**
   * @brief retain_map = false
   *
   **/
  else {
    data_t gtheta_d = std::visit(
        overloaded{[](const Tensor &gtheta_t) -> data_t { return gtheta_t.item<data_t>(); },
                   [](data_t gtheta_d) -> data_t { return gtheta_d; },
                   [&](std::monostate) -> data_t {
                     auto gtheta_d = Calc_gtheta_::call<false>(shvarImpl, err, epsilon);
                     return gtheta_d;
                   }},
        gtheta);
    auto gtheta_alphac = (gtheta_d * mat::alphac);
    auto deom = gtheta_alphac * gtheta_alphac;
    auto numerator = 1.5 * dot_alpha.item<data_t>() * (mat::lamdacs - mat::lamdar);
    auto temp = Safe_divide_::call(numerator, deom, err, epsilon);
    return mat::lamdar + temp;
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
template <bool retain_map>
auto Calc_GV_::call(const torch::Tensor &self, ErrorCode *err, Scalar_Type pressure)
    -> Scalar_type<retain_map> {
  using namespace utils;
  /**
   * @brief retain_map =true
   *
   **/
  if constexpr (retain_map) {
    auto p_t = Pressure_::call<true>(self, err);
    auto nu = mat::nu_min + (mat::nu_max - mat::nu_min) * exp(-mat::nu_v * (p_t / PA));
    auto Gv = 3.0 * (1.0 - 2.0 * nu) / 2.0 / (1.0 + nu);
    return Gv;
  } else {
    data_t p_d =
        std::visit(overloaded{[](const Tensor &p_t) -> data_t { return p_t.item<data_t>(); },
                              [](data_t p_d) -> data_t { return p_d; },
                              [&](std::monostate) -> data_t {
                                auto gtheta_d = Pressure_::call<false>(self, err);
                                return gtheta_d;
                              }},
                   pressure);
    auto nu = mat::nu_min + (mat::nu_max - mat::nu_min) * exp(-mat::nu_v * (p_d / PA));
    auto Gv = 3.0 * (1.0 - 2.0 * nu) / 2.0 / (1.0 + nu);
    return Gv;
  }
}
/**
 * @brief
 * @tparam retain_map
 * @param  shvarImpl
 * @param  voidr
 * @param  err
 * @param  epsilon
 *
 * @return std::conditional_t<retain_map, std::pair<Tensor, Tensor>, std::pair<data_t, data_t>>
 **/
template <bool retain_map>
auto Calc_shear_bulk_::call(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                            ErrorCode *err, utils::data_t epsilon)
    -> std::conditional_t<retain_map, std::pair<Tensor, Tensor>, std::pair<data_t, data_t>> {
  using namespace utils;
  const auto &stress = shvarImpl.get_stress_tensor();
  if constexpr (retain_map) {
    auto pressure = Pressure_::call<true>(stress, err);
    auto lamda_a = Calc_lamda_::call<true>(shvarImpl, err, epsilon);
    auto Gv = Calc_GV_::call<true>(stress, err);
    auto temp = lamda_a * pow((pressure / PA), mat::ksi);
    auto BH_1 = (voidr - mat::voidrl) * mat::k * mat::ksi * temp;
    auto BH_2 = mat::y * (mat::voidref - mat::voidrl) * exp(-temp) *
                (pow(mat::pe, mat::y) * pressure) / pow((pressure + mat::ch * PA), mat::y + 1);
    auto bulk = (1.0 + voidr) * pressure / (BH_1 + BH_2);
    auto shear = Gv * bulk;
    return std::make_pair(shear, bulk);
  } else {
    auto pressure = Pressure_::call(stress, err);
    auto lamda_a = Calc_lamda_::call(shvarImpl, err, epsilon);
    auto Gv = Calc_GV_::call(stress, err);
    auto temp = lamda_a * pow((pressure / PA), mat::ksi);
    auto BH_1 = (voidr - mat::voidrl) * mat::k * mat::ksi * temp;
    auto BH_2 = mat::y * (mat::voidref - mat::voidrl) * exp(-temp) *
                (pow(mat::pe, mat::y) * pressure) / pow((pressure + mat::ch * PA), mat::y + 1);
    auto bulk = (1.0 + voidr) * pressure / (BH_1 + BH_2);
    auto shear = Gv * bulk;
    return std::make_pair(shear, bulk);
  }
}
/**
 * @brief
 * @tparam retain_map
 * @param  shvarImpl
 * @param  voidr
 * @param  err
 * @param  epsilon
 *
 * @return torch::Tensor
 **/
template <bool retain_map>
auto Stiffness_::call(const impl::ShareVarImpl &shvarImpl, utils::data_t voidr, ErrorCode *err,
                      utils::data_t epsilon) -> torch::Tensor {
  const auto delta = torch::eye(3);
  const auto vol_delta = delta.view({3, 3, 1, 1}) * delta.view({1, 1, 3, 3});
  const auto sym_delta = 0.5 * (delta.view({3, 1, 1, 3}) * delta.view({1, 3, 3, 1}) +
                                delta.view({3, 1, 3, 1}) * delta.view({1, 3, 1, 3}));
  if constexpr (retain_map) {
    auto [shear, bulk] = Calc_shear_bulk_::call<true>(shvarImpl, voidr, err, epsilon);
    auto lamda = bulk - 2.0 / 3.0 * shear;
    auto stiff = lamda * vol_delta + 2.0 * shear * sym_delta;
    return stiff;
  } else {
    auto [shear, bulk] = Calc_shear_bulk_::call<false>(shvarImpl, voidr, err, epsilon);
    auto lamda = bulk - 2.0 / 3.0 * shear;
    auto stiff = lamda * vol_delta + 2.0 * shear * sym_delta;
    return stiff;
  }
}
/**
 * @brief
 * @tparam retain_map
 * @param  shvarImpl
 * @param  err
 *
 * @return Scalar_type<retain_map>
 **/
template <bool retain_map>
auto Calc_yield_::call(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err)
    -> Scalar_type<retain_map> {
  using namespace utils;
  const auto &alpha = shvarImpl.get_alpha_tensor();
  //
  const auto &stress = shvarImpl.get_stress_tensor();
  const auto &p0 = shvarImpl.get_p0_tensor();
  if constexpr (retain_map) {
    auto p_t = Pressure_::call<true>(stress, err);
    auto s_dev = Deviatoric_::call<true>(stress, err);
    auto temp = s_dev - p_t * alpha;
    auto dot_product = sum(temp * temp);
    auto df =
        1.5 * dot_product - (mat::fm * p_t) * (mat::fm * p_t) * (1.0 - pow(p_t / p0, mat::fn));
    return df;
  } else {
    auto p_d = Pressure_::call<false>(stress, err);
    auto s_dev = Deviatoric_::call<false>(stress, err);
    auto temp = s_dev - p_d * alpha;
    auto dot_product = sum(temp * temp);
    auto df =
        1.5 * dot_product - (mat::fm * p_d) * (mat::fm * p_d) * (1.0 - pow(p_d / p0, mat::fn));
    return df.item<data_t>();
  }
}
auto Pfpsigma_::call(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err) -> torch::Tensor {
  // 2. 关键：确保 obj 是干净的叶子节点
  auto obj = shvarImpl.clone();
  obj.detach_();
  auto &stress = obj.get_stress();
  stress->requires_grad_(true);
  auto f_func = Calc_yield_::call<true>(obj, err);
  if (stress->grad().defined()) {
    stress->grad().zero_();
  }
  f_func.backward();
  auto pfsig = stress->grad().clone();
#ifdef DEBUG_SPAN_ENABLED
  auto pfsig_view = std::span<data_t>(pfsig.data_ptr<data_t>(), pfsig.numel());
#endif
  return pfsig;
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
template <bool retain_map>
auto Calc_psim_::call(const torch::Tensor &self, utils::data_t voidr, ErrorCode *err,
                      Scalar_Type pressure) -> Scalar_type<retain_map> {
  using namespace utils;
  /**
   * @brief retain_map = true
   *
   **/
  if constexpr (retain_map) {
    auto p_t = Pressure_::call<true>(self, err);
    auto voidc =
        (mat::voidref - mat::voidrl) * exp(-mat::lamdacs * pow(p_t / PA, mat::ksi)) + mat::voidrl;
    auto psim = voidr - voidc;
    return psim;
  }
  /**
   * @brief retain_map = false
   *
   **/
  else {
    data_t p_d =
        std::visit(overloaded{[](const Tensor &p_t) -> data_t { return p_t.item<data_t>(); },
                              [](data_t p_d) -> data_t { return p_d; },
                              [&](std::monostate) -> data_t {
                                auto gtheta_d = Pressure_::call<false>(self, err);
                                return gtheta_d;
                              }},
                   pressure);
    auto voidc =
        (mat::voidref - mat::voidrl) * exp(-mat::lamdacs * pow(p_d / PA, mat::ksi)) + mat::voidrl;
    auto psim = voidr - voidc;
    return psim;
  }
}

/**
 * @brief
 * @tparam retain_map
 * @param  shvarImpl
 * @param  voidr
 * @param  err
 * @param  epsilon
 *
 * @return Scalar_type<retain_map>
 **/
template <bool retain_map>
auto Calc_psim_alpha_::call(const impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                            ErrorCode *err, utils::data_t epsilon, Scalar_Type pressure,
                            Scalar_Type lamda_alpha) -> Scalar_type<retain_map> {
  using namespace utils;
  const auto &stress = shvarImpl.get_stress_tensor();
  /**
   * @brief retain_map = true
   *
   **/
  if constexpr (retain_map) {
    auto p_t = Pressure_::call<true>(stress, err);
    auto lamda_alpha_t = Calc_lamda_::call<true>(shvarImpl, err, epsilon);
    auto void_alpha = (mat::voidref - mat::voidrl) * exp(-lamda_alpha_t * pow(p_t / PA, mat::ksi)) *
                          (1.0 + pow((mat::pe / (p_t + mat::ch * PA)), mat::y)) +
                      mat::voidrl;
    auto psim_alpha = voidr - void_alpha;
    return psim_alpha;
  }
  /**
   * @brief retain_map = false
   *
   **/
  else {
    data_t p_d =
        std::visit(overloaded{[](const Tensor &p_t) -> data_t { return p_t.item<data_t>(); },
                              [](data_t p_d) -> data_t { return p_d; },
                              [&stress, &err](std::monostate) -> data_t {
                                auto gtheta_d = Pressure_::call<false>(stress, err);
                                return gtheta_d;
                              }},
                   pressure);
    data_t lamda_alpha_d = std::visit(
        overloaded{[](const Tensor &lamda_a_t) -> data_t { return lamda_a_t.item<data_t>(); },
                   [](data_t lamda_a_d) -> data_t { return lamda_a_d; },
                   [&shvarImpl, &err, epsilon](std::monostate) -> data_t {
                     auto lamda_a_d = Calc_lamda_::call<false>(shvarImpl, err, epsilon);
                     return lamda_a_d;
                   }},
        lamda_alpha);
    auto void_alpha = (mat::voidref - mat::voidrl) * exp(-lamda_alpha_d * pow(p_d / PA, mat::ksi)) *
                          (1.0 + pow((mat::pe / (p_d + mat::ch * PA)), mat::y)) +
                      mat::voidrl;
    auto psim_alpha = voidr - void_alpha;
    return psim_alpha;
  }
}
/**
 * @brief
 * @tparam retain_map
 * @param  shvarImpl
 * @param  voidr
 * @param  options
 * @param  err
 * @param  epsilon
 *
 * @return std::conditional_t<retain_map, std::pair<torch::Tensor, torch::Tensor>,
 * std::pair<utils::data_t, utils::data_t>>
 **/
template <bool retain_map>
auto Dilatancy_::call(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                      const core::PlasticOptions &options, ErrorCode *err, utils::data_t epsilon)
    -> std::conditional_t<retain_map, std::pair<torch::Tensor, torch::Tensor>,
                          std::pair<utils::data_t, utils::data_t>> {
  using namespace utils;

  const auto &stress = shvarImpl.get_stress_tensor();
  const auto &alpha = shvarImpl.get_alpha_tensor();
  if constexpr (retain_map) {
    auto p_t = Pressure_::call<true>(stress, err);
    auto norm = LoadingDirection_::call<true>(shvarImpl, err, epsilon);
    auto psim = Calc_psim_::call<true>(stress, voidr, err, p_t);
    auto gtheta = Calc_gtheta_::call<true>(shvarImpl, err, epsilon);
    auto psim_alpha = Calc_psim_alpha_::call<true>(shvarImpl, voidr, err, epsilon);
    auto C_alpha = 2.0 / (1.0 + exp(-mat::beta * psim_alpha));
    auto alpha_d = RAD23 * mat::alphac * gtheta * exp(mat::nd * psim) * norm;
    auto temp = alpha_d - alpha;
    auto dpla =
        RAD32 * mat::ad / gtheta * exp(mat::pd * C_alpha * sqrt(p_t / PA)) * sum(temp * norm);
    auto dot_alpha_d = sum(alpha_d * alpha_d);
    auto dot_alpha = sum(alpha * alpha);
    auto cond = max(1.0 - C_alpha, torch::tensor({0.0}));
    auto X_alpha = RAD32 * (sqrt(dot_alpha_d) - sqrt(dot_alpha)) * cond * cond + C_alpha;
#ifdef DEBUG_SPAN_ENABLED
    auto p_t_view = std::span<data_t>(p_t.data_ptr<data_t>(), p_t.numel());
    auto norm_view = std::span<data_t>(norm.data_ptr<data_t>(), norm.numel());
    auto gtheta_view = std::span<data_t>(gtheta.data_ptr<data_t>(), gtheta.numel());
    auto psim_view = std::span<data_t>(psim.data_ptr<data_t>(), psim.numel());
    auto psim_alpha_view = std::span<data_t>(psim_alpha.data_ptr<data_t>(), psim_alpha.numel());
    auto alpha_d_view = std::span<data_t>(alpha_d.data_ptr<data_t>(), alpha_d.numel());
    auto temp_view = std::span<data_t>(temp.data_ptr<data_t>(), temp.numel());
    auto dpla_view = std::span<data_t>(dpla.data_ptr<data_t>(), dpla.numel());
    auto dot_alpha_d_view = std::span<data_t>(dot_alpha_d.data_ptr<data_t>(), dot_alpha_d.numel());
    auto dot_alpha_view = std::span<data_t>(dot_alpha.data_ptr<data_t>(), dot_alpha.numel());
    auto cond_view = std::span<data_t>(cond.data_ptr<data_t>(), cond.numel());
    auto X_alpha_view = std::span<data_t>(X_alpha.data_ptr<data_t>(), X_alpha.numel());
#endif
    return std::make_pair(dpla, X_alpha);
  } else {
    auto p_d = Pressure_::call<false>(stress, err);
    auto norm = LoadingDirection_::call<false>(shvarImpl, err, epsilon);
#ifdef DEBUG_SPAN_ENABLED
    auto norm_view = std::span<data_t>(norm.data_ptr<data_t>(), norm.numel());
#endif
    auto psim = Calc_psim_::call<false>(stress, voidr, err, p_d);
    auto gtheta = Calc_gtheta_::call<false>(shvarImpl, err, epsilon);
    auto psim_alpha = Calc_psim_alpha_::call<false>(shvarImpl, voidr, err, epsilon);
    auto C_alpha = 2.0 / (1.0 + exp(-mat::beta * psim_alpha));
    auto alpha_d = RAD23 * mat::alphac * gtheta * exp(mat::nd * psim) * norm;
    auto temp = alpha_d - alpha;
#ifdef DEBUG_SPAN_ENABLED
    auto alpha_d_view = std::span<data_t>(alpha_d.data_ptr<data_t>(), alpha_d.numel());
    auto temp_view = std::span<data_t>(temp.data_ptr<data_t>(), temp.numel());
#endif
    auto dpla = RAD32 * mat::ad / gtheta * exp(mat::pd * C_alpha * sqrt(p_d / PA)) *
                sum(temp * norm).item<data_t>();
    auto dot_alpha_d = sum(alpha_d * alpha_d).item<data_t>();
    auto dot_alpha = sum(alpha * alpha).item<data_t>();
    auto cond = std::max((1.0 - C_alpha), 0.0);
    auto X_alpha = RAD32 * (sqrt(dot_alpha_d) - sqrt(dot_alpha)) * cond * cond + C_alpha;

    return std::make_pair(dpla, X_alpha);
  }
}

auto Pgpsigma_::call(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                     const core::PlasticOptions &options, ErrorCode *err, utils::data_t epsilon)
    -> torch::Tensor {
  using namespace utils;
  auto stress = shvarImpl.get_stress_tensor();
  auto alpha = shvarImpl.get_alpha_tensor();
  auto delta = torch::eye(3);
  auto norm = LoadingDirection_::call<true>(shvarImpl, err, epsilon);
  auto dot_norm = matmul(norm, norm);
  auto ratio = StressRatio_::call<true>(stress, err, epsilon);
  auto cos3t = Cos3theta_::call<true>(shvarImpl, err, epsilon);
  const auto &[gtheta, pgcos3t] = Calc_gtheta_::call<true, true>(shvarImpl, err, epsilon);
  const auto &[dpla, X_alpha] = Dilatancy_::call<true>(shvarImpl, voidr, options, err, epsilon);
  auto dgdth = pgcos3t / gtheta;
  auto B = 1.0 + 3.0 * cos3t * dgdth;
  auto C = 3.0 * RAD6 * dgdth;
  auto R_ = B * norm - C * (dot_norm - delta / 3.0);
  auto Rnorm = R_ / R_.norm(2);
  auto r_ef = sqrt(3.0 / 2.0 * sum((ratio - alpha) * (ratio - alpha)));
  auto exprf = exp(-mat::v * r_ef);
  auto Ep = RAD32 * Rnorm * r_ef + 3.0 / 2.0 * mat::x * ratio * exprf;
  auto Ev = dpla * r_ef + X_alpha * exprf;
  auto pgsig = Ep + Ev * delta / 3.0;
#ifdef DEBUG_SPAN_ENABLED
  auto ratio_view = std::span<data_t>(ratio.data_ptr<data_t>(), ratio.numel());
  auto norm_view = std::span<data_t>(norm.data_ptr<data_t>(), norm.numel());
  auto cos3t_view = std::span<data_t>(cos3t.data_ptr<data_t>(), cos3t.numel());
  auto gtheta_view = std::span<data_t>(gtheta.data_ptr<data_t>(), gtheta.numel());
  auto dot_norm_view = std::span<data_t>(dot_norm.data_ptr<data_t>(), dot_norm.numel());
  auto pgcos3t_view = std::span<data_t>(pgcos3t.data_ptr<data_t>(), pgcos3t.numel());
  auto dgdth_view = std::span<data_t>(dgdth.data_ptr<data_t>(), dgdth.numel());
  auto dpla_view = std::span<data_t>(dpla.data_ptr<data_t>(), dpla.numel());
  auto X_alpha_view = std::span<data_t>(X_alpha.data_ptr<data_t>(), X_alpha.numel());
  auto B_view = std::span<data_t>(B.data_ptr<data_t>(), B.numel());
  auto C_view = std::span<data_t>(C.data_ptr<data_t>(), C.numel());
  auto R_view = std::span<data_t>(R_.data_ptr<data_t>(), R_.numel());
  auto Rnorm_view = std::span<data_t>(Rnorm.data_ptr<data_t>(), Rnorm.numel());
  auto r_ef_view = std::span<data_t>(r_ef.data_ptr<data_t>(), r_ef.numel());
  auto exprf_view = std::span<data_t>(exprf.data_ptr<data_t>(), exprf.numel());
  auto Ep_view = std::span<data_t>(Ep.data_ptr<data_t>(), Ep.numel());
  auto Ev_view = std::span<data_t>(Ev.data_ptr<data_t>(), Ev.numel());
  auto pgsig_view = std::span<data_t>(pgsig.data_ptr<data_t>(), pgsig.numel());
#endif
  return pgsig;
}
/**
 * @brief
 * @tparam retain_map
 * @param  shvarImpl
 * @param  stvarsImpl
 * @param  options
 * @param  err
 * @param  epsilon
 *
 * @return std::tuple<torch::Tensor, torch::Tensor, Scalar_type<retain_map>>
 **/
template <bool retain_map>
auto Evolution_Dkp_::call(const core::impl::ShareVarImpl &shvarImpl,
                          const core::impl::StateVarImpl &stvarsImpl,
                          const core::PlasticOptions &options, ErrorCode *err,
                          utils::data_t epsilon)
    -> std::tuple<torch::Tensor, torch::Tensor, Scalar_type<retain_map>> {
  using namespace utils;
  auto obj = shvarImpl.clone();
  obj.detach_();
  const auto &stress = obj.get_stress_tensor();
  const auto &alpha = obj.get_alpha_tensor();
  const auto &p0 = obj.get_p0_tensor();
  //
  alpha.requires_grad_(true);
  p0.requires_grad_(true);
  auto f_func = Calc_yield_::call<true>(obj, err);
  if (alpha.grad().defined()) {
    alpha.grad().zero_();
  }
  if (p0.grad().defined()) {
    p0.grad().zero_();
  }
  f_func.backward();
  auto pfp0 = p0.grad();
  auto pfalpha = alpha.grad();
#ifdef DEBUG_SPAN_ENABLED
  auto pfp0_view = std::span<data_t>(pfp0.data_ptr<data_t>(), pfp0.numel());
  auto pfalpha_view = std::span<data_t>(pfalpha.data_ptr<data_t>(), pfalpha.numel());
#endif
  /**
   * @brief
   *
   **/
  auto voidr = stvarsImpl.get_voidr();
  const auto &alpha_ini = stvarsImpl.get_alphaIni_tensor();
  if constexpr (retain_map) {
    auto p_t = Pressure_::call<true>(stress);
    auto norm = LoadingDirection_::call<true>(shvarImpl, err, epsilon);
    auto ratio = StressRatio_::call<true>(stress, err, epsilon);
    auto cos3t = Cos3theta_::call<true>(shvarImpl, err, epsilon);
    auto gtheta = Calc_gtheta_::call<true>(shvarImpl, err, epsilon);
    auto lamda_alpha = Calc_lamda_::call<true>(shvarImpl, err, epsilon);
    auto Gv = Calc_GV_::call<true>(stress, err);
    auto psim = Calc_psim_::call<true>(stress, voidr, err, p_t);
    auto psim_alpha = Calc_psim_alpha_::call<true>(shvarImpl, voidr, err, epsilon);
    auto h_e = log((mat::voidref - mat::voidrl) / (voidr - mat::voidrl) *
                   (1.0 + pow(mat::pe / (p_t + mat::ch * PA), mat::y))) /
               lamda_alpha;
    auto h_ocr = 1.0 + mat::pr * (1.0 - p_t / p0) * pow(p_t / PA, 0.2);
    auto C_alpha = 2.0 / (1.0 + exp(-mat::beta * psim_alpha));
    auto b_0 = Gv * mat::h0 * exp(-mat::ps * C_alpha * sqrt(p_t / PA)) * h_ocr * h_e *
               pow(p_t / PA, -mat::ksi);
    auto r_ef = sqrt(3.0 / 2.0 * sum((ratio - alpha) * (ratio - alpha)));
    auto exprf = exp(-mat::v * r_ef);
    auto alpha_alpha_ini = alpha - alpha_ini;
    auto temp = sum(alpha_alpha_ini * norm);
    auto denom = temp * (1.0 - exprf) + exprf;
    auto h = Safe_divide_::call(b_0, denom, err, epsilon);
    auto alpha_b =
        RAD23 * mat::alphac * gtheta * exp(mat::nb * max(-psim, torch::tensor(0.0))) * norm;
    auto Ralpha = h * r_ef * (alpha_b - alpha);
    auto void_alpha = voidr - psim_alpha;
    auto Rp0_denom = C_alpha * (void_alpha - mat::voidrl) * lamda_alpha * (1.0 - mat::k) *
                     mat::ksi * pow(p0 / PA, mat::ksi);
    auto Rp0 = Safe_divide_::call((1.0 + void_alpha) * p0 * exprf, Rp0_denom);
    auto kp = -(sum(pfalpha * Ralpha) + pfp0 * Rp0);
    return std::make_tuple(Ralpha, Rp0, kp);
  } else {
    auto p_d = Pressure_::call<false>(stress);
    auto norm = LoadingDirection_::call<false>(shvarImpl, err, epsilon);
    auto ratio = StressRatio_::call<false>(stress, err, epsilon);
#ifdef STRICT_CHECK_ENABLED
    auto norm_view = std::span<data_t>(norm.data_ptr<data_t>(), norm.numel());
    auto ratio_view = std::span<data_t>(ratio.data_ptr<data_t>(), ratio.numel());
#endif
    auto cos3t = Cos3theta_::call<false>(shvarImpl, err, epsilon);
    auto gtheta = Calc_gtheta_::call<false>(shvarImpl, err, epsilon, cos3t);
    auto lamda_alpha = Calc_lamda_::call<false>(shvarImpl, err, epsilon, gtheta);
    auto Gv = Calc_GV_::call<false>(stress, err, p_d);
    auto psim = Calc_psim_::call<false>(stress, voidr, err, p_d);
    auto psim_alpha = Calc_psim_alpha_::call<false>(shvarImpl, voidr, err, epsilon);
    auto h_e = log((mat::voidref - mat::voidrl) / (voidr - mat::voidrl) *
                   (1.0 + pow(mat::pe / (p_d + mat::ch * PA), mat::y))) /
               lamda_alpha;
    auto h_ocr = 1.0 + mat::pr * (1.0 - p_d / p0.item<data_t>()) * pow(p_d / PA, 0.2);
    auto C_alpha = 2.0 / (1.0 + exp(-mat::beta * psim_alpha));
    auto b_0 = Gv * mat::h0 * exp(-mat::ps * C_alpha * sqrt(p_d / PA)) * h_ocr * h_e *
               pow(p_d / PA, -mat::ksi);
    auto r_ef = sqrt(3.0 / 2.0 * sum((ratio - alpha) * (ratio - alpha))).item<data_t>();
    auto exprf = exp(-mat::v * r_ef);
    auto alpha_alpha_ini = alpha - alpha_ini;
#ifdef DEBUG_SPAN_ENABLED
    auto alpha_alpha_ini_view =
        std::span<data_t>(alpha_alpha_ini.data_ptr<data_t>(), alpha_alpha_ini.numel());
#endif
    auto temp = sum(alpha_alpha_ini * norm).item<data_t>();
    auto denom = temp * (1.0 - exprf) + exprf;
    auto h = Safe_divide_::call(b_0, denom, err, epsilon);
    auto alpha_b = RAD23 * mat::alphac * gtheta * exp(mat::nb * std::max(-psim, 0.0)) * norm;
#ifdef DEBUG_SPAN_ENABLED
    auto alpha_b_view = std::span<data_t>(alpha_b.data_ptr<data_t>(), alpha_b.numel());
#endif
    auto Ralpha = h * r_ef * (alpha_b - alpha);
#ifdef DEBUG_SPAN_ENABLED
    auto Ralpha_view = std::span<data_t>(Ralpha.data_ptr<data_t>(), Ralpha.numel());
#endif
    auto void_alpha = voidr - psim_alpha;
    auto Rp0_denom = C_alpha * (void_alpha - mat::voidrl) * lamda_alpha * (1.0 - mat::k) *
                     mat::ksi * pow(p0 / PA, mat::ksi);
#ifdef DEBUG_SPAN_ENABLED
    auto Rp0_denom_view = std::span<data_t>(Rp0_denom.data_ptr<data_t>(), Rp0_denom.numel());
#endif
    auto Rp0 = Safe_divide_::call((1.0 + void_alpha) * p0 * exprf, Rp0_denom);
#ifdef DEBUG_SPAN_ENABLED
    auto Rp0_view = std::span<data_t>(Rp0.data_ptr<data_t>(), Rp0.numel());
#endif
    auto kp = -(sum(pfalpha * Ralpha) + pfp0 * Rp0).item<data_t>();
    return std::make_tuple(Ralpha, Rp0, kp);
  }
}
template MYUMAT_API auto
Safe_divide_::call<torch::Tensor, torch::Tensor>(torch::Tensor, torch::Tensor, ErrorCode *err,
                                                 utils::data_t epsilon) -> torch::Tensor;
template MYUMAT_API auto
Safe_divide_::call<torch::Tensor, utils::data_t>(torch::Tensor, utils::data_t, ErrorCode *err,
                                                 utils::data_t epsilon) -> torch::Tensor;
template MYUMAT_API auto
Safe_divide_::call<utils::data_t, torch::Tensor>(utils::data_t, torch::Tensor, ErrorCode *err,
                                                 utils::data_t epsilon) -> torch::Tensor;
template MYUMAT_API auto
Safe_divide_::call<utils::data_t, utils::data_t>(utils::data_t, utils::data_t, ErrorCode *err,
                                                 utils::data_t epsilon) -> utils::data_t;
template MYUMAT_API auto Calc_Cosine_angle_::call<true>(const torch::Tensor &lhs,
                                                        const torch::Tensor &rhs, ErrorCode *err,
                                                        utils::data_t epsilon) -> torch::Tensor;
template MYUMAT_API auto Calc_Cosine_angle_::call<false>(const torch::Tensor &lhs,
                                                         const torch::Tensor &rhs, ErrorCode *err,
                                                         utils::data_t epsilon) -> utils::data_t;
template MYUMAT_API auto Pressure_::call<true>(const torch::Tensor &self, ErrorCode *err)
    -> torch::Tensor;
template MYUMAT_API auto Pressure_::call<false>(const torch::Tensor &self, ErrorCode *err)
    -> utils::data_t;
template MYUMAT_API auto Deviatoric_::call<true>(const torch::Tensor &self, ErrorCode *err)
    -> torch::Tensor;
template MYUMAT_API auto Deviatoric_::call<false>(const torch::Tensor &self, ErrorCode *err)
    -> torch::Tensor;
template MYUMAT_API auto StressRatio_::call<true>(const torch::Tensor &self,
                                                  ErrorCode *err = nullptr,
                                                  utils::data_t epsilon = 1e-12) -> torch::Tensor;
template MYUMAT_API auto StressRatio_::call<false>(const torch::Tensor &self,
                                                   ErrorCode *err = nullptr,
                                                   utils::data_t epsilon = 1e-12) -> torch::Tensor;
template MYUMAT_API auto Calc_Rm_::call<true>(const torch::Tensor &self, ErrorCode *err = nullptr,
                                              utils::data_t epsilon = 1e-12, torch::Tensor)
    -> torch::Tensor;
template MYUMAT_API auto Calc_Rm_::call<false>(const torch::Tensor &self, ErrorCode *err = nullptr,
                                               utils::data_t epsilon = 1e-12, torch::Tensor)
    -> utils::data_t;
template MYUMAT_API auto LoadingDirection_::call<true>(const core::impl::ShareVarImpl &shvarImpl,
                                                       ErrorCode *err = nullptr,
                                                       utils::data_t epsilon = 1e-12,
                                                       torch::Tensor ratio) -> torch::Tensor;
template MYUMAT_API auto LoadingDirection_::call<false>(const core::impl::ShareVarImpl &shvarImpl,
                                                        ErrorCode *err = nullptr,
                                                        utils::data_t epsilon = 1e-12,
                                                        torch::Tensor ratio) -> torch::Tensor;
template MYUMAT_API auto Cos3theta_::call<true>(const core::impl::ShareVarImpl &shvarImpl,
                                                ErrorCode *err = nullptr,
                                                utils::data_t epsilon = 1e-12, Tensor)
    -> torch::Tensor;
template MYUMAT_API auto Cos3theta_::call<false>(const core::impl::ShareVarImpl &shvarImpl,
                                                 ErrorCode *err = nullptr,
                                                 utils::data_t epsilon = 1e-12,
                                                 torch::Tensor norm = {}) -> utils::data_t;
template MYUMAT_API auto Calc_gtheta_::call<true, true>(const core::impl::ShareVarImpl &shvarImpl,
                                                        ErrorCode *err, utils::data_t epsilon,
                                                        Scalar_Type cos3t)
    -> std::pair<torch::Tensor, torch::Tensor>;
template MYUMAT_API auto Calc_gtheta_::call<true, false>(const core::impl::ShareVarImpl &shvarImpl,
                                                         ErrorCode *err, utils::data_t epsilon,
                                                         Scalar_Type cos3t) -> torch::Tensor;

template MYUMAT_API auto Calc_gtheta_::call<false, true>(const core::impl::ShareVarImpl &shvarImpl,
                                                         ErrorCode *err, utils::data_t epsilon,
                                                         Scalar_Type cos3t)
    -> std::pair<utils::data_t, torch::Tensor>;

template MYUMAT_API auto Calc_gtheta_::call<false, false>(const core::impl::ShareVarImpl &shvarImpl,
                                                          ErrorCode *err, utils::data_t epsilon,
                                                          Scalar_Type cos3t) -> utils::data_t;

template MYUMAT_API auto Calc_lamda_::call<true>(const core::impl::ShareVarImpl &shvarImpl,
                                                 ErrorCode *err = nullptr,
                                                 utils::data_t epsilon = 1e-12, Scalar_Type)
    -> torch::Tensor;
template MYUMAT_API auto Calc_lamda_::call<false>(const core::impl::ShareVarImpl &shvarImpl,
                                                  ErrorCode *err = nullptr,
                                                  utils::data_t epsilon = 1e-12, Scalar_Type)
    -> utils::data_t;
template MYUMAT_API auto Calc_GV_::call<true>(const torch::Tensor &self, ErrorCode *err = nullptr,
                                              Scalar_Type) -> torch::Tensor;
template MYUMAT_API auto Calc_GV_::call<false>(const torch::Tensor &self, ErrorCode *err = nullptr,
                                               Scalar_Type) -> utils::data_t;
template MYUMAT_API auto Calc_shear_bulk_::call<true>(const core::impl::ShareVarImpl &shvarImpl,
                                                      utils::data_t voidr, ErrorCode *err = nullptr,
                                                      utils::data_t epsilon = 1e-12)
    -> std::pair<Tensor, Tensor>;
template MYUMAT_API auto
Calc_shear_bulk_::call<false>(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                              ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
    -> std::pair<data_t, data_t>;
template MYUMAT_API auto Stiffness_::call<true>(const impl::ShareVarImpl &shvarImpl,
                                                utils::data_t voidr, ErrorCode *err = nullptr,
                                                utils::data_t epsilon = 1e-12) -> torch::Tensor;
template MYUMAT_API auto Stiffness_::call<false>(const impl::ShareVarImpl &shvarImpl,
                                                 utils::data_t voidr, ErrorCode *err = nullptr,
                                                 utils::data_t epsilon = 1e-12) -> torch::Tensor;
template MYUMAT_API auto Calc_yield_::call<true>(const core::impl::ShareVarImpl &shvarImpl,
                                                 ErrorCode *err = nullptr) -> torch::Tensor;
template MYUMAT_API auto Calc_yield_::call<false>(const core::impl::ShareVarImpl &shvarImpl,
                                                  ErrorCode *err = nullptr) -> utils::data_t;
template MYUMAT_API auto Calc_psim_::call<true>(const torch::Tensor &self, utils::data_t voidr,
                                                ErrorCode *err, Scalar_Type pressure = {})
    -> torch::Tensor;
template MYUMAT_API auto Calc_psim_::call<false>(const torch::Tensor &self, utils::data_t voidr,
                                                 ErrorCode *err, Scalar_Type pressure = {})
    -> utils::data_t;
template MYUMAT_API auto Calc_psim_alpha_::call<true>(const impl::ShareVarImpl &shvarImpl,
                                                      utils::data_t voidr, ErrorCode *err = nullptr,
                                                      utils::data_t epsilon = 1e-12,
                                                      Scalar_Type pressure, Scalar_Type lamda_alpha)
    -> torch::Tensor;
template MYUMAT_API auto Calc_psim_alpha_::call<false>(const impl::ShareVarImpl &shvarImpl,
                                                       utils::data_t voidr,
                                                       ErrorCode *err = nullptr,
                                                       utils::data_t epsilon = 1e-12, Scalar_Type,
                                                       Scalar_Type) -> utils::data_t;
template MYUMAT_API auto
Dilatancy_::call<true>(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                       const core::PlasticOptions &options = {}, ErrorCode *err = nullptr,
                       utils::data_t epsilon = 1e-12) -> std::pair<torch::Tensor, torch::Tensor>;
template MYUMAT_API auto
Dilatancy_::call<false>(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                        const core::PlasticOptions &options = {}, ErrorCode *err = nullptr,
                        utils::data_t epsilon = 1e-12) -> std::pair<utils::data_t, utils::data_t>;
template MYUMAT_API auto Evolution_Dkp_::call<true>(const core::impl::ShareVarImpl &shvarImpl,
                                                    const core::impl::StateVarImpl &stvarsImpl,
                                                    const core::PlasticOptions &options = {},
                                                    ErrorCode *err = nullptr,
                                                    utils::data_t epsilon = 1e-12)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
template MYUMAT_API auto Evolution_Dkp_::call<false>(const core::impl::ShareVarImpl &shvarImpl,
                                                     const core::impl::StateVarImpl &stvarsImpl,
                                                     const core::PlasticOptions &options = {},
                                                     ErrorCode *err = nullptr,
                                                     utils::data_t epsilon = 1e-12)
    -> std::tuple<torch::Tensor, torch::Tensor, utils::data_t>;

// Safe_divide_ 模板实例化
template auto Safe_divide_::call<torch::Tensor, double>(torch::Tensor numerator, double denominator,
                                                        ErrorCode *err, utils::data_t epsilon)
    -> torch::Tensor;

template auto
Safe_divide_::call<torch::Tensor, torch::Tensor>(torch::Tensor numerator, torch::Tensor denominator,
                                                 ErrorCode *err, utils::data_t epsilon)
    -> torch::Tensor;

template auto Safe_divide_::call<double, double>(double numerator, double denominator,
                                                 ErrorCode *err, utils::data_t epsilon) -> double;

// #ifdef __clang__
// #pragma clang diagnostic pop
// #endif
} // namespace umat::core::impl
