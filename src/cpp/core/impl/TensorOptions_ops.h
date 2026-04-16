#ifndef CORE_TENSOROPTIONS_OPS_H
#define CORE_TENSOROPTIONS_OPS_H
#include "ShareVarImpl.h"
#include "core/ShareVarOptions.h"
#include "core/auxiliary.h"
#include "torch/torch.h"
#include "utils/TypeMap.h"
#include "utils/base_config.h"
#include "utils/concept.h"
#include "utils/export.h"
#include "utils/material_config.h"
#include <span>

namespace umat::core::impl {

struct base_type {
  template <bool is_true>
  using Scalar_type = std::conditional_t<is_true, torch::Tensor, utils::data_t>;
  using Tensor = torch::Tensor;
  using data_t = utils::data_t;
};
struct MYUMAT_API Safe_divide_ {

  template <utils::Scalartype T, utils::Scalartype U>
  [[nodiscard]] static auto call(T numerator, U denominator, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12)
      -> std::conditional_t<(std::is_same_v<std::decay_t<T>, utils::data_t> &&
                             std::is_same_v<std::decay_t<U>, utils::data_t>),
                            utils::data_t, torch::Tensor>;
};
struct Is_isotropic_ {
  [[nodiscard]] static auto call(const torch::Tensor &self) -> bool;
};
struct MYUMAT_API Is_nan_inf_ {
  [[nodiscard]] static auto call(const torch::Tensor &tensor) -> bool;
};
struct MYUMAT_API Calc_shvar_norm_ {
  [[nodiscard]] static auto call(const ShareVarImpl &shvarsImpl, ErrorCode *err = nullptr)
      -> torch::Tensor;
};
struct MYUMAT_API Calc_Cosine_angle_ {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &lhs, const torch::Tensor &rhs,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
      -> std::conditional_t<retain_map, torch::Tensor, utils::data_t>;
};
struct MYUMAT_API Pressure_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &self, ErrorCode *err = nullptr)
      -> Scalar_type<retain_map>;
};

struct MYUMAT_API Deviatoric_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &self, ErrorCode *err = nullptr)
      -> torch::Tensor;
};
/**
 * @brief
 *
 **/
struct MYUMAT_API StressRatio_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &self, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12) -> torch::Tensor;
};
struct MYUMAT_API Calc_Rm_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &self, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12, torch::Tensor ratio = {})
      -> Scalar_type<retain_map>;
};
struct MYUMAT_API LoadingDirection_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                 torch::Tensor ratio = {}) -> torch::Tensor;
};
struct MYUMAT_API Cos3theta_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                 torch::Tensor norm = {}) -> Scalar_type<retain_map>;
};
struct MYUMAT_API Calc_gtheta_ : base_type {
  template <bool retain_map = false, bool is_grad = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                 Scalar_Type cos3t = {})
      -> std::conditional_t<is_grad, std::pair<Scalar_type<retain_map>, torch::Tensor>,
                            Scalar_type<retain_map>>;
};
struct MYUMAT_API Calc_lamda_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                 Scalar_Type gtheta = {}) -> Scalar_type<retain_map>;
};
struct MYUMAT_API Calc_GV_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &self, ErrorCode *err = nullptr,
                                 Scalar_Type pressure = {}) -> Scalar_type<retain_map>;
};

struct MYUMAT_API Calc_shear_bulk_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
      -> std::conditional_t<retain_map, std::pair<Tensor, Tensor>, std::pair<data_t, data_t>>;
};

struct MYUMAT_API Stiffness_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12)
      -> torch::Tensor;
};

struct MYUMAT_API Calc_yield_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 ErrorCode *err = nullptr) -> Scalar_type<retain_map>;
};
struct MYUMAT_API Pfpsigma_ {
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 ErrorCode *err = nullptr) -> torch::Tensor;
};
struct MYUMAT_API Pgpsigma_ {
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                                 const core::PlasticOptions &options = {}, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12) -> torch::Tensor;
};
struct MYUMAT_API Calc_psim_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const torch::Tensor &self, utils::data_t voidr, ErrorCode *err,
                                 Scalar_Type pressure = {}) -> Scalar_type<retain_map>;
};
struct MYUMAT_API Calc_psim_alpha_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                                 ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                 Scalar_Type pressure = {}, Scalar_Type lamda_alpha = {})
      -> Scalar_type<retain_map>;
};
struct MYUMAT_API Dilatancy_ {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                                 const core::PlasticOptions &options = {}, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12)
      -> std::conditional_t<retain_map, std::pair<torch::Tensor, torch::Tensor>,
                            std::pair<utils::data_t, utils::data_t>>;
};

struct MYUMAT_API Evolution_Dkp_ : base_type {
  template <bool retain_map = false>
  [[nodiscard]] static auto call(const core::impl::ShareVarImpl &shvarImpl,
                                 const core::impl::StateVarImpl &stvarImpl,
                                 const core::PlasticOptions &options = {}, ErrorCode *err = nullptr,
                                 utils::data_t epsilon = 1e-12)
      -> std::tuple<torch::Tensor, torch::Tensor, Scalar_type<retain_map>>;
};
// 实例化函数
// Safe_divide_
extern template MYUMAT_API auto
Safe_divide_::call<torch::Tensor, torch::Tensor>(torch::Tensor, torch::Tensor, ErrorCode *err,
                                                 utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto
Safe_divide_::call<torch::Tensor, utils::data_t>(torch::Tensor, utils::data_t, ErrorCode *err,
                                                 utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto
Safe_divide_::call<utils::data_t, torch::Tensor>(utils::data_t, torch::Tensor, ErrorCode *err,
                                                 utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto
Safe_divide_::call<utils::data_t, utils::data_t>(utils::data_t, utils::data_t, ErrorCode *err,
                                                 utils::data_t epsilon) -> utils::data_t;
// Calc_Cosine_angle_
extern template MYUMAT_API auto
Calc_Cosine_angle_::call<true>(const torch::Tensor &lhs, const torch::Tensor &rhs, ErrorCode *err,
                               utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto
Calc_Cosine_angle_::call<false>(const torch::Tensor &lhs, const torch::Tensor &rhs, ErrorCode *err,
                                utils::data_t epsilon) -> utils::data_t;
// Pressure_
extern template MYUMAT_API auto Pressure_::call<true>(const torch::Tensor &self, ErrorCode *err)
    -> torch::Tensor;
extern template MYUMAT_API auto Pressure_::call<false>(const torch::Tensor &self, ErrorCode *err)
    -> utils::data_t;
extern template MYUMAT_API auto Deviatoric_::call<true>(const torch::Tensor &self, ErrorCode *err)
    -> torch::Tensor;
extern template MYUMAT_API auto Deviatoric_::call<false>(const torch::Tensor &self, ErrorCode *err)
    -> torch::Tensor;
extern template MYUMAT_API auto StressRatio_::call<true>(const torch::Tensor &self, ErrorCode *err,
                                                         utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto StressRatio_::call<false>(const torch::Tensor &self, ErrorCode *err,
                                                          utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto Calc_Rm_::call<true>(const torch::Tensor &self, ErrorCode *err,
                                                     utils::data_t epsilon, torch::Tensor)
    -> torch::Tensor;
extern template MYUMAT_API auto Calc_Rm_::call<false>(const torch::Tensor &self, ErrorCode *err,
                                                      utils::data_t epsilon, torch::Tensor)
    -> utils::data_t;
extern template MYUMAT_API auto
LoadingDirection_::call<true>(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                              utils::data_t epsilon, torch::Tensor ratio) -> torch::Tensor;
extern template MYUMAT_API auto
LoadingDirection_::call<false>(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                               utils::data_t epsilon, torch::Tensor ratio) -> torch::Tensor;
extern template MYUMAT_API auto Cos3theta_::call<true>(const core::impl::ShareVarImpl &shvarImpl,
                                                       ErrorCode *err, utils::data_t epsilon,
                                                       torch::Tensor norm) -> torch::Tensor;
extern template MYUMAT_API auto Cos3theta_::call<false>(const core::impl::ShareVarImpl &shvarImpl,
                                                        ErrorCode *err, utils::data_t epsilon,
                                                        torch::Tensor norm) -> utils::data_t;
extern template MYUMAT_API auto
Calc_gtheta_::call<true, true>(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                               utils::data_t epsilon, Scalar_Type cos3t)
    -> std::pair<torch::Tensor, torch::Tensor>;
extern template MYUMAT_API auto
Calc_gtheta_::call<true, false>(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                                utils::data_t epsilon, Scalar_Type cos3t) -> torch::Tensor;

extern template MYUMAT_API auto
Calc_gtheta_::call<false, true>(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                                utils::data_t epsilon, Scalar_Type cos3t)
    -> std::pair<utils::data_t, torch::Tensor>;

extern template MYUMAT_API auto
Calc_gtheta_::call<false, false>(const core::impl::ShareVarImpl &shvarImpl, ErrorCode *err,
                                 utils::data_t epsilon, Scalar_Type cos3t) -> utils::data_t;

extern template MYUMAT_API auto Calc_lamda_::call<true>(const core::impl::ShareVarImpl &shvarImpl,
                                                        ErrorCode *err, utils::data_t epsilon,
                                                        Scalar_Type) -> torch::Tensor;
extern template MYUMAT_API auto Calc_lamda_::call<false>(const core::impl::ShareVarImpl &shvarImpl,
                                                         ErrorCode *err, utils::data_t epsilon,
                                                         Scalar_Type gtheta) -> utils::data_t;
extern template MYUMAT_API auto Calc_GV_::call<true>(const torch::Tensor &self, ErrorCode *err,
                                                     Scalar_Type) -> torch::Tensor;
extern template MYUMAT_API auto Calc_GV_::call<false>(const torch::Tensor &self, ErrorCode *err,
                                                      Scalar_Type pressure) -> utils::data_t;
extern template MYUMAT_API auto
Calc_shear_bulk_::call<true>(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                             ErrorCode *err, utils::data_t epsilon) -> std::pair<Tensor, Tensor>;
extern template MYUMAT_API auto
Calc_shear_bulk_::call<false>(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                              ErrorCode *err, utils::data_t epsilon) -> std::pair<data_t, data_t>;
extern template MYUMAT_API auto Stiffness_::call<true>(const impl::ShareVarImpl &shvarImpl,
                                                       utils::data_t voidr, ErrorCode *err,
                                                       utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto Stiffness_::call<false>(const impl::ShareVarImpl &shvarImpl,
                                                        utils::data_t voidr, ErrorCode *err,
                                                        utils::data_t epsilon) -> torch::Tensor;
extern template MYUMAT_API auto Calc_yield_::call<true>(const core::impl::ShareVarImpl &shvarImpl,
                                                        ErrorCode *err) -> torch::Tensor;
extern template MYUMAT_API auto Calc_yield_::call<false>(const core::impl::ShareVarImpl &shvarImpl,
                                                         ErrorCode *err) -> utils::data_t;
extern template MYUMAT_API auto Calc_psim_::call<true>(const torch::Tensor &self,
                                                       utils::data_t voidr, ErrorCode *err,
                                                       Scalar_Type pressure) -> torch::Tensor;
extern template MYUMAT_API auto Calc_psim_::call<false>(const torch::Tensor &self,
                                                        utils::data_t voidr, ErrorCode *err,
                                                        Scalar_Type pressure) -> utils::data_t;
extern template MYUMAT_API auto Calc_psim_alpha_::call<true>(const impl::ShareVarImpl &shvarImpl,
                                                             utils::data_t voidr, ErrorCode *err,
                                                             utils::data_t epsilon, Scalar_Type,
                                                             Scalar_Type) -> torch::Tensor;
extern template MYUMAT_API auto Calc_psim_alpha_::call<false>(const impl::ShareVarImpl &shvarImpl,
                                                              utils::data_t voidr, ErrorCode *err,
                                                              utils::data_t epsilon, Scalar_Type,
                                                              Scalar_Type) -> utils::data_t;
extern template MYUMAT_API auto
Dilatancy_::call<true>(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                       const core::PlasticOptions &options, ErrorCode *err, utils::data_t epsilon)
    -> std::pair<torch::Tensor, torch::Tensor>;
extern template MYUMAT_API auto
Dilatancy_::call<false>(const core::impl::ShareVarImpl &shvarImpl, utils::data_t voidr,
                        const core::PlasticOptions &options, ErrorCode *err, utils::data_t epsilon)
    -> std::pair<utils::data_t, utils::data_t>;
extern template MYUMAT_API auto Evolution_Dkp_::call<true>(
    const core::impl::ShareVarImpl &shvarImpl, const core::impl::StateVarImpl &stvarsImpl,
    const core::PlasticOptions &options, ErrorCode *err, utils::data_t epsilon)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
extern template MYUMAT_API auto Evolution_Dkp_::call<false>(
    const core::impl::ShareVarImpl &shvarImpl, const core::impl::StateVarImpl &stvarsImpl,
    const core::PlasticOptions &options, ErrorCode *err, utils::data_t epsilon)
    -> std::tuple<torch::Tensor, torch::Tensor, utils::data_t>;
} // namespace umat::core::impl

#endif // CORE_TENSOROPTIONS_OPS_H
