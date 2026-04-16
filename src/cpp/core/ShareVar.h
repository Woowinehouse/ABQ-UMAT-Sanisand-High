#ifndef CORE_SHAREVAR_H
#define CORE_SHAREVAR_H

#include "ShareVarOptions.h"
#include "core/StateVar.h"
#include "core/impl/ShareVarImpl.h"
#include "core/impl/TensorOptions_ops.h"
#include "utils/TypeMap.h"
#include "utils/export.h"

#ifdef __GTEST__BUILD_
#include <gtest/gtest.h>
#endif
namespace umat::core {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif
/**
 * @brief Shared variables container for UMAT (User Material) constitutive model.
 *
 * This class encapsulates the stress state variables (stress, back-stress alpha,
 * p0 tensor) used in constitutive modeling. It provides methods for elastic
 * and plastic computations, state management, and tensor operations.
 *
 * The class uses intrusive pointer semantics for efficient memory management
 * and copy-on-write behavior through ShareVarImpl.
 **/
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class MYUMAT_API ShareVar final {
  // friendly
  friend class ShareVarTest;
  friend StateVar;
#ifdef __GTEST__BUILD_
  FRIEND_TEST(ShareVarTest, DefaultConstructor);
  FRIEND_TEST(ShareVarTest, Assignment);
#endif
  //
  using ShareVarImpl = impl::ShareVarImpl;
  using Tensor = torch::Tensor;
  using data_t = utils::data_t;
  using State = utils::StressState;
  using StressTensor = core::StressTensor;

  protected:
  // member
  c10::intrusive_ptr<ShareVarImpl> impl_;

  public:
  ShareVar() = default;
  explicit ShareVar(c10::intrusive_ptr<ShareVarImpl> ptr) : impl_(std::move(ptr)) {}
  ShareVar(StressTensor stress, StressTensor alpha, StressTensor p0);
  ShareVar(Tensor stress, Tensor alpha, Tensor p0, utils::StressState state);

  ///@name constructors
  ///@{
  static auto create(StressTensor stress, StressTensor alpha, StressTensor p0) -> ShareVar;
  static auto create(Tensor stress, Tensor alpha, Tensor p0, State state) -> ShareVar;

  ShareVar(const ShareVar &other) = default;
  ShareVar(ShareVar &&other) noexcept : ShareVar() { swap(*this, other); }
  /// @}
  ///@name destructor
  ~ShareVar() = default;

  ///@name operator
  /// @{
  // @brief assignment
  auto operator=(ShareVar other) -> ShareVar & {
    swap(*this, other);
    return *this;
  }
  // @brief unary operator
  // binary operator
  auto operator-() const -> ShareVar { return minus(); }
  auto operator+(const core::ShareVar &rhs) const -> ShareVar { return add(rhs); }
  auto operator-(const core::ShareVar &rhs) const -> ShareVar { return sub(rhs); }
  auto operator*(const core::ShareVar &rhs) const -> ShareVar { return mul(rhs); }
  auto operator/(const core::ShareVar &rhs) const -> ShareVar { return div(rhs); }
  auto operator/(utils::data_t scalar) const -> ShareVar { return div_scalar(scalar); }
  auto operator*(data_t scalar) const -> ShareVar {
    auto impl = c10::make_intrusive<impl::ShareVarImpl>(impl_->mul_scalar(scalar));
    return ShareVar(std::move(impl));
  }
  //
  auto operator+=(const core::ShareVar &rhs) -> ShareVar & { return add_(rhs); }
  auto operator-=(const core::ShareVar &rhs) -> ShareVar & { return sub_(rhs); }
  auto operator*=(const core::ShareVar &rhs) -> ShareVar & { return mul_(rhs); }
  auto operator/=(const core::ShareVar &rhs) -> ShareVar & { return div_(rhs); }
  /// @}
  [[nodiscard]] auto minus() const -> ShareVar;
  [[nodiscard]] auto add(const ShareVar &rhs, data_t scalar = 1.0) const -> ShareVar;
  [[nodiscard]] auto sub(const ShareVar &rhs, data_t scalar = 1.0) const -> ShareVar;
  [[nodiscard]] auto mul(const ShareVar &rhs, data_t scalar = 1.0) const -> ShareVar;
  [[nodiscard]] auto div(const ShareVar &rhs) const -> ShareVar;
  [[nodiscard]] auto div_scalar(utils::data_t scalar) const -> ShareVar;
  auto add_(const ShareVar &other, data_t scalar = 1.0) -> ShareVar &;
  auto sub_(const ShareVar &other, data_t scalar = 1.0) -> ShareVar &;
  auto mul_(const ShareVar &other, data_t scalar = 1.0) -> ShareVar &;
  auto div_(const ShareVar &other) -> ShareVar &;
  /// @name scope guard
  /// @{
  [[nodiscard]] auto backup_state() const -> ShareVar;
  auto restore_state(const ShareVar &backup) -> void;
  /// @}

  ///@name Getters
  ///@{
  [[nodiscard]] auto stress_data_ptr() const -> const void * { return impl_->stress_data_ptr(); }
  [[nodiscard]] auto alpha_data_ptr() const -> const void * { return impl_->alpha_data_ptr(); }
  [[nodiscard]] auto p0_data_ptr() const -> const void * { return impl_->p0_data_ptr(); }
  // get const data_ptr
  [[nodiscard]] auto mutable_stress_data_ptr() const -> void * {
    return impl_->mutable_stress_data_ptr();
  }
  [[nodiscard]] auto mutable_alpha_data_ptr() const -> void * {
    return impl_->mutable_alpha_data_ptr();
  }
  [[nodiscard]] auto mutable_p0_data_ptr() const -> void * { return impl_->mutable_p0_data_ptr(); }
  [[nodiscard]] auto data_ptr() const -> impl::ShareVarImpl * { return impl_.get(); }
  [[nodiscard]] auto GetShareVarImpl() const -> const impl::ShareVarImpl & { return *impl_.get(); }
  [[nodiscard]] auto unsafeGetShareVarImpl() -> impl::ShareVarImpl & { return *impl_.get(); }
  // get StressTensor
  auto get_stress() const noexcept -> const StressTensor & { return impl_->get_stress(); };
  auto get_alpha() const noexcept -> const StressTensor & { return impl_->get_alpha(); };
  auto get_p0() const noexcept -> const StressTensor & { return impl_->get_p0(); };
  auto unsafe_get_stress() -> StressTensor & { return impl_->unsafe_get_stress(); };
  auto unsafe_get_alpha() -> StressTensor & { return impl_->unsafe_get_alpha(); };
  auto unsafe_get_p0() -> StressTensor & { return impl_->unsafe_get_p0(); };
  // get tensor
  auto get_stress_tensor() const noexcept -> const Tensor & { return impl_->get_stress_tensor(); };
  auto get_alpha_tensor() const noexcept -> const Tensor & { return impl_->get_alpha_tensor(); };
  auto get_p0_tensor() const noexcept -> const Tensor & { return impl_->get_p0_tensor(); };
  auto unsafe_get_stress_tensor() noexcept -> Tensor & {
    return impl_->unsafe_get_stress_tensor();
  };
  auto unsafe_get_alpha_tensor() noexcept -> Tensor & { return impl_->unsafe_get_alpha_tensor(); };
  auto unsafe_get_p0_tensor() noexcept -> Tensor & { return impl_->unsafe_get_p0_tensor(); };

  // @brief get the member ptr
  auto stress_ptr() const noexcept -> const StressTensor * { return impl_->stress_ptr(); }
  auto alpha_ptr() const noexcept -> const StressTensor * { return impl_->alpha_ptr(); }
  auto p0_ptr() const noexcept -> const StressTensor * { return impl_->p0_ptr(); }
  auto unsafe_stress_ptr() -> StressTensor * { return impl_->unsafe_stress_ptr(); }
  auto unsafe_alpha_ptr() -> StressTensor * { return impl_->unsafe_alpha_ptr(); }
  auto unsafe_p0_ptr() -> StressTensor * { return impl_->unsafe_p0_ptr(); }

  [[nodiscard]] auto stress_tensor_ptr() const noexcept -> const Tensor * {
    return impl_->stress_tensor_ptr();
  }
  [[nodiscard]] auto alpha_tensor_ptr() const noexcept -> const Tensor * {
    return impl_->alpha_tensor_ptr();
  }
  [[nodiscard]] auto p0_tensor_ptr() const noexcept -> const Tensor * {
    return impl_->p0_tensor_ptr();
  }
  auto unsafe_stress_tensor_ptr() -> Tensor * { return impl_->unsafe_stress_tensor_ptr(); }
  auto unsafe_alpha_tensor_ptr() -> Tensor * { return impl_->unsafe_alpha_tensor_ptr(); }
  auto unsafe_p0_tensor_ptr() -> Tensor * { return impl_->unsafe_p0_tensor_ptr(); }
  // use count
  [[nodiscard]] auto get_stress_count() const -> size_t { return impl_->get_stress_count(); }
  [[nodiscard]] auto get_alpha_count() const -> size_t { return impl_->get_alpha_count(); }
  [[nodiscard]] auto get_p0_count() const -> size_t { return impl_->get_p0_count(); }
  [[nodiscard]] auto get_state() const -> State { return impl_->get_state(); }
  [[nodiscard]] auto is_lowstress() const -> bool { return impl_->is_low_stress(); }
  ///@}

  /// @name Setters
  /// @{
  auto set_stress(StressTensor new_tensor) -> void { impl_->set_stress(new_tensor); }
  auto set_alpha(StressTensor new_alpha) -> void { impl_->set_alpha(new_alpha); }
  auto set_p0(StressTensor new_p0) -> void { impl_->set_p0(new_p0); }
  auto update_stress(Tensor stress_tensor, State state) -> void {
    return update_stress(StressTensor(stress_tensor, state));
  }
  auto update_stress(const StressTensor &dstress) -> void { impl_->update_stress(dstress); }
  auto update_alpha(const StressTensor &dalpha) -> void { impl_->update_alpha(dalpha); }
  auto update_p0(const StressTensor &dp0) -> void { impl_->update_p0(dp0); }
  auto update_shareVar(const ShareVar &dshvar) -> void;
  auto update_shareVar(const StressTensor &dstress, const StressTensor &dalpha,
                       const StressTensor &dp0) -> void;
  auto update_shareVar(Tensor dstress, Tensor dalpha, Tensor dp0, utils::StressState state) -> void;
  [[nodiscard]] inline auto create_shvar_with_dstress(StressTensor dstress) const
      -> core::ShareVar {
    return ShareVar(
        c10::make_intrusive<impl::ShareVarImpl>(impl_->create_shvar_with_dstress(dstress)));
  }
  [[nodiscard]] auto create_shvar_from_new_stress(const StressTensor &new_stress) const
      -> core::ShareVar {
    return ShareVar(
        c10::make_intrusive<impl::ShareVarImpl>(impl_->create_shvar_from_new_stress(new_stress)));
  }
  /// @}

  ///@name other methods
  /// @{
  [[nodiscard]] auto clone() const -> ShareVar {
    return ShareVar(c10::make_intrusive<impl::ShareVarImpl>(impl_->clone()));
  }
  [[nodiscard]] auto lazy_clone() const -> ShareVar {
    return ShareVar(c10::make_intrusive<impl::ShareVarImpl>(impl_->lazy_clone()));
  }
  [[nodiscard]] auto use_count() const -> size_t { return impl_.use_count(); }
  [[nodiscard]] auto unique() const -> bool { return impl_.unique(); }

  auto validate() const -> void { impl_->validate(); }
  /// @}

  /// @name elastic methods
  ///@{
  template <bool is_true>
  using Scalar_type = std::conditional_t<is_true, torch::Tensor, utils::data_t>;
  /**
   * @brief
   * @tparam retain_map
   * @param  err
   *
   * @return Scalar_type<retain_map>
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto mean_pressure(ErrorCode *err = nullptr) -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Pressure_::call<true>(get_stress_tensor(), err);
      return res;
    } else {
      auto res = impl::Pressure_::call<false>(get_stress_tensor(), err);
      return res;
    }
  }
  /**
   * @brief
   * @tparam retain_map
   * @param  err
   *
   * @return torch::Tensor
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto deviatoric(ErrorCode *err = nullptr) const -> torch::Tensor {
    if constexpr (retain_map) {
      auto res = impl::Deviatoric_::call<true>(get_stress_tensor(), err);
      return res;
    } else {
      auto res = impl::Deviatoric_::call<false>(get_stress_tensor(), err);
      return res;
    }
  }
  /**
   * @brief
   * @tparam retain_map
   * @param  err
   * @param  epsilon
   *
   * @return torch::Tensor
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto StressRatio(ErrorCode *err = nullptr, data_t epsilon = 1e-12) const
      -> torch::Tensor {
    if constexpr (retain_map) {
      auto res = impl::StressRatio_::call<true>(get_stress_tensor(), err, epsilon);
      return res;
    } else {
      auto res = impl::StressRatio_::call<false>(get_stress_tensor(), err, epsilon);
      return res;
    }
  }
  /**
   * @brief
   * @tparam retain_map
   * @param  err
   * @param  epsilon
   * @param  ratio
   *
   * @return Scalar_type<retain_map>
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto calc_Rm(ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                             torch::Tensor ratio = {}) -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Calc_Rm_::call<true>(get_stress_tensor(), err, epsilon, ratio);
      return res;
    } else {
      auto res = impl::Calc_Rm_::call<false>(get_stress_tensor(), err, epsilon, ratio);
      return res;
    }
  }
  /**
   * @brief
   * @tparam retain_map
   * @param  err
   * @param  epsilon
   * @param  ratio
   *
   * @return torch::Tensor
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto loadingDirection(ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                      torch::Tensor ratio = {}) const -> torch::Tensor {
    if constexpr (retain_map) {
      auto res = impl::LoadingDirection_::call<true>(GetShareVarImpl(), err, epsilon, ratio);
      return res;
    } else {
      auto res = impl::LoadingDirection_::call<false>(GetShareVarImpl(), err, epsilon, ratio);
      return res;
    }
  }
  /**
   * @brief
   * @tparam retain_map
   * @param  err
   * @param  epsilon
   * @param  norm
   *
   * @return Scalar_type<retain_map>
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto cos3theta(ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                               torch::Tensor norm = {}) const -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Cos3theta_::call<true>(GetShareVarImpl(), err, epsilon, norm);
      return res;
    } else {
      auto res = impl::Cos3theta_::call<false>(GetShareVarImpl(), err, epsilon, norm);
      return res;
    }
  }
  /**
   * @brief
   * @tparam retain_map
   * @tparam is_grad
   * @param  err
   * @param  epsilon
   * @param  cos3t
   *
   * @return std::conditional_t<is_grad, std::pair<Scalar_type<retain_map>, torch::Tensor>,
   * Scalar_type<retain_map>>
   **/
  template <bool retain_map = false, bool is_grad = false>
  [[nodiscard]] auto calc_gtheta(ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                 Scalar_Type cos3t = {}) const
      -> std::conditional_t<is_grad, std::pair<Scalar_type<retain_map>, torch::Tensor>,
                            Scalar_type<retain_map>> {
    if constexpr (retain_map) {
      if constexpr (is_grad) {
        auto res = impl::Calc_gtheta_::call<true, true>(GetShareVarImpl(), err, epsilon, cos3t);
        return res;
      } else {
        auto res = impl::Calc_gtheta_::call<true, false>(GetShareVarImpl(), err, epsilon, cos3t);
        return res;
      }
    } else {
      if constexpr (is_grad) {
        auto res = impl::Calc_gtheta_::call<false, true>(GetShareVarImpl(), err, epsilon, cos3t);
        return res;
      } else {
        auto res = impl::Calc_gtheta_::call<false, false>(GetShareVarImpl(), err, epsilon, cos3t);
        return res;
      }
    }
  }
  /**
   * @brief
   * @tparam retain_map
   * @param  err
   * @param  epsilon
   * @param  gtheta
   *
   * @return Scalar_type<retain_map>
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto calc_lamda(ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                                Scalar_Type gtheta = {}) const -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Calc_lamda_::call<true>(GetShareVarImpl(), err, epsilon, gtheta);
      return res;
    } else {
      auto res = impl::Calc_lamda_::call<false>(GetShareVarImpl(), err, epsilon, gtheta);
      return res;
    }
  }
  /**
   * @brief
   * @tparam retain_map
   * @param  err
   * @param  pressure
   *
   * @return Scalar_type<retain_map>
   **/
  template <bool retain_map = false>
  auto calc_GV(ErrorCode *err = nullptr, Scalar_Type pressure = {}) const
      -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Calc_GV_::call<true>(get_stress_tensor(), err, pressure);
      return res;
    } else {
      auto res = impl::Calc_GV_::call<false>(get_stress_tensor(), err, pressure);
      return res;
    }
  }
  /**
   * @brief
   * @tparam retain_map
   * @param  voidr
   * @param  err
   * @param  epsilon
   *
   * @return std::conditional_t<retain_map, std::pair<Tensor, Tensor>, std::pair<data_t, data_t>>
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto calc_shear_bulk(data_t voidr, ErrorCode *err = nullptr,
                                     data_t epsilon = 1e-12) const
      -> std::conditional_t<retain_map, std::pair<Tensor, Tensor>, std::pair<data_t, data_t>> {
    if constexpr (retain_map) {
      auto res = impl::Calc_shear_bulk_::call<true>(GetShareVarImpl(), voidr, err, epsilon);
      return res;
    } else {
      auto res = impl::Calc_shear_bulk_::call<false>(GetShareVarImpl(), voidr, err, epsilon);
      return res;
    }
  }
  /**
   * @brief
   * @tparam retain_map
   * @param  voidr
   * @param  err
   * @param  epsilon
   *
   * @return torch::Tensor
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto stiffness(utils::data_t voidr, ErrorCode *err = nullptr,
                               utils::data_t epsilon = 1e-12) const -> torch::Tensor {
    if constexpr (retain_map) {
      auto res = impl::Stiffness_::call<true>(GetShareVarImpl(), voidr, err, epsilon);
      return res;
    } else {
      auto res = impl::Stiffness_::call<false>(GetShareVarImpl(), voidr, err, epsilon);
      return res;
    }
  }
  /**
   * @brief
   * @tparam retain_map
   * @param  err
   *
   * @return Scalar_type<retain_map>
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto calc_yield(ErrorCode *err = nullptr) const -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Calc_yield_::call<true>(GetShareVarImpl(), err);
      return res;
    } else {
      auto res = impl::Calc_yield_::call<false>(GetShareVarImpl(), err);
      return res;
    }
  }
  /// @}

  /// @name plastic methods
  /// @{
  /**
   * @brief
   * @param  err
   *
   * @return torch::Tensor
   **/
  [[nodiscard]] auto pfpsigma(ErrorCode *err = nullptr) const -> torch::Tensor {
    return impl::Pfpsigma_::call(GetShareVarImpl(), err);
  }
  /**
   * @brief
   * @param  voidr
   * @param  options
   * @param  err
   * @param  epsilon
   *
   * @return torch::Tensor
   **/
  [[nodiscard]] auto pgpsigma(utils::data_t voidr, const core::PlasticOptions &options = {},
                              ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12) const
      -> torch::Tensor {
    return impl::Pgpsigma_::call(GetShareVarImpl(), voidr, options, err, epsilon);
  }
  /**
   * @brief
   * @tparam retain_map
   * @tparam Dtype
   * @param  voidr
   * @param  err
   * @param  pressure
   *
   * @return Scalar_type<retain_map>
   **/
  template <bool retain_map = false>
  [[nodiscard]] auto calc_psim(utils::data_t voidr, ErrorCode *err, Scalar_Type pressure = {}) const
      -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Calc_psim_::call<true>(get_stress_tensor(), voidr, err, pressure);
      return res;
    } else {
      auto res = impl::Calc_psim_::call<false>(get_stress_tensor(), voidr, err, pressure);
      return res;
    }
  }
  /**
   * @brief
   * @tparam retain_map
   * @param  voidr
   * @param  err
   * @param  epsilon
   * @param  pressure
   * @param  lamda_alpha
   *
   * @return Scalar_type<retain_map>
   **/
  template <bool retain_map = false>
  auto calc_psim_alpha(utils::data_t voidr, ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12,
                       Scalar_Type pressure = {}, Scalar_Type lamda_alpha = {}) const
      -> Scalar_type<retain_map> {
    if constexpr (retain_map) {
      auto res = impl::Calc_psim_alpha_::call<true>(GetShareVarImpl(), voidr, err, epsilon,
                                                    pressure, lamda_alpha);
      return res;
    } else {
      auto res = impl::Calc_psim_alpha_::call<false>(GetShareVarImpl(), voidr, err, epsilon,
                                                     pressure, lamda_alpha);
      return res;
    }
  }

  template <bool retain_map = false>
  [[nodiscard]] auto dilatancy(utils::data_t voidr, const core::PlasticOptions &options = {},
                               ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12) const
      -> std::conditional_t<retain_map, std::pair<torch::Tensor, torch::Tensor>,
                            std::pair<utils::data_t, utils::data_t>> {
    auto &shvarImpl = GetShareVarImpl();
    if constexpr (retain_map) {
      auto res = impl::Dilatancy_::call<true>(shvarImpl, voidr, options, err, epsilon);
      return res;
    } else {
      auto res = impl::Dilatancy_::call<false>(shvarImpl, voidr, options, err, epsilon);
      return res;
    }
  }
  template <bool retain_map = false>
  [[nodiscard]] auto evolution_Kp(const StateVar &stvars, const PlasticOptions &options = {},
                                  ErrorCode *err = nullptr, utils::data_t epsilon = 1e-12) const
      -> std::tuple<torch::Tensor, torch::Tensor, Scalar_type<retain_map>> {
    auto &shvarImpl = GetShareVarImpl();
    auto &stvarImpl = stvars.GetShareVarImpl();
    if constexpr (retain_map) {
      auto res = impl::Evolution_Dkp_::call<true>(shvarImpl, stvarImpl, options, err, epsilon);
      return res;
    } else {
      auto res = impl::Evolution_Dkp_::call<false>(shvarImpl, stvarImpl, options, err, epsilon);
      return res;
    }
  }
  /// @}
  ///@name check methods
  /// @{
  [[nodiscard]] auto check_invariant() const -> bool { return impl_->check_invariant(); }
  [[nodiscard]] auto isnan() const -> bool { return impl_->is_nan(); }
  [[nodiscard]] auto isinf() const -> bool { return impl_->is_inf(); }
  auto detach_() -> void { impl_->detach_(); }
  /// @}
  /// @name friend function
  friend auto swap(ShareVar &lhs, ShareVar &rhs) noexcept -> void { lhs.impl_.swap(rhs.impl_); }
  friend auto operator<<(std::ostream &os, const ShareVar &self) -> std::ostream & {
    return os << *(self.impl_);
  }
  friend auto operator*(data_t scalar, const ShareVar &self) -> ShareVar { return self * scalar; }
  friend auto allclose(const ShareVar &lhs, const ShareVar &rhs) -> bool {
    return allclose(lhs.GetShareVarImpl(), rhs.GetShareVarImpl());
  }
}; // class ShareVar

// 完美转发
template <typename... Args>
inline auto make_ShareVar(Args &&...args) -> ShareVar {
  return ShareVar::create(std::forward<Args>(args)...);
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

} // namespace umat::core

#endif // CORE_SHAREVAR_H
