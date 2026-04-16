#ifndef CORE_IMPL_SHAREVARIMPL_H
#define CORE_IMPL_SHAREVARIMPL_H
// imple headers
#include "StateVarImpl.h"
#include "core/ShareVarOptions.h"
#include "core/StressTensor.h"

// utils headers
#include "utils/TypeMap.h"
#include "utils/base_config.h"
#include "utils/export.h"
#include <torch/torch.h>
#include <utility>

namespace umat::core {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif
class StateVar;

namespace impl {

class ShareVarImpl;
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class MYUMAT_API ShareVarImpl : public c10::intrusive_ptr_target {
  using Tensor = torch::Tensor;
  using data_t = utils::data_t;
  using State = utils::StressState;
  using StressTensor = core::StressTensor;

  private:
  // member
  StressTensor stress_;
  StressTensor alpha_;
  StressTensor p0_;
  bool is_lowstress = true;
  //
  ShareVarImpl() = default;

  public:
  ///@name constructors
  ///@{
  ShareVarImpl(StressTensor stress, StressTensor alpha, StressTensor p0);
  ShareVarImpl(Tensor stress, Tensor alpha, Tensor p0, State state);
  ShareVarImpl(const ShareVarImpl & /*other*/) = delete;
  ShareVarImpl(ShareVarImpl &&other) noexcept : ShareVarImpl() { swap(*this, other); }
  ///@}
  ~ShareVarImpl() = default;
  ///@name operators
  ///@{
  // @brief 赋值操作
  auto operator=(const ShareVarImpl &other) -> ShareVarImpl & = delete;
  auto operator=(ShareVarImpl other) && noexcept -> ShareVarImpl & {
    swap(*this, other);
    return *this;
  }
  // minus
  auto operator-() const -> ShareVarImpl { return minus(); }
  // add
  auto operator+(const ShareVarImpl &rhs) const -> ShareVarImpl { return add(rhs); }
  auto operator-(const ShareVarImpl &rhs) const -> ShareVarImpl { return sub(rhs); }
  auto operator*(const ShareVarImpl &rhs) const -> ShareVarImpl { return mul(rhs); }
  auto operator/(const ShareVarImpl &rhs) const -> ShareVarImpl { return div(rhs); }
  auto operator/(data_t scalar) const -> ShareVarImpl { return div_scalar(scalar); }
  auto operator*(data_t scalar) const -> ShareVarImpl { return mul_scalar(scalar); }
  // @brief 原地操作
  auto operator+=(const ShareVarImpl &other) -> ShareVarImpl & { return add_(other); }
  auto operator-=(const ShareVarImpl &other) -> ShareVarImpl & { return sub_(other); }
  auto operator*=(const ShareVarImpl &other) -> ShareVarImpl & { return mul_(other); }
  auto operator/=(const ShareVarImpl &other) -> ShareVarImpl & { return div_(other); }

  auto minus() const -> ShareVarImpl;
  auto add(const ShareVarImpl &rhs, data_t scalar = 1.0) const -> ShareVarImpl;
  auto sub(const ShareVarImpl &rhs, data_t scalar = 1.0) const -> ShareVarImpl;
  auto mul(const ShareVarImpl &rhs, data_t scalar = 1.0) const -> ShareVarImpl;
  auto div(const ShareVarImpl &rhs) const -> ShareVarImpl;
  auto mul_scalar(data_t scalar) const -> ShareVarImpl;
  auto div_scalar(data_t scalar) const -> ShareVarImpl;
  //
  auto add_(const ShareVarImpl &rhs, data_t scalar = 1.0) -> ShareVarImpl &;
  auto sub_(const ShareVarImpl &rhs, data_t scalar = 1.0) -> ShareVarImpl &;
  auto mul_(const ShareVarImpl &rhs, data_t scalar = 1.0) -> ShareVarImpl &;
  auto div_(const ShareVarImpl &rhs) -> ShareVarImpl &;
  //
  ///@}
  ///@name Getters
  ///@{
  /// @brief get the member class
  // get data ptr
  auto stress_data_ptr() const -> const void * { return stress_->const_data_ptr(); }
  auto alpha_data_ptr() const -> const void * { return alpha_->const_data_ptr(); }
  auto p0_data_ptr() const -> const void * { return p0_->const_data_ptr(); }
  // get const data_ptr
  auto mutable_stress_data_ptr() const -> void * { return stress_->data_ptr(); }
  auto mutable_alpha_data_ptr() const -> void * { return alpha_->data_ptr(); }
  auto mutable_p0_data_ptr() const -> void * { return p0_->data_ptr(); }

  auto get_stress() const noexcept -> const StressTensor & { return stress_; };
  auto get_alpha() const noexcept -> const StressTensor & { return alpha_; };
  auto get_p0() const noexcept -> const StressTensor & { return p0_; };
  auto unsafe_get_stress() -> StressTensor & { return stress_; };
  auto unsafe_get_alpha() -> StressTensor & { return alpha_; };
  auto unsafe_get_p0() -> StressTensor & { return p0_; };
  // get tensor
  auto get_stress_tensor() const noexcept -> const Tensor & { return stress_.get_tensor(); };
  auto get_alpha_tensor() const noexcept -> const Tensor & { return alpha_.get_tensor(); };
  auto get_p0_tensor() const noexcept -> const Tensor & { return p0_.get_tensor(); };
  auto unsafe_get_stress_tensor() noexcept -> Tensor & { return stress_.unsafe_get_tensor(); };
  auto unsafe_get_alpha_tensor() noexcept -> Tensor & { return alpha_.unsafe_get_tensor(); };
  auto unsafe_get_p0_tensor() noexcept -> Tensor & { return p0_.unsafe_get_tensor(); };

  // @brief get the member ptr
  auto stress_ptr() const noexcept -> const StressTensor * { return &stress_; }
  auto alpha_ptr() const noexcept -> const StressTensor * { return &alpha_; }
  auto p0_ptr() const noexcept -> const StressTensor * { return &p0_; }
  auto unsafe_stress_ptr() -> StressTensor * { return &stress_; }
  auto unsafe_alpha_ptr() -> StressTensor * { return &alpha_; }
  auto unsafe_p0_ptr() -> StressTensor * { return &p0_; }

  auto stress_tensor_ptr() const noexcept -> const Tensor * { return stress_.tensor_ptr(); }
  auto alpha_tensor_ptr() const noexcept -> const Tensor * { return alpha_.tensor_ptr(); }
  auto p0_tensor_ptr() const noexcept -> const Tensor * { return p0_.tensor_ptr(); }
  auto unsafe_stress_tensor_ptr() -> Tensor * { return stress_.unsafe_tensor_ptr(); }
  auto unsafe_alpha_tensor_ptr() -> Tensor * { return alpha_.unsafe_tensor_ptr(); }
  auto unsafe_p0_tensor_ptr() -> Tensor * { return p0_.unsafe_tensor_ptr(); }
  // use count
  auto get_stress_count() const -> size_t { return stress_.tensor_count(); }
  auto get_alpha_count() const -> size_t { return alpha_.tensor_count(); }
  auto get_p0_count() const -> size_t { return p0_.tensor_count(); }
  //
  auto is_low_stress() const -> bool { return is_lowstress; }
  auto get_state() const -> State { return stress_.get_state(); }
  ///@}
  ///@name Setters
  ///@{
  auto set_stress(StressTensor stress) -> void { swap(stress_, stress); };
  auto set_alpha(StressTensor alpha) -> void { swap(alpha_, alpha); };
  auto set_p0(StressTensor p0) -> void { swap(p0_, p0); };
  [[nodiscard]] auto create_shvar_with_dstress(StressTensor dstress) const
      -> core::impl::ShareVarImpl;
  [[nodiscard]] auto create_shvar_from_new_stress(const StressTensor &new_stress) const
      -> core::impl::ShareVarImpl;
  ///@}
  /// @name update
  /// @{
  auto update_stress(const StressTensor &dstress) -> void { stress_ += dstress; };
  auto update_alpha(const StressTensor &dalpha) -> void { alpha_ += dalpha; };
  auto update_p0(const StressTensor &dp0) -> void { p0_ += dp0; };
  auto update_shareVarImpl(const StressTensor &dstress, const StressTensor &dalpha,
                           const StressTensor &dp0) -> void {
    stress_ += dstress;
    alpha_ += dalpha;
    p0_ += dp0;
  }
  auto update_shareVarImpl(const torch::Tensor &dstress, const torch::Tensor &dalpha,
                           const torch::Tensor &dp0, utils::StressState state) -> void {
    STD_TORCH_CHECK(has_same_state_(get_state(), state),
                    "update_shareVarImpl: state mismatch! Expected: ",
                    utils::stress_state_to_string(get_state()),
                    ", Actual: ", utils::stress_state_to_string(state));

    stress_.unsafe_get_tensor() += dstress;
    alpha_.unsafe_get_tensor() += dalpha;
    p0_.unsafe_get_tensor() += dp0;
  }
  /// @}

  /// @name scope guard
  /// @{
  /// @brief
  auto backup_state() const -> ShareVarImpl { return clone(); }
  auto restore_state(ShareVarImpl backup) -> void { swap(*this, backup); }
  /// @}

  ///@name other methods
  /// @{
  //@brief
  auto reset() -> void {
    stress_.reset();
    alpha_.reset();
    p0_.reset();
    is_lowstress = true;
  }
  auto clone() const -> ShareVarImpl { return {stress_.clone(), alpha_.clone(), p0_.clone()}; }
  auto lazy_clone() const -> ShareVarImpl {
    return {stress_.lazy_clone(), alpha_.lazy_clone(), p0_.lazy_clone()};
  }
  auto detach_() -> void {
    stress_->detach_();
    alpha_->detach_();
    p0_->detach_();
  }
  /// @}
  ///@name elastic
  ///@{
  ///@}

  ///@name check methods
  /// @{
  auto validate() const -> void {
    stress_.validate();
    alpha_.validate();
    p0_.validate();
  }
  /// @brief check all member whether has nan of inf
  // stress
  auto stress_is_nan_tensor() const noexcept -> Tensor { return stress_.is_nan_tensor(); }
  auto stress_is_inf_tensor() const noexcept -> Tensor { return alpha_.is_inf_tensor(); }
  auto stress_is_nan() const noexcept -> bool { return stress_.is_nan(); }
  auto stress_is_inf() const noexcept -> bool { return stress_.is_inf(); }
  auto stress_is_nan_inf() const noexcept -> bool { return stress_.is_nan_inf(); }
  // alpha
  auto alpha_is_nan_tensor() const noexcept -> Tensor { return alpha_.is_nan_tensor(); }
  auto alpha_is_inf_tensor() const noexcept -> Tensor { return alpha_.is_inf_tensor(); }
  auto alpha_is_nan() const noexcept -> bool { return alpha_.is_nan(); }
  auto alpha_is_inf() const noexcept -> bool { return alpha_.is_inf(); }
  auto alpha_is_nan_inf() const noexcept -> bool { return alpha_.is_nan_inf(); }
  // p0
  auto p0_is_nan_tensor() const noexcept -> Tensor { return p0_.is_nan_tensor(); }
  auto p0_is_inf_tensor() const noexcept -> Tensor { return p0_.is_inf_tensor(); }
  auto p0_is_nan() const noexcept -> bool { return p0_.is_nan(); }
  auto p0_is_inf() const noexcept -> bool { return p0_.is_inf(); }
  auto p0_is_nan_inf() const noexcept -> bool { return p0_.is_nan_inf(); }
  // all member
  auto is_nan() const -> bool { return stress_is_nan() || alpha_is_nan() || p0_is_nan(); }
  auto is_inf() const -> bool { return stress_is_inf() || alpha_is_inf() || p0_is_inf(); }
  auto check_invariant() -> bool { return !is_nan() && !is_inf(); }
  //
  auto is_isotropic() const -> bool;
  /// @}

  private:
  auto lowstress() -> bool;

  friend auto operator<<(std::ostream &os, const ShareVarImpl &self) -> std::ostream & {
    os << "Stress: \n" << self.stress_ << "\n";
    os << "Alpha: \n" << self.alpha_ << "\n";
    os << "p0: \n" << self.p0_ << "\n";
    return os;
  }
  friend auto swap(ShareVarImpl &lhs, ShareVarImpl &rhs) noexcept -> void {
    using std::swap;
    swap(lhs.stress_, rhs.stress_);
    swap(lhs.alpha_, rhs.alpha_);
    swap(lhs.p0_, rhs.p0_);
    swap(lhs.is_lowstress, rhs.is_lowstress);
  }
  friend auto operator*(data_t scalar, const ShareVarImpl &rhs) -> ShareVarImpl {
    return rhs.mul_scalar(scalar);
  }
  friend auto allclose(const ShareVarImpl &lhs, const ShareVarImpl &rhs) -> bool {
    return allclose(lhs.get_stress(), rhs.get_stress()) &&
           allclose(lhs.get_alpha(), rhs.get_alpha()) && allclose(lhs.get_p0(), rhs.get_p0());
  }
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif
} // namespace impl
} // namespace umat::core

#endif // CORE_IMPL_SHAREVARIMPL_H
