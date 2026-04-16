#include "core/ShareVar.h"
#include "c10/util/Exception.h"
#include "c10/util/intrusive_ptr.h"
#include "core/StressTensor.h"
#include "core/impl/ShareVarImpl.h"
#include "torch/optim/optimizer.h"
#include "torch/torch.h"
#include "utils/TypeMap.h"
#include "utils/base_config.h"
#include <utility>

namespace umat::core {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif

///@name constructors
///@{
ShareVar::ShareVar(StressTensor stress, StressTensor alpha, StressTensor p0)
    : impl_(c10::make_intrusive<impl::ShareVarImpl>(stress, alpha, p0)) {}
ShareVar::ShareVar(torch::Tensor stress, torch::Tensor alpha, torch::Tensor p0,
                   utils::StressState state)
    : impl_(c10::make_intrusive<impl::ShareVarImpl>(stress, alpha, p0, state)) {}
///@}
/**
 * @brief
 * @param  stress
 * @param  alpha
 * @param  p0
 *
 * @return ShareVar
 **/
auto ShareVar::create(StressTensor stress, StressTensor alpha, StressTensor p0) -> ShareVar {
  TORCH_CHECK(stress.is_valid(), "ShareVar create failed: invalid stress tensor");
  TORCH_CHECK(alpha.is_valid(), "ShareVar create failed: invalid alpha tensor");
  auto impl = c10::make_intrusive<impl::ShareVarImpl>(stress, alpha, p0);
  return ShareVar(std::move(impl));
}
auto ShareVar::create(torch::Tensor stress, torch::Tensor alpha, torch::Tensor p0,
                      utils::StressState state) -> ShareVar {
  // 1. 非空校验
  TORCH_CHECK(stress.defined() == true,
              "ShareVar create failed: stress tensor is undefined (null)");
  TORCH_CHECK(alpha.defined() == true, "ShareVar create failed: alpha tensor is undefined (null)");
  TORCH_CHECK(p0.defined() == true, "ShareVar create failed: p0 tensor is undefined (null)");
  // 2. 非nan或inf校验
  // create pointer
  auto impl = c10::make_intrusive<impl::ShareVarImpl>(stress, alpha, p0, state);
  return ShareVar(std::move(impl));
}
///@name ScopeGuard
///@{
auto ShareVar::backup_state() const -> ShareVar {
  return ShareVar(c10::make_intrusive<impl::ShareVarImpl>(impl_->backup_state()));
}
auto ShareVar::restore_state(const ShareVar &backup) -> void {
  impl_->restore_state(std::move(*(backup.impl_)));
}
///@}
///@name Getters
///@{
///@}
/// @name Setters
///@{
auto ShareVar::update_shareVar(const ShareVar &dshvar) -> void { *this += dshvar; }
auto ShareVar::update_shareVar(const core::StressTensor &dstress, const core::StressTensor &dalpha,
                               const core::StressTensor &dp0) -> void {
  auto dshvar = ShareVar(dstress, dalpha, dp0);
  *this += dshvar;
}
auto ShareVar::update_shareVar(torch::Tensor dstress, torch::Tensor dalpha, torch::Tensor dp0,
                               utils::StressState state) -> void {
  auto dshvar = ShareVar(dstress, dalpha, dp0, state);
  *this += dshvar;
}
///@}

auto ShareVar::minus() const -> ShareVar {
  auto impl = c10::make_intrusive<ShareVarImpl>(-GetShareVarImpl());
  return ShareVar(std::move(impl));
}
auto ShareVar::add(const ShareVar &rhs, data_t scalar) const -> ShareVar {
  auto impl =
      c10::make_intrusive<impl::ShareVarImpl>(GetShareVarImpl() + scalar * rhs.GetShareVarImpl());
  return ShareVar(std::move(impl));
}
auto ShareVar::sub(const ShareVar &rhs, data_t scalar) const -> ShareVar {
  auto impl =
      c10::make_intrusive<impl::ShareVarImpl>(GetShareVarImpl() - scalar * rhs.GetShareVarImpl());
  return ShareVar(std::move(impl));
}
auto ShareVar::mul(const ShareVar &rhs, data_t scalar) const -> ShareVar {
  auto impl =
      c10::make_intrusive<impl::ShareVarImpl>(GetShareVarImpl() * scalar * rhs.GetShareVarImpl());
  return ShareVar(std::move(impl));
}
auto ShareVar::div(const ShareVar &rhs) const -> ShareVar {
  auto impl = c10::make_intrusive<impl::ShareVarImpl>(GetShareVarImpl() / rhs.GetShareVarImpl());
  return ShareVar(std::move(impl));
}
auto ShareVar::div_scalar(utils::data_t scalar) const -> ShareVar {
  auto impl = c10::make_intrusive<impl::ShareVarImpl>(GetShareVarImpl() / scalar);
  return ShareVar(std::move(impl));
}
auto ShareVar::add_(const ShareVar &other, data_t scalar) -> ShareVar & {
  unsafeGetShareVarImpl() += scalar * other.GetShareVarImpl();
  return *this;
}
auto ShareVar::sub_(const ShareVar &other, data_t scalar) -> ShareVar & {
  unsafeGetShareVarImpl() -= scalar * other.GetShareVarImpl();
  return *this;
}
auto ShareVar::mul_(const ShareVar &other, data_t scalar) -> ShareVar & {
  unsafeGetShareVarImpl() *= scalar * other.GetShareVarImpl();
  return *this;
}
auto ShareVar::div_(const ShareVar &other) -> ShareVar & {
  unsafeGetShareVarImpl() /= other.GetShareVarImpl();
  return *this;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

} // namespace umat::core
