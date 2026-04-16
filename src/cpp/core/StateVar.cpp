#include "core/StateVar.h"
#include "c10/util/intrusive_ptr.h"
#include "core/ShareVar.h"
#include "core/StressTensor.h"
#include "core/impl/StateVarImpl.h"
#include "torch/headeronly/util/Exception.h"
#include "torch/optim/optimizer.h"
#include "utils/TypeMap.h"
#include "utils/base_config.h"
#include <utility>

namespace umat {

using namespace utils;
using namespace torch;
namespace core {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif

StateVar::StateVar(utils::data_t voidr, torch::Tensor alphaIni, utils::StressState state,
                   utils::data_t pnewdt)
    : impl_(c10::make_intrusive<impl::StateVarImpl>(voidr, alphaIni, state, pnewdt)) {}
StateVar::StateVar(utils::data_t voidr, StressTensor alphaIni, utils::data_t pnewdt)
    : impl_(c10::make_intrusive<impl::StateVarImpl>(voidr, alphaIni, pnewdt)) {}
auto StateVar::create(utils::data_t voidr, torch::Tensor alphaIni, utils::StressState state,
                      utils::data_t pnewdt) -> StateVar {
  STD_TORCH_CHECK(alphaIni.defined() == true,
                  "StateVar create failed: stress tensor is undefined (null)");
  auto impl = c10::make_intrusive<StateVarImpl>(voidr, alphaIni, state, pnewdt);
  return StateVar(std::move(impl));
}
auto StateVar::create(utils::data_t voidr, StressTensor alphaIni, utils::data_t pnewdt)
    -> StateVar {
  STD_TORCH_CHECK(alphaIni.is_valid(), "StateVar create failed: invalid alphaIni tensor");
  auto impl = c10::make_intrusive<StateVarImpl>(voidr, alphaIni, pnewdt);
  return StateVar(std::move(impl));
}

///@name ScopeGuard
/// @{
auto StateVar::backup_state() const -> StateVar { return clone(); }
auto StateVar::restore_state(StateVar &backup) -> void {
  // 检查自赋值
  if (this != &backup) {
    impl_ = std::move(backup.impl_);
    backup.reset();
  }
}
/// @}
auto StateVar::update_voidr(const StressTensor &depsln) -> void { impl_->update_voidr(depsln); }
auto StateVar::update_stvar(const core::ShareVar &shvar, const StressTensor &depsln) -> StateVar {
  auto &shvarImpl = shvar.GetShareVarImpl();
  auto impl = c10::make_intrusive<StateVarImpl>(impl_->update_statevarImpl(shvarImpl, depsln));
  return StateVar(std::move(impl));
}
///@name other methods
///@{
auto StateVar::clone() const -> StateVar {
  auto impl = c10::make_intrusive<StateVarImpl>(impl_->clone());
  return StateVar(std::move(impl));
}
auto StateVar::swap(StateVar &rhs) noexcept -> void { impl_.swap(rhs.impl_); }
auto StateVar::reset() -> void { impl_.reset(); }

///@}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
} // namespace core
} // namespace umat
