#ifndef CORE_STATE_VAR_H
#define CORE_STATE_VAR_H
#include "core/impl/StateVarImpl.h"
#include "utils/export.h"

namespace umat::core {
class ShareVarTest;
class StateVar;
class MYUMAT_API StateVar final {
  using StateVarImpl = impl::StateVarImpl;
  using Tensor = torch::Tensor;
  using data_t = utils::data_t;
  using State = utils::StressState;
  using StressTensor = core::StressTensor;

  protected:
  c10::intrusive_ptr<StateVarImpl> impl_;
  //
  private:
  StateVar() = default;

  public:
  explicit StateVar(c10::intrusive_ptr<impl::StateVarImpl> ptr) : impl_(std::move(ptr)) {}
  StateVar(utils::data_t voidr, torch::Tensor alphaIni, utils::StressState state,
           utils::data_t pnewdt);
  StateVar(utils::data_t voidr, StressTensor alphaIni, utils::data_t pnewdt);

  static auto create(utils::data_t voidr, torch::Tensor alphaIni, utils::StressState state,
                     utils::data_t pnewdt) -> StateVar;
  static auto create(utils::data_t voidr, StressTensor alphaIni, utils::data_t pnewdt) -> StateVar;
  ///@name Getters
  /// @{
  [[nodiscard]] auto get_voidr() const -> utils::data_t { return impl_->get_voidr(); }
  [[nodiscard]] auto unsafe_get_alphaIni() -> core::StressTensor & {
    return impl_->unsafe_get_alphaIni();
  }
  [[nodiscard]] auto get_alphaIni() const -> const core::StressTensor & {
    return impl_->get_alphaIni();
  }
  [[nodiscard]] auto unsafe_get_alphaIni_tensor() -> Tensor & {
    return impl_->unsafe_get_alphaIni_tensor();
  }
  [[nodiscard]] auto get_alphaIni_tensor() const -> const Tensor & {
    return impl_->get_alphaIni_tensor();
  }
  [[nodiscard]] auto get_pnewdt() const -> utils::data_t { return impl_->get_pnewdt(); }
  [[nodiscard]] auto GetShareVarImpl() const -> const impl::StateVarImpl & { return *impl_.get(); }
  [[nodiscard]] auto data_ptr() const -> impl::StateVarImpl * { return impl_.get(); }
  /// @}
  ///@name Setters
  /// @{
  auto set_voidr() -> void;
  auto set_alphaIni() -> void;
  auto update_voidr(const StressTensor &depsln) -> void;
  auto update_stvar(const core::ShareVar &shvar, const StressTensor &depsln) -> StateVar;
  auto set_pnewdt(data_t new_pnewdt) const -> void { return impl_->set_pnewdt(new_pnewdt); }
  /// @}

  ///@name ScopeGuard
  /// @{
  [[nodiscard]] auto backup_state() const -> StateVar;
  auto restore_state(StateVar &stvars) -> void;
  /// @}

  ///@name other methods
  ///@{
  [[nodiscard]] auto lazy_clone() const -> StateVar {
    return StateVar(c10::make_intrusive<impl::StateVarImpl>(impl_->lazy_clone()));
  }
  [[nodiscard]] auto clone() const -> StateVar;
  auto swap(StateVar &rhs) noexcept -> void;
  auto reset() -> void;
  [[nodiscard]] auto validate() const -> bool;

  ///@}
  friend auto operator<<(std::ostream &os, const StateVar &self) -> std::ostream & {
    return os << *(self.impl_);
  }
}; // class StateVar

template <typename... Args>
auto make_Statevar(Args... args) -> StateVar {
  return StateVar::create(std::forward<Args>(args)...);
}
} // namespace umat::core

#endif // CORE_STATE_VAR_H
