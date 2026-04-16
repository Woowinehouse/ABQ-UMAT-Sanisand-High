#ifndef CORE_STATE_VAR_H
#define CORE_STATE_VAR_H
#include "core/impl/StateVarImpl.h"
#include "utils/export.h"

namespace umat::core {
class ShareVarTest;
class StateVar;
/**
 * @brief State variables container for UMAT (User Material) constitutive model.
 *
 * This class encapsulates internal state variables used in constitutive modeling,
 * including void ratio (e), initial back-stress (α_ini), and time increment scaling
 * parameter (pnewdt). These variables evolve during plastic deformation and affect
 * material hardening/softening behavior.
 *
 * The class uses intrusive pointer semantics for efficient memory management
 * and copy-on-write behavior through StateVarImpl.
 **/
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
  /**
   * @brief Get void ratio (e)
   * @return Current void ratio value
   */
  [[nodiscard]] auto get_voidr() const -> utils::data_t { return impl_->get_voidr(); }

  /**
   * @brief Get mutable reference to initial back-stress tensor (unsafe)
   * @warning This method bypasses copy-on-write and directly modifies the internal data
   * @return Mutable reference to initial back-stress tensor
   */
  [[nodiscard]] auto unsafe_get_alphaIni() -> core::StressTensor & {
    return impl_->unsafe_get_alphaIni();
  }

  /**
   * @brief Get const reference to initial back-stress tensor
   * @return Const reference to initial back-stress tensor
   */
  [[nodiscard]] auto get_alphaIni() const -> const core::StressTensor & {
    return impl_->get_alphaIni();
  }

  /**
   * @brief Get mutable reference to initial back-stress tensor as raw torch::Tensor (unsafe)
   * @warning This method bypasses copy-on-write and directly modifies the internal data
   * @return Mutable reference to initial back-stress tensor
   */
  [[nodiscard]] auto unsafe_get_alphaIni_tensor() -> Tensor & {
    return impl_->unsafe_get_alphaIni_tensor();
  }

  /**
   * @brief Get const reference to initial back-stress tensor as raw torch::Tensor
   * @return Const reference to initial back-stress tensor
   */
  [[nodiscard]] auto get_alphaIni_tensor() const -> const Tensor & {
    return impl_->get_alphaIni_tensor();
  }

  /**
   * @brief Get time increment scaling parameter (pnewdt)
   * @return Current pnewdt value (used for automatic time stepping)
   */
  [[nodiscard]] auto get_pnewdt() const -> utils::data_t { return impl_->get_pnewdt(); }

  /**
   * @brief Get const reference to underlying implementation
   * @return Const reference to StateVarImpl
   */
  [[nodiscard]] auto GetShareVarImpl() const -> const impl::StateVarImpl & { return *impl_.get(); }

  /**
   * @brief Get raw pointer to underlying implementation
   * @return Raw pointer to StateVarImpl
   */
  [[nodiscard]] auto data_ptr() const -> impl::StateVarImpl * { return impl_.get(); }
  /// @}
  ///@name Setters
  /// @{
  /**
   * @brief Set void ratio from internal state
   * Updates void ratio based on current stress state and plastic strain increment
   */
  auto set_voidr() -> void;

  /**
   * @brief Set initial back-stress from internal state
   * Updates initial back-stress based on current stress state
   */
  auto set_alphaIni() -> void;

  /**
   * @brief Update void ratio based on plastic strain increment
   * @param depsln Plastic strain increment tensor
   * Computes new void ratio: e_new = e_old + (1 + e_old) * trace(dε^p)
   */
  auto update_voidr(const StressTensor &depsln) -> void;

  /**
   * @brief Update state variables based on stress state and plastic strain
   * @param shvar Current shared variables (stress state)
   * @param depsln Plastic strain increment tensor
   * @return Updated StateVar object
   * Updates both void ratio and back-stress based on plastic deformation
   */
  auto update_stvar(const core::ShareVar &shvar, const StressTensor &depsln) -> StateVar;

  /**
   * @brief Set time increment scaling parameter (pnewdt)
   * @param new_pnewdt New pnewdt value
   * pnewdt < 1.0 suggests reducing time step for convergence
   */
  auto set_pnewdt(data_t new_pnewdt) const -> void { return impl_->set_pnewdt(new_pnewdt); }
  /// @}

  ///@name ScopeGuard
  /// @{
  /**
   * @brief Create a backup copy of current state
   * @return Backup StateVar object
   * Used for rollback operations if plasticity integration fails
   */
  [[nodiscard]] auto backup_state() const -> StateVar;

  /**
   * @brief Restore state from backup
   * @param stvars Backup StateVar object to restore from
   * Used for rollback operations if plasticity integration fails
   */
  auto restore_state(StateVar &stvars) -> void;
  /// @}

  ///@name other methods
  ///@{
  /**
   * @brief Create a shallow copy (lazy clone)
   * @return Shallow copy of StateVar
   * Shares underlying data until modification (copy-on-write)
   */
  [[nodiscard]] auto lazy_clone() const -> StateVar {
    return StateVar(c10::make_intrusive<impl::StateVarImpl>(impl_->lazy_clone()));
  }

  /**
   * @brief Create a deep copy
   * @return Deep copy of StateVar
   * Creates independent copy of all internal data
   */
  [[nodiscard]] auto clone() const -> StateVar;

  /**
   * @brief Swap contents with another StateVar
   * @param rhs Other StateVar to swap with
   */
  auto swap(StateVar &rhs) noexcept -> void;

  /**
   * @brief Reset to default state
   */
  auto reset() -> void;

  /**
   * @brief Validate internal state
   * @return True if state is valid, false otherwise
   * Checks for NaN/inf values and consistency of internal data
   */
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
