#ifndef CORE_IMPL_STATEVARIMPL_H
#define CORE_IMPL_STATEVARIMPL_H
#include "core/StressTensor.h"
#include "utils/base_config.h"
#include "utils/export.h"
#include <torch/torch.h>

namespace umat::core {
class StateVar;
namespace impl {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif
class StateVarImpl;
/**
 * @brief Implementation class for state variables in UMAT constitutive model
 *
 * This class encapsulates internal state variables that evolve during plastic deformation:
 * - voidr_: Void ratio (e) - porosity measure for soils
 * - alphaIni_: Initial back-stress tensor α_ini (kinematic hardening)
 * - pnewdt_: Time increment scaling parameter for automatic time stepping
 *
 * The class uses intrusive pointer semantics for efficient memory management
 * and provides methods for state variable updates during plasticity integration.
 */
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class MYUMAT_API StateVarImpl : public c10::intrusive_ptr_target {
  // friend class
  friend class core::StateVar;
  using Tensor = torch::Tensor;
  using data_t = utils::data_t;
  using state = utils::StressState;
  using StressTensor = core::StressTensor;

  private:
  // member
  data_t voidr_ = 0.0;             ///< Void ratio (e) - porosity measure for soils
  StressTensor alphaIni_;          ///< Initial back-stress tensor α_ini (kinematic hardening)
  mutable data_t pnewdt_ = 1.0e36; ///< Time increment scaling parameter for automatic time stepping
  //
  StateVarImpl() = default;

  public:
  ///@name constructors
  ///@{
  StateVarImpl(data_t voidr, StressTensor alphaIni, data_t pnewdt)
      : voidr_(voidr), alphaIni_(std::move(alphaIni)), pnewdt_(pnewdt) {}

  StateVarImpl(data_t voidr, Tensor alphaIni, state state, data_t pnewdt)
      : voidr_(voidr), alphaIni_(StressTensor(alphaIni, state)), pnewdt_(pnewdt) {}

  StateVarImpl(const StateVarImpl & /*other*/) = delete;
  StateVarImpl(StateVarImpl &&other) noexcept : StateVarImpl() { swap(*this, other); }
  ///@}
  ~StateVarImpl() { reset(); }
  ///@name operator
  /// @{
  auto operator=(const StateVarImpl &other) -> StateVarImpl & = delete;
  auto operator=(StateVarImpl other) && noexcept -> StateVarImpl & {
    swap(*this, other);
    return *this;
  }
  /// @}
  ///@name Getters
  /// @{
  /**
   * @brief Get void ratio (e)
   * @return Current void ratio value
   */
  [[nodiscard]] auto get_voidr() const -> utils::data_t { return voidr_; }

  /**
   * @brief Get mutable reference to initial back-stress tensor (unsafe)
   * @warning This method bypasses copy-on-write and directly modifies the internal data
   * @return Mutable reference to initial back-stress tensor
   */
  [[nodiscard]] auto unsafe_get_alphaIni() -> core::StressTensor & { return alphaIni_; }

  /**
   * @brief Get const reference to initial back-stress tensor
   * @return Const reference to initial back-stress tensor
   */
  [[nodiscard]] auto get_alphaIni() const -> const core::StressTensor & { return alphaIni_; }

  /**
   * @brief Get mutable reference to initial back-stress tensor as raw torch::Tensor (unsafe)
   * @warning This method bypasses copy-on-write
   * @return Mutable reference to initial back-stress tensor
   */
  [[nodiscard]] auto unsafe_get_alphaIni_tensor() -> Tensor & {
    return alphaIni_.unsafe_get_tensor();
  }

  /**
   * @brief Get const reference to initial back-stress tensor as raw torch::Tensor
   * @return Const reference to initial back-stress tensor
   */
  [[nodiscard]] auto get_alphaIni_tensor() const -> const Tensor & {
    return alphaIni_.get_tensor();
  }

  /**
   * @brief Get time increment scaling parameter (pnewdt)
   * @return Current pnewdt value (used for automatic time stepping)
   */
  [[nodiscard]] auto get_pnewdt() const -> utils::data_t { return pnewdt_; }
  /// @}
  ///@name Setters
  /// @{
  /**
   * @brief Set void ratio
   * @param new_voidr New void ratio value
   */
  auto set_voidr(data_t new_voidr) -> void { voidr_ = new_voidr; }

  /**
   * @brief Set initial back-stress tensor (swap semantics)
   * @param new_alpha_ini New initial back-stress tensor
   */
  auto set_alphaIni(StressTensor new_alpha_ini) -> void { swap(alphaIni_, new_alpha_ini); }

  /**
   * @brief Set time increment scaling parameter (pnewdt)
   * @param new_pnewdt New pnewdt value
   * pnewdt < 1.0 suggests reducing time step for convergence
   */
  auto set_pnewdt(data_t new_pnewdt) -> void { pnewdt_ = new_pnewdt; }
  /// @}
  /// @name Update
  ///@{
  /**
   * @brief Update void ratio based on plastic strain increment
   * @param depsln Plastic strain increment tensor
   * Computes new void ratio: e_new = e_old + (1 + e_old) * trace(dε^p)
   */
  auto update_voidr(const StressTensor &depsln) -> void;

  /**
   * @brief Update state variables based on stress state and plastic strain
   * @param shvarImpl Shared variables implementation (stress state)
   * @param depsln Plastic strain increment tensor
   * @return Updated StateVarImpl object
   * Updates both void ratio and back-stress based on plastic deformation
   */
  auto update_statevarImpl(const ShareVarImpl &shvarImpl, const core::StressTensor &depsln)
      -> StateVarImpl;
  ///@}
  ///@name other methods
  ///@{
  /**
   * @brief Create a shallow copy (lazy clone)
   * @return Shallow copy of StateVarImpl
   * Shares underlying data until modification (copy-on-write)
   */
  auto lazy_clone() const -> StateVarImpl { return {voidr_, alphaIni_.lazy_clone(), pnewdt_}; }

  /**
   * @brief Create a deep copy
   * @return Deep copy of StateVarImpl
   * Creates independent copy of all internal data
   */
  auto clone() const -> StateVarImpl { return {voidr_, alphaIni_.clone(), pnewdt_}; }

  /**
   * @brief Restore state from backup
   * @param backup Backup StateVarImpl object to restore from
   * Used for rollback operations if plasticity integration fails
   */
  auto restore_state(StateVarImpl backup) -> void { swap(*this, backup); }

  /**
   * @brief Reset to default state
   */
  auto reset() -> void {
    alphaIni_.reset();
    voidr_ = 0.0;
  }
  ///@}
  // friendly functions
  friend auto operator<<(std::ostream &os, const StateVarImpl &self) -> std::ostream & {
    os << "Void ratio: \n" << self.voidr_ << "\n";
    os << "Alpha initial: \n" << self.alphaIni_ << "\n";
    os << "pnewdt : \n" << self.pnewdt_ << "\n";
    return os;
  }
  friend auto swap(StateVarImpl &lhs, StateVarImpl &rhs) noexcept -> void {
    using std::swap;
    swap(lhs.voidr_, rhs.voidr_);
    swap(lhs.alphaIni_, rhs.alphaIni_);
    swap(lhs.pnewdt_, rhs.pnewdt_);
  }
};
} // namespace impl
} // namespace umat::core

#endif // CORE_IMPL_STATEVARIMPL_H
