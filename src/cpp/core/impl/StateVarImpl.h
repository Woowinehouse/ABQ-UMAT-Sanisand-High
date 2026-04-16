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
  data_t voidr_ = 0.0; // void ratio
  StressTensor alphaIni_;
  mutable data_t pnewdt_ = 1.0e36;
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
  [[nodiscard]] auto get_voidr() const -> utils::data_t { return voidr_; }
  [[nodiscard]] auto unsafe_get_alphaIni() -> core::StressTensor & { return alphaIni_; }
  [[nodiscard]] auto get_alphaIni() const -> const core::StressTensor & { return alphaIni_; }
  [[nodiscard]] auto unsafe_get_alphaIni_tensor() -> Tensor & {
    return alphaIni_.unsafe_get_tensor();
  }
  [[nodiscard]] auto get_alphaIni_tensor() const -> const Tensor & {
    return alphaIni_.get_tensor();
  }
  [[nodiscard]] auto get_pnewdt() const -> utils::data_t { return pnewdt_; }
  /// @}
  ///@name Setters
  /// @{
  auto set_voidr(data_t new_voidr) -> void { voidr_ = new_voidr; }
  auto set_alphaIni(StressTensor new_alpha_ini) -> void { swap(alphaIni_, new_alpha_ini); }
  auto set_pnewdt(data_t new_pnewdt) -> void { pnewdt_ = new_pnewdt; }
  /// @}
  /// @name Update
  ///@{
  auto update_voidr(const StressTensor &depsln) -> void;
  auto update_statevarImpl(const ShareVarImpl &shvarImpl, const core::StressTensor &depsln)
      -> StateVarImpl;
  ///@}
  ///@name other methods
  ///@{
  auto lazy_clone() const -> StateVarImpl { return {voidr_, alphaIni_.lazy_clone(), pnewdt_}; }
  auto clone() const -> StateVarImpl { return {voidr_, alphaIni_.clone(), pnewdt_}; }
  auto restore_state(StateVarImpl backup) -> void { swap(*this, backup); }
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
