#include "StateVarImpl.h"
#include "ATen/ops/sum.h"
#include "TensorOptions_ops.h"
#include "core/StressTensor.h"
#include <torch/optim/optimizer.h>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif
namespace umat {
using namespace utils;
using namespace torch;

namespace core::impl {

///@name constructors
///@{
auto StateVarImpl::update_voidr(const StressTensor &depsln) -> void {
  auto depsv = depsln->trace();
  voidr_ -= (1.0 + voidr_) * depsv.item<data_t>();
}
///@}

auto StateVarImpl::update_statevarImpl(const ShareVarImpl &shvarImpl,
                                       const core::StressTensor &depsln) -> StateVarImpl {
  auto depsv = depsln->trace().item<data_t>();
  auto updvoidr = voidr_ - (1.0 + voidr_) * depsv;

  auto updalphaIni = [&]() -> StressTensor {
    auto norm = LoadingDirection_::call<false>(shvarImpl);
    auto alpha = shvarImpl.get_alpha();
    auto alpha_alphaini = alpha - alphaIni_;
    auto temp = sum((*alpha_alphaini) * norm).item<data_t>();
    return temp < 0.0 ? alpha : alphaIni_;
  }();
  return {updvoidr, updalphaIni, pnewdt_};
}

} // namespace core::impl

} // namespace umat