/*****************************************************************************
 *  Project Name
 *  Copyright (C) 2026 Your Name <your.email@example.com>
 *
 *  This file is part of Project Name.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the
 *  "Software"), to deal in the Software without restriction, including
 *  without limitation the rights to use, copy, modify, merge, publish,
 *  distribute, sublicense, and/or sell copies of the Software, and to
 *  permit persons to whom the Software is furnished to do so, subject to
 *  the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included
 *  in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 *  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 *  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *  @file     ShareVarImpl.cpp
 *  @brief    简要说明
 *  @details  详细描述
 *
 *  @author   Your Name
 *  @email    your.email@example.com
 *  @version  1.0.0.1
 *  @date     2026/02/14
 *  @license  MIT License
 *---------------------------------------------------------------------------*
 *  Remark         : 说明备注
 *---------------------------------------------------------------------------*
 *  Change History :
 *  <Date>     | <Version> | <Author>       | <Description>
 *  2026/02/14 | 1.0.0.1   | Your Name      | Create file
 *****************************************************************************/
// implement header
#include "ShareVarImpl.h"
#include "core/StressTensor.h"

// utils header
#include "utils/TypeMap.h"
#include "utils/base_config.h"
// torch header
#include <ATen/ops/clamp.h>
#include <ATen/ops/equal.h>
#include <ATen/ops/linalg_eigh.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/norm.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/tensor.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/overloaded.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

// system include header
#include <cmath>
#include <torch/optim/optimizer.h>
#include <utility>

namespace umat {
using namespace utils;
using namespace torch;

namespace core::impl {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif
///@name constructors
///@{
ShareVarImpl::ShareVarImpl(StressTensor stress, StressTensor alpha, StressTensor p0)
    : stress_(std::move(stress)), alpha_(std::move(alpha)), p0_(std::move(p0)),
      is_lowstress(lowstress()) {}
ShareVarImpl::ShareVarImpl(Tensor stress, Tensor alpha, Tensor p0, State state)
    : stress_(StressTensor(std::move(stress), state).lazy_clone()),
      alpha_(StressTensor(std::move(alpha), state).lazy_clone()),
      p0_(StressTensor(std::move(p0), state).lazy_clone()), is_lowstress(lowstress()) {}
///@}

///@name operator
///@{
///@}
///@name Getters
/// @{
auto ShareVarImpl::create_shvar_with_dstress(core::StressTensor dstress) const
    -> core::impl::ShareVarImpl {
  auto new_var = clone();
  new_var.update_stress(dstress);
  return new_var;
}
auto ShareVarImpl::create_shvar_from_new_stress(const core::StressTensor &new_stress) const
    -> core::impl::ShareVarImpl {
  auto new_var = clone();
  new_var.set_stress(new_stress);
  return new_var;
}
/// @}
/// @name scope guard
/// @{
/// @brief
auto ShareVarImpl::minus() const -> ShareVarImpl { return {-stress_, -alpha_, -p0_}; }
auto ShareVarImpl::add(const ShareVarImpl &rhs, data_t scalar) const -> ShareVarImpl {
  return {stress_ + scalar * rhs.stress_, alpha_ + scalar * rhs.alpha_, p0_ + scalar * rhs.p0_};
}
auto ShareVarImpl::sub(const ShareVarImpl &rhs, data_t scalar) const -> ShareVarImpl {
  return {stress_ - scalar * rhs.stress_, alpha_ - scalar * rhs.alpha_, p0_ - scalar * rhs.p0_};
}
auto ShareVarImpl::mul(const ShareVarImpl &rhs, data_t scalar) const -> ShareVarImpl {
  return {stress_ * scalar * rhs.stress_, alpha_ * scalar * rhs.alpha_, p0_ * scalar * rhs.p0_};
}
auto ShareVarImpl::div(const ShareVarImpl &rhs) const -> ShareVarImpl {
  return {stress_ / rhs.stress_, alpha_ / rhs.alpha_, p0_ / rhs.p0_};
}
auto ShareVarImpl::mul_scalar(data_t scalar) const -> ShareVarImpl {
  return {scalar * stress_, scalar * alpha_, scalar * p0_};
}
auto ShareVarImpl::div_scalar(data_t scalar) const -> ShareVarImpl {
  return {stress_ / scalar, alpha_ / scalar, p0_ / scalar};
}
auto ShareVarImpl::add_(const ShareVarImpl &rhs, data_t scalar) -> ShareVarImpl & {
  stress_ += scalar * rhs.stress_;
  alpha_ += scalar * rhs.alpha_;
  p0_ += scalar * rhs.p0_;
  is_lowstress = is_low_stress();
  return *this;
}
auto ShareVarImpl::sub_(const ShareVarImpl &rhs, data_t scalar) -> ShareVarImpl & {
  stress_ -= scalar * rhs.stress_;
  alpha_ -= scalar * rhs.alpha_;
  p0_ -= scalar * rhs.p0_;
  is_lowstress = is_low_stress();
  return *this;
}
auto ShareVarImpl::mul_(const ShareVarImpl &rhs, data_t scalar) -> ShareVarImpl & {
  stress_ *= scalar * rhs.stress_;
  alpha_ *= scalar * rhs.alpha_;
  p0_ *= scalar * rhs.p0_;
  is_lowstress = is_low_stress();
  return *this;
}
auto ShareVarImpl::div_(const ShareVarImpl &rhs) -> ShareVarImpl & {
  stress_ /= rhs.stress_;
  alpha_ /= rhs.alpha_;
  p0_ /= rhs.p0_;
  is_lowstress = is_low_stress();
  return *this;
}
/// @}
///@name other methods
///@{
///@}
///@name elastic
///@}

/**
 * @brief
 *
 **/
auto ShareVarImpl::lowstress() -> bool {
  auto pressure = stress_->trace() / 3.0;
  return pressure.item<data_t>() < 1e-6 ? true : false;
}
auto ShareVarImpl::is_isotropic() const -> bool {
  auto eigen_results = torch::linalg_eigh(stress_.get_tensor(), "L");
  auto eigenvalues = std::get<0>(eigen_results);
  auto value1 = eigenvalues[0].item<data_t>();
  auto value2 = eigenvalues[1].item<data_t>();
  auto value3 = eigenvalues[2].item<data_t>();
  bool diff12 = (fabs(value1 - value2) < EXP);
  bool diff23 = (fabs(value2 - value3) < EXP);
  return diff12 && diff23;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
} // namespace core::impl

} // namespace umat
