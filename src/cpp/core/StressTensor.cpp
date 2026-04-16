#include "StressTensor.h"
#include "ATen/ops/isinf.h"
#include "ATen/ops/isnan.h"
#include "torch/headeronly/util/Exception.h"
#include "torch/optim/optimizer.h"
#include "utils/TypeMap.h"
#include "utils/base_config.h"
#include <stdexcept>
#include <string>
#include <utility>

namespace umat::core {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif
///@name constructors
///@{
auto StressTensor::create(torch::Tensor tensor, utils::StressState stress_state) -> StressTensor {
  STD_TORCH_CHECK(is_valid_state(stress_state), " stress state cannot be Unknown");
  if (torch::isnan(tensor).any().item<bool>()) {
    throw std::invalid_argument("StressTensor::create: input tensor contains NaN values");
  }
  if (torch::isinf(tensor).any().item<bool>()) {
    throw std::invalid_argument("StressTensor::create: input tensor contains Inf values");
  }
#ifdef __ENABLE_DEBUG__
  STD_TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
#endif

  return StressTensor(tensor, stress_state);
}
auto StressTensor::create(utils::data_t oneElt, utils::StressState stress_state) -> StressTensor {
  return StressTensor(oneElt, stress_state);
}
///@}
///@name operator
///@{
auto StressTensor::minus() const -> StressTensor { return StressTensor(-tensor_, state_); }
auto StressTensor::add(const StressTensor &rhs, data_t scalar) const -> StressTensor {
  validate_compatible(rhs);
  auto new_tensor = tensor_ + scalar * rhs.tensor_;
  return StressTensor(std::move(new_tensor), state_);
}
auto StressTensor::sub(const StressTensor &rhs, data_t scalar) const -> StressTensor {
  validate_compatible(rhs);
  auto new_tensor = tensor_ - scalar * rhs.tensor_;
  return StressTensor(std::move(new_tensor), state_);
}
auto StressTensor::mul(const StressTensor &rhs, data_t scalar) const -> StressTensor {
  validate_compatible(rhs);
  auto new_tensor = tensor_ * scalar * rhs.tensor_;
  return StressTensor(std::move(new_tensor), state_);
}
auto StressTensor::div(const StressTensor &rhs) const -> StressTensor {
  validate_compatible(rhs);
  auto new_tensor = tensor_ / rhs.tensor_;
  return StressTensor(std::move(new_tensor), state_);
}
auto StressTensor::add_(const StressTensor &other, data_t scalar) -> StressTensor & {
  validate_compatible(other);
  tensor_ += scalar * other.tensor_;
  return *this;
}
auto StressTensor::sub_(const StressTensor &other, data_t scalar) -> StressTensor & {
  validate_compatible(other);
  tensor_ -= scalar * other.tensor_;
  return *this;
}
auto StressTensor::mul_(const StressTensor &other, data_t scalar) -> StressTensor & {
  validate_compatible(other);
  tensor_ *= scalar * other.tensor_;
  return *this;
}
auto StressTensor::div_(const StressTensor &other) -> StressTensor & {
  validate_compatible(other);
  tensor_ /= other.tensor_;
  return *this;
}
///@}

///@name other methods
///@{
///@}
auto StressTensor::is_valid() -> bool {
  return tensor_.defined() && state_ != utils::StressState::Unknown && check_tensor_dim(tensor_) &&
         !is_nan_inf();
}
auto StressTensor::validate() const -> void {
  // check the stress state
  STD_TORCH_CHECK(is_valid_state(state_), " stress state cannot be Unknown");
  // 检查tensor定义
  STD_TORCH_CHECK(tensor_.defined(), "StressTensor validation failed: tensor is undefined");
  // 检查tensor维度
  // STD_TORCH_CHECK(check_tensor_dim(tensor_),
  //                 "StressTensor validation failed: expected 3x3 tensor, got " +
  //                     std::to_string(tensor_.dim()) + "D tensor with size " +
  //                     std::to_string(tensor_.size(0)) + "x" + std::to_string(tensor_.size(1)));
  // 4. 检查NaN/Inf
  if (is_nan()) {
    throw std::runtime_error("StressTensor validation failed: contains NaN values");
  }
  if (is_inf()) {
    throw std::runtime_error("StressTensor validation failed: contains Inf values");
  }
}
auto StressTensor::validate_compatible(const StressTensor &other) const -> void {
  validate();
  other.validate();

  if (this->state_ != other.state_) {
    throw std::invalid_argument("StressTensor +=: state mismatch - current(" +
                                std::to_string(static_cast<int>(this->state_)) + ") vs other(" +
                                std::to_string(static_cast<int>(other.state_)) + ")");
  }
  // 3. 检查设备一致
  if (!this->check_device_compatible(other)) {
    throw std::invalid_argument("StressTensor compatibility check failed: device mismatch - " +
                                this->get_tensor().device().str() + " vs " +
                                other.get_tensor().device().str());
  }
}

} // namespace umat::core
