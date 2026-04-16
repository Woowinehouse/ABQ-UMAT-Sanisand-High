#ifndef CORE_STRESSTENSOR_H
#define CORE_STRESSTENSOR_H
#include "utils/TypeMap.h"
#include "utils/Visit.hpp"
#include "utils/base_config.h"
#include "utils/config.h"
#include "utils/export.h"
#include <span>
#include <torch/torch.h>
#include <utility>

namespace umat::core {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif
// forward declaration
class StressTensor;
namespace impl {
class ShareVarImpl;
class StateVarImpl;
} // namespace impl

// auxiliary function
MYUMAT_API inline auto check_tensor_dim(torch::Tensor tensor) noexcept -> bool {
  return tensor.defined() && tensor.dim() == 2 && tensor.size(0) == 3 && tensor.size(1) == 3;
}
constexpr inline auto is_valid_state(utils::StressState state) noexcept -> bool {
  return state != utils::StressState::Unknown;
}
MYUMAT_API inline auto has_same_state_(utils::StressState lhs, utils::StressState rhs) -> bool {
  return lhs == rhs;
}
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class MYUMAT_API StressTensor {
  // friendly classes
  friend class impl::ShareVarImpl;
  friend class impl::StateVarImpl;
  //
  using Tensor = torch::Tensor;
  using State = utils::StressState;
  using data_t = utils::data_t;

  // member
  private:
  Tensor tensor_; // 3x3应力张量
  State state_ = State::Unknown;
#ifdef __ENABLE_DEBUG__
  std::span<data_t> tensor_view_;
#endif
  // default
  StressTensor() = default;

  public:
  // default

  /// @name constructor
  ///@{

  explicit StressTensor(Tensor tensor, State stress_state)
      : tensor_(std::move(tensor)), state_(stress_state) {
#ifdef __ENABLE_DEBUG__
    tensor_view_ = std::span<data_t>(tensor_.data_ptr<data_t>(), (size_t)tensor_.numel());
#endif
  }
  explicit StressTensor(utils::data_t oneElt, State stress_state)
      : tensor_(torch::tensor({oneElt})), state_(stress_state) {
#ifdef __ENABLE_DEBUG__
    tensor_view_ = std::span<data_t>(tensor_.data_ptr<data_t>(), (size_t)tensor_.numel());
#endif
  }

  // @brief
  StressTensor(const StressTensor &other) = default;
  //
  StressTensor(StressTensor &&other) noexcept { swap(*this, other); }
  ///@}
  /// @brief deconstructor
  ~StressTensor() = default;

  [[nodiscard]] static auto create(Tensor tensor, State stress_state) -> StressTensor;
  [[nodiscard]] static auto create(utils::data_t oneElt, State stress_state) -> StressTensor;
  ///@name operator
  /// @{
  // @brief copy assignment
  auto operator=(StressTensor other) -> StressTensor & {
    swap(*this, other);
    return *this;
  }
  // @brief 访问符
  auto operator->() const -> const Tensor * { return &tensor_; }
  auto operator->() -> Tensor * { return &tensor_; }
  // @brief 解引用
  auto operator*() const -> const Tensor & { return tensor_; }
  auto operator*() -> Tensor & { return tensor_; }
  // unary operator
  auto operator-() const -> StressTensor { return minus(); }
  // binary operator
  auto operator+(StressTensor rhs) const -> StressTensor { return add(rhs); }
  auto operator-(StressTensor rhs) const -> StressTensor { return sub(rhs); }
  auto operator*(StressTensor rhs) const -> StressTensor { return mul(rhs); }
  auto operator/(StressTensor rhs) const -> StressTensor { return div(rhs); }
  // scalar
  auto operator*(data_t scalar) const -> StressTensor {
    auto new_tensor = tensor_ * scalar;
    return StressTensor(new_tensor, state_);
  }

  auto operator/(data_t scalar) const -> StressTensor {
    auto new_tensor = tensor_ / scalar;
    return StressTensor(new_tensor, state_);
  }
  //
  auto operator+=(StressTensor other) -> StressTensor & { return add_(other); }
  auto operator-=(StressTensor other) -> StressTensor & { return sub_(other); }
  auto operator*=(StressTensor other) -> StressTensor & { return mul_(other); }
  auto operator/=(StressTensor other) -> StressTensor & { return div_(other); }
  //
  auto operator[](int64_t idx) const -> Tensor { return tensor_[idx]; }
  [[nodiscard]] auto minus() const -> StressTensor;
  [[nodiscard]] auto add(const StressTensor &rhs, data_t scalar = 1.0) const -> StressTensor;
  [[nodiscard]] auto sub(const StressTensor &rhs, data_t scalar = 1.0) const -> StressTensor;
  [[nodiscard]] auto mul(const StressTensor &rhs, data_t scalar = 1.0) const -> StressTensor;
  [[nodiscard]] auto div(const StressTensor &rhs) const -> StressTensor;

  [[nodiscard]] auto add_(const StressTensor &other, data_t scalar = 1.0) -> StressTensor &;
  [[nodiscard]] auto sub_(const StressTensor &other, data_t scalar = 1.0) -> StressTensor &;
  [[nodiscard]] auto mul_(const StressTensor &other, data_t scalar = 1.0) -> StressTensor &;
  [[nodiscard]] auto div_(const StressTensor &other) -> StressTensor &;
  /// @}

  /// @name getters
  /// @{
  // 获取自定义元数据
  [[nodiscard]] auto get_data_ptr() const -> const void * { return tensor_.const_data_ptr(); }
  [[nodiscard]] auto mutable_get_data_ptr() const -> void * { return tensor_.data_ptr(); }
  [[nodiscard]] auto get_state() const noexcept -> State { return state_; }
  [[nodiscard]] auto unsafe_get_tensor() noexcept -> Tensor & { return tensor_; }
  [[nodiscard]] auto get_tensor() const noexcept -> const Tensor & { return tensor_; }
  [[nodiscard]] auto device() const noexcept -> torch::Device { return tensor_.device(); }
  [[nodiscard]] auto unsafe_tensor_ptr() noexcept -> Tensor * { return &tensor_; }
  [[nodiscard]] auto tensor_ptr() const noexcept -> const Tensor * { return &tensor_; }
  [[nodiscard]] auto tensor_count() const noexcept -> size_t { return tensor_.use_count(); }

  /// @}

  ///@name other methods
  /// @{
  auto reset() -> void {
    tensor_.reset();
    state_ = State::Unknown;
  }
  [[nodiscard]] auto clone() const -> StressTensor { return StressTensor(tensor_.clone(), state_); }
  [[nodiscard]] auto lazy_clone() const -> StressTensor {
    auto new_tensor = tensor_._lazy_clone();
    return StressTensor(new_tensor, state_);
  }
  [[nodiscard]] auto is_unique_tensor() const noexcept -> bool {
    return tensor_.is_uniquely_owned();
  }
  /// @}

  ///@name check methods
  /// @{
  [[nodiscard]] auto is_nan_tensor() const noexcept -> Tensor {
    return tensor_.defined() ? torch::isnan(tensor_) : Tensor(nullptr);
  }
  [[nodiscard]] auto is_inf_tensor() const noexcept -> Tensor {
    return tensor_.defined() ? torch::isinf(tensor_) : Tensor(nullptr);
  }
  [[nodiscard]] auto is_nan() const -> bool {
    return tensor_.defined() && is_nan_tensor().any().item<bool>();
  }
  [[nodiscard]] auto is_inf() const -> bool {
    return tensor_.defined() && is_inf_tensor().any().item<bool>();
  }
  [[nodiscard]] auto is_nan_inf() const -> bool { return is_nan() || is_inf(); }
  [[nodiscard]] auto is_valid() -> bool;
  auto validate() const -> void;
  auto validate_compatible(const StressTensor &other) const -> void;

  [[nodiscard]] auto check_device_compatible(const StressTensor &other) const noexcept {
    return !tensor_.defined() || !other.tensor_.defined() ||
           tensor_.device() == other.tensor_.device();
  }
  /// @}

  ///@name friend
  ///@{
  friend auto operator<<(std::ostream &os, const StressTensor &self) -> std::ostream & {
    os << self.tensor_;
    return os;
  }
  friend auto swap(StressTensor &lhs, StressTensor &rhs) noexcept -> void {
    using std::swap;
    swap(lhs.tensor_, rhs.tensor_);
    swap(lhs.state_, rhs.state_);
#ifdef __ENABLE_DEBUG__
    swap(lhs.tensor_view_, rhs.tensor_view_);
#endif
  }
  ///@}
}; // class StressTensor

template <typename... Args>
auto make_StressTensor(Args... args) -> StressTensor {
  return StressTensor::create(std::forward<Args>(args)...);
}

MYUMAT_API inline auto operator*(utils::data_t scalar, StressTensor tensor) -> StressTensor {
  return tensor * scalar;
}
MYUMAT_API inline auto allclose(const StressTensor &lhs, const StressTensor &rhs) -> bool {
  return allclose(*lhs, *rhs);
}

namespace detail {
/**
 * @brief
 * @tparam T
 * @param  arr
 * @param  state
 * @param  scalar
 *
 * @return StressTensor
 **/
template <typename Dtype>
inline auto make_StressTensor_core(Dtype *arr, utils::StressState state, double scalar = 1.0)
    -> StressTensor {

  TORCH_CHECK(arr != nullptr, "convert_array_to_StressTensor: input pointer cannot be null!");
  // 2. 优化：直接构造tensor（避免zeros+逐个赋值）
  c10::SmallVector<Dtype> tensor_data(9, 0.0); // 3x3 初始化为0
  // 根据应力状态选择
  switch (state) {
  case utils::StressState::PlaneStress:
    tensor_data[0] = arr[0];                              // (0,0)
    tensor_data[4] = arr[1];                              // (1,1)
    tensor_data[1] = arr[2] / static_cast<Dtype>(scalar); // (0,1)
    tensor_data[3] = tensor_data[1];                      // 对称 (1,0)
    break;
  case utils::StressState::PlaneStrain:
    tensor_data[0] = arr[0];                              // (0,0)
    tensor_data[4] = arr[1];                              // (1,1)
    tensor_data[8] = arr[2];                              // (2,2)
    tensor_data[1] = arr[3] / static_cast<Dtype>(scalar); // (0,1)
    tensor_data[3] = tensor_data[1];                      // 对称 (1,0)
    break;
  case utils::StressState::ThreeDStress:
    tensor_data[0] = arr[0];                              // (0,0)
    tensor_data[4] = arr[1];                              // (1,1)
    tensor_data[8] = arr[2];                              // (2,2)
    tensor_data[1] = arr[3] / static_cast<Dtype>(scalar); // (0,1)
    tensor_data[3] = tensor_data[1];                      // (1,0)
    tensor_data[2] = arr[4] / static_cast<Dtype>(scalar); // (0,2)
    tensor_data[6] = tensor_data[2];                      // (2,0)
    tensor_data[5] = arr[5] / static_cast<Dtype>(scalar); // (1,2)
    tensor_data[7] = tensor_data[5];                      // (2,1)
    break;
  default:
    STD_TORCH_CHECK(false, "make_StressTensor: unsupported state ",
                    utils::stress_state_to_string(state));
  }
#ifdef __ENABLE_DEBUG__
  auto tensor_data_view = std::span<Dtype>(tensor_data.data(), tensor_data.size());
#endif
  auto tensor = torch::from_blob(tensor_data.data(), {3, 3},
                                 utils::TorchTypeMap<Dtype>::type)
                    .clone(); // clone 避免依赖栈内存
  return StressTensor(tensor, state);
}
template <typename Dtype>
inline auto convert_tensor_to_array_core(Dtype *arr, int arr_size, const StressTensor &tensor,
                                         double scalar = 1.0) -> void {
  const auto state = tensor.get_state();
  const auto expected_size = static_cast<size_t>(state);
  TORCH_CHECK(arr != nullptr, "convert_tensor_to_array: output array pointer cannot be null");
  TORCH_CHECK(std::cmp_greater_equal(arr_size, expected_size),
              "convert_tensor_to_array: array size (", arr_size, ") < expected size for ",
              utils::stress_state_to_string(state), " (", expected_size, ")");
  if (tensor->dtype() != utils::TorchTypeMap<Dtype>::type) {
    throw std::runtime_error(std::string("Tensor type mismatch! Expected: ") +
                             torch::toString(utils::TorchTypeMap<Dtype>::type) +
                             ", Actual: " + torch::toString(tensor->dtype()));
  }
  //
  const auto &tensor_data = tensor.get_tensor();
  switch (tensor.get_state()) {
  case utils::StressState::PlaneStress:
    arr[0] = tensor_data[0][0].item<Dtype>();
    arr[1] = tensor_data[1][1].item<Dtype>();
    arr[2] = tensor_data[0][1].item<Dtype>() * static_cast<Dtype>(scalar);
    break;
  case utils::StressState::PlaneStrain:
    arr[0] = tensor_data[0][0].item<Dtype>();
    arr[1] = tensor_data[1][1].item<Dtype>();
    arr[2] = tensor_data[2][2].item<Dtype>();
    arr[3] = tensor_data[0][1].item<Dtype>() * static_cast<Dtype>(scalar);
    break;
  case utils::StressState::ThreeDStress:
    arr[0] = tensor_data[0][0].item<Dtype>();
    arr[1] = tensor_data[1][1].item<Dtype>();
    arr[2] = tensor_data[2][2].item<Dtype>();
    arr[3] = tensor_data[0][1].item<Dtype>() * static_cast<Dtype>(scalar);
    arr[4] = tensor_data[0][2].item<Dtype>() * static_cast<Dtype>(scalar);
    arr[5] = tensor_data[1][2].item<Dtype>() * static_cast<Dtype>(scalar);
    break;
  default:
    TORCH_CHECK(false, "convert_tensor_to_array: unsupported state ",
                utils::stress_state_to_string(state));
  }
}
template <typename Dtype>
inline auto convert_tensor4_to_array_core(Dtype *arr, int arr_size, const StressTensor &tensor4)
    -> void {
  TORCH_CHECK(arr != nullptr, "convert_tensor_to_array: output array pointer cannot be null");
  const auto state = tensor4.get_state();
  auto expected_size = static_cast<int>(state);
  TORCH_CHECK(std::cmp_greater_equal(arr_size * arr_size, expected_size * expected_size),
              "convert_tensor_to_array: array size (", arr_size, " x ", arr_size,
              ") < expected size for ", utils::stress_state_to_string(state), " (", expected_size,
              "x", expected_size, ")");
  auto &tensor = tensor4.get_tensor();
  auto [size1, size2] = [](int size) -> std::pair<std::vector<int>, std::vector<int>> {
    std::vector<int> size1, size2;
    switch (size) {
    case 3:
      size1 = {0, 1, 0};
      size2 = {0, 1, 1};
      break;
    case 4:
      size1 = {0, 1, 2, 0};
      size2 = {0, 1, 2, 1};
      break;
    case 6:
      size1 = {0, 1, 2, 0, 0, 1};
      size2 = {0, 1, 2, 1, 2, 2};
      break;
    default:
      TORCH_CHECK(false, "convert_tensor_to_array: unsupported size ", size);
    }
    return {size1, size2};
  }(expected_size);
  auto tensor2 = torch::zeros({expected_size, expected_size});
  for (int i = 0; i < size1.size(); i++) {
    auto i1 = size1[i];
    auto i2 = size2[i];
    for (int j = 0; j < size2.size(); j++) {
      auto j1 = size1[j];
      auto j2 = size2[j];
      tensor2[i][j] =
          static_cast<utils::data_t>(0.25) * (tensor[i1][i2][j1][j2] + tensor[i1][i2][j2][j1] +
                                              tensor[i2][i1][j1][j2] + tensor[i2][i1][j1][j2]);
    }
  }
#ifdef __ENABLE_DEBUG__
  auto tensor2_view = std::span<Dtype>(tensor2.data_ptr<Dtype>(), tensor2.numel());
#endif
  // 获取tensor2指针
  auto tensor_arr = tensor2.t().flatten(0).clone();
  auto tensor_arr_ptr = tensor_arr.data_ptr<Dtype>();
  // copy data
  memcpy(arr, tensor_arr_ptr, tensor2.numel() * sizeof(Dtype));
}
} // namespace detail
template <typename T>
inline auto make_StressTensor(T *arr, int size, double scalar = 1.0) -> StressTensor
  requires(utils::IsTorchScalarType<T>)
{
  utils::StressState state = utils::size_to_stress_state(size);
  return detail::make_StressTensor_core(arr, state, scalar);
}
template <typename T>
inline auto make_StressTensor(T *arr, int pos, int size, double scalar = 1.0)
  requires(utils::IsTorchScalarType<T>)
{
  utils::StressState state = utils::size_to_stress_state(size);
  return detail::make_StressTensor_core(arr + pos, state, scalar);
}
template <typename T>
inline auto make_StressTensor(T *arr, utils::StressState state, double scalar = 1.0) -> StressTensor
  requires(utils::IsTorchScalarType<T>)
{
  return detail::make_StressTensor_core(arr, state, scalar);
}
template <typename T>
inline auto make_StressTensor(T *arr, int pos, utils::StressState state, double scalar = 1.0)
    -> StressTensor
  requires(utils::IsTorchScalarType<T>)
{
  return detail::make_StressTensor_core(arr + pos, state, scalar);
}

template <typename T>
inline auto convert_tensor_to_array(T *arr, int offset, int arr_size, const StressTensor &tensor,
                                    T scalar = 1.0) -> void
  requires(utils::IsTorchScalarType<T>)
{
  detail::convert_tensor_to_array_core(arr + offset, arr_size - offset, tensor, scalar);
}
template <typename T>
inline auto convert_tensor_to_array(T *arr, int arr_size, const StressTensor &tensor,
                                    T scalar = 1.0) -> void
  requires(utils::IsTorchScalarType<T>)
{
  detail::convert_tensor_to_array_core(arr, arr_size, tensor, scalar);
}
template <typename Dtype>
inline auto convert_tensor4_to_array(Dtype *arr, int arr_size, const torch::Tensor &tensor4_val,
                                     utils::StressState state) -> void
  requires(utils::IsTorchScalarType<Dtype>)
{
  auto tensor4 = StressTensor(tensor4_val, state);
  return detail::convert_tensor4_to_array_core(arr, arr_size, tensor4);
}
template <typename Dtype>
inline auto convert_tensor4_to_array(Dtype *arr, int arr_size, const StressTensor &tensor4) -> void
  requires(utils::IsTorchScalarType<Dtype>)
{
  return detail::convert_tensor4_to_array_core(arr, arr_size, tensor4);
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif
} // namespace umat::core

#endif // CORE_STRESSTENSOR_H
