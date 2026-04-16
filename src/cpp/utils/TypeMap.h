#ifndef UTILS_TYPEMAP_H
#define UTILS_TYPEMAP_H
#include "base_config.h"
#include <stdexcept>
#include <torch/types.h>
#include <type_traits>
// forward declaration
namespace at {
class Tensor;
}
namespace umat {

namespace utils {
// 辅助模板：将 C++ 类型映射到 PyTorch 张量类型
template <typename T>
struct TorchTypeMap;
// 特化：常用类型的映射
template <>
struct TorchTypeMap<float> {
  static constexpr auto type = torch::kFloat32;
};
template <>
struct TorchTypeMap<double> {
  static constexpr auto type = torch::kFloat64;
};
template <>
struct TorchTypeMap<int> {
  static constexpr auto type = torch::kInt32;
};
template <>
struct TorchTypeMap<int64_t> {
  static constexpr auto type = torch::kInt64;
};
template <>
struct TorchTypeMap<short> {
  static constexpr auto type = torch::kInt16;
};
template <>
struct TorchTypeMap<unsigned char> {
  static constexpr auto type = torch::kUInt8;
};

template <typename T>
constexpr bool IsTorchScalarType =
    std::is_same_v<std::remove_const_t<decltype(TorchTypeMap<T>::type)>, torch::ScalarType>;
/// @name StressState
/// @{
constexpr size_t STRESS_TENSOR_DIM = 3;   // 应力张量维度（3x3）
constexpr size_t PLANE_STRESS_SIZE = 3;   // 平面应力数组长度
constexpr size_t PLANE_STRAIN_SIZE = 4;   // 平面应变数组长度
constexpr size_t THREE_D_STRESS_SIZE = 6; // 三维应力数组长度

enum class StressState : size_t {
  PlaneStress = PLANE_STRESS_SIZE,    // 平面应力：size=3
  PlaneStrain = PLANE_STRAIN_SIZE,    // 平面应变：size=4
  ThreeDStress = THREE_D_STRESS_SIZE, // 三维应力：size=6
  Unknown = 0                         // 未知状态（非法size）
}; // enum class StressState
// 新增：枚举转字符串（便于日志/异常输出）
constexpr auto stress_state_to_string(StressState state) noexcept -> const char * {
  switch (state) {
  case StressState::PlaneStress:
    return "PlaneStress";
  case StressState::PlaneStrain:
    return "PlaneStrain";
  case StressState::ThreeDStress:
    return "ThreeDStress";
  default:
    return "Unknown";
  }
}
// 2. 辅助函数：根据size值转换为应力状态枚举（核心映射逻辑）
constexpr auto size_to_stress_state(size_t size) noexcept -> StressState {
  if (size == PLANE_STRESS_SIZE)
    return StressState::PlaneStress;
  if (size == PLANE_STRAIN_SIZE)
    return StressState::PlaneStrain;
  if (size == THREE_D_STRESS_SIZE)
    return StressState::ThreeDStress;
  return StressState::Unknown;
}

///@}
} // namespace utils
namespace core {
class ShareVar;
class StateVar;
class StressTensor;

} // namespace core
enum class ErrorCode : size_t {
  Success = 0,  // 成功
  NanError = 1, // Not A Number
  InfError = 2, // Infinity
  IosError = 3, // 各向同性错误
  ValueError = 4,
  TypeError = 5, //
  GradError = 6,
  IterError = 7,
  InputError = 8,
  UnKnownError = 9 // 未知错误
};
constexpr auto convert_errorcode_to_string(ErrorCode err) noexcept -> const char * {
  switch (err) {
  case ErrorCode::Success:
    return "Success";
  case ErrorCode::NanError:
    return "NanError";
  case ErrorCode::InfError:
    return "InfError";
  case ErrorCode::IosError:
    return "IosError";
  case ErrorCode::TypeError:
    return "TypeError";
  case ErrorCode::GradError:
    return "GradError";
  case ErrorCode::IterError:
    return "IterError";
  case ErrorCode::UnKnownError:
    return "UnKnownError";
  default:
    return " ";
  }
}
constexpr auto convert_ErrorCode_to_num(ErrorCode err) noexcept -> size_t {
  return static_cast<size_t>(err);
}

using pair_tensor = std::pair<torch::Tensor, torch::Tensor>;
using pair_data = std::pair<utils::data_t, utils::data_t>;
using pair_Scalar_Type = std::variant<std::monostate, pair_tensor, pair_data, ErrorCode>;
using pair_Tensor_Type = std::variant<std::monostate, pair_tensor, ErrorCode>;
using Tensor_type = std::variant<std::monostate, torch::Tensor, ErrorCode>;
using Stress_Typs = std::variant<core::ShareVar, torch::Tensor, core::StressTensor>;
using Voidr_Types = std::variant<core::StateVar, utils::data_t>;
using Share_Type = std::variant<std::monostate, core::ShareVar, ErrorCode>;
using Scalar_Type = std::variant<std::monostate, torch::Tensor, utils::data_t>;
} // namespace umat
#endif // UTILS_TYPEMAP_H