#ifndef UTILS_BASE_CONFIG_H
#define UTILS_BASE_CONFIG_H
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace umat::utils {
// 使用固定宽度类型确保跨平台一致性
using index_t = std::uint32_t;
using data_t = double; // IEEE 754双精度浮点数，通常为64位
using char_t = uint8_t;
// 定义常量
constexpr std::uint8_t BITS_PER_BYTE = 8; // 每个字节的位数
//
static bool torch_Initialized = false;
// define constant
constexpr index_t INDEX_MAX = std::numeric_limits<index_t>::max() >> 1; // 右移一位以避免溢出
constexpr index_t INDEX_MIN = 0;
constexpr data_t DATA_MAX = std::numeric_limits<data_t>::max();
constexpr data_t DATA_MIN = std::numeric_limits<data_t>::lowest(); // 使用lowest以获取最小负值
constexpr index_t index_zero = 0;
constexpr index_t index_two = 2;
constexpr data_t DATA_ONE = 1.0;
constexpr data_t PA = static_cast<data_t>(101.325);
constexpr data_t EXP = static_cast<data_t>(1e-12);
const data_t RAD23 = static_cast<data_t>(std::sqrt(2.0 / 3.0));
const data_t RAD32 = static_cast<data_t>(std::sqrt(3.0 / 2.0));
const data_t RAD6 = static_cast<data_t>(std::sqrt(6.0));
constexpr data_t FTOLR = static_cast<data_t>(1e-6);
// 静态断言确保类型符合预期
static_assert(sizeof(index_t) * BITS_PER_BYTE >= 32, "index_t must be at least 32 bits");
static_assert(std::is_unsigned<index_t>::value, "index_t must be an unsigned type");
static_assert(sizeof(data_t) * BITS_PER_BYTE >= 64, "data_t must be at least 64 bits");
static_assert(std::is_floating_point<data_t>::value, "data_t must be a floating point type");

// 其他通用配置和类型定义可以放在这里
constexpr index_t INDEX_BITS = sizeof(index_t) * BITS_PER_BYTE;
constexpr index_t INDEX_HALF_MAX = INDEX_MAX >> 1;     // 右移一位以获取半最大值
constexpr data_t EPSILON = static_cast<data_t>(1e-15); // 用于浮点比较的微小值
constexpr size_t gAlignment = 64;

} // namespace umat::utils

#endif // UTILS_BASE_CONFIG_H
