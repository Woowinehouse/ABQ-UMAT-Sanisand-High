#ifndef UTILS_MATERIAL_CONFIG_H
#define UTILS_MATERIAL_CONFIG_H
#include "utils/base_config.h"

namespace umat::utils {
// Material model configuration can be added here in the future
class mat {
  public:
  static constexpr data_t k = static_cast<double>(0.15);
  static constexpr data_t fm = static_cast<double>(0.05);
  static constexpr data_t fn = static_cast<double>(20.0);
  static constexpr data_t alphac = static_cast<double>(1.2);
  static constexpr data_t c = static_cast<double>(0.712);
  static constexpr data_t w = static_cast<double>(-0.25);
  static constexpr data_t voidref = static_cast<double>(0.934);
  static constexpr data_t lamdacs = static_cast<double>(0.0268);
  static constexpr data_t ksi = static_cast<double>(0.7);
  static constexpr data_t voidrl = static_cast<double>(0.15);
  static constexpr data_t lamdar = static_cast<double>(0.012);
  static constexpr data_t beta = static_cast<double>(22);
  static constexpr data_t y = static_cast<double>(0.3);
  static constexpr data_t pe = static_cast<double>(1);
  static constexpr data_t pr = static_cast<double>(1);
  static constexpr data_t nb = static_cast<double>(1.25);
  static constexpr data_t h0 = static_cast<double>(26.0);
  static constexpr data_t ch = static_cast<double>(13.0);
  static constexpr data_t ps = static_cast<double>(0.15);
  static constexpr data_t nu_min = static_cast<double>(0.15);
  static constexpr data_t nu_max = static_cast<double>(0.45);
  static constexpr data_t nu_v = static_cast<double>(0.5);
  static constexpr data_t nd = static_cast<double>(3.5);
  static constexpr data_t ad = static_cast<double>(0.6);
  static constexpr data_t pd = static_cast<double>(0.06);
  static constexpr data_t x = static_cast<double>(0.35);
  static constexpr data_t v = static_cast<double>(1000);
};

} // namespace umat::utils

#endif // UTILS_MATERIAL_CONFIG_H