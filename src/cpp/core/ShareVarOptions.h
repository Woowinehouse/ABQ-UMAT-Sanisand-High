#ifndef CORE_IMPL_SHAREVAROPTIONS_H
#define CORE_IMPL_SHAREVAROPTIONS_H
#include "utils/TypeMap.h"
#include "utils/base_config.h"
#include <torch/torch.h>
#include <utility>

namespace umat::core {

struct PlasticOptions {
  private:
  Tensor_type norm_ = std::monostate{};
  Scalar_Type cos3t_ = std::monostate{};
  Scalar_Type gtheta_ = std::monostate{};
  Scalar_Type psim_ = std::monostate{};
  Scalar_Type psim_alpha_ = std::monostate{};

  bool has_norm_ = false;
  bool has_cos3t_ = false;
  bool has_gtheta_ = false;
  bool has_psim_ = false;
  bool has_psim_alpha_ = false;

  public:
  PlasticOptions() = default;
  explicit PlasticOptions(Tensor_type norm, Scalar_Type cos3t, Scalar_Type gtheta, Scalar_Type psim,
                          Scalar_Type psim_alpha)
      : norm_(std::move(norm)), cos3t_(std::move(cos3t)), gtheta_(std::move(gtheta)),
        psim_(std::move(psim)), psim_alpha_(std::move(psim_alpha)) {}
  auto set_norm(Tensor_type norm) -> PlasticOptions & {
    if (std::holds_alternative<torch::Tensor>(norm)) {
      return *this;
    }
    this->norm_ = norm;
    this->has_norm_ = true;
    return *this;
  }
  auto set_cos3t(Scalar_Type cos3t) -> PlasticOptions & {
    if (std::holds_alternative<std::monostate>(cos3t)) {
      return *this;
    }
    this->cos3t_ = cos3t;
    this->has_cos3t_ = true;
    return *this;
  }
  auto set_gtheta(Scalar_Type gtheta) -> PlasticOptions & {
    if (std::holds_alternative<std::monostate>(gtheta)) {
      return *this;
    }
    this->gtheta_ = gtheta;
    this->has_gtheta_ = true;
    return *this;
  }
  auto set_psim(Scalar_Type psim) -> PlasticOptions & {
    if (std::holds_alternative<std::monostate>(psim)) {
      return *this;
    }
    this->psim_ = psim;
    this->has_psim_ = true;
    return *this;
  }
  auto set_psim_alpha(Scalar_Type psim_alpha) -> PlasticOptions & {
    if (std::holds_alternative<std::monostate>(psim_alpha)) {
      return *this;
    }
    this->psim_alpha_ = psim_alpha;
    this->has_psim_alpha_ = true;
    return *this;
  }
  //
  [[nodiscard]] auto get_norm() const -> Tensor_type { return norm_; }
  [[nodiscard]] auto get_cos3t() const -> Scalar_Type { return cos3t_; }
  [[nodiscard]] auto get_gtheta() const -> Scalar_Type { return gtheta_; }
  [[nodiscard]] auto get_psim() const -> Scalar_Type { return psim_; }
  //
  [[nodiscard]] auto has_norm() const -> bool { return has_norm_; }
  [[nodiscard]] auto has_cos3t() const -> bool { return has_cos3t_; }
  [[nodiscard]] auto has_gtheta() const -> bool { return has_gtheta_; }
  [[nodiscard]] auto has_psim() const -> bool { return has_psim_; }
  [[nodiscard]] auto has_psim_alpha() const -> bool { return has_psim_alpha_; }
};

inline auto make_ShareVarOptions(Tensor_type norm, Scalar_Type cos3t, Scalar_Type gtheta,
                                 Scalar_Type psim, Scalar_Type psim_alpha) -> PlasticOptions {
  return PlasticOptions(norm, cos3t, gtheta, psim, psim_alpha);
}
} // namespace umat::core

#endif // CORE_IMPL_SHAREVAROPTIONS_H