#ifndef UTILS_PREPROCESS_H
#define UTILS_PREPROCESS_H
#include "utils/TypeMap.h"
#include "utils/base_config.h"
#include "utils/config.h"
#include "utils/export.h"
#include <torch/torch.h>

namespace umat::core {
static std::once_flag init_flag;
static int initialization_error = 0;

struct MYUMAT_API Initialize {
  inline static std::string datafile_path = "";
  inline static std::string msgfile_path = "";
  inline static bool has_initialize_abapath = false;
  inline static bool has_initialize_torch = false;
  inline static auto torch_is_initialize() -> bool { return has_initialize_torch; }
  inline static auto abapath_is_initialize() -> bool { return has_initialize_abapath; }
  static auto initialize_torch_config() -> void;
  static auto initialize_abapath_config(char *jobname, int *lenjobname, char *outdir,
                                        int *lenoutdir) -> void;
  static auto get_msgfile_path() -> std::string {
    STD_TORCH_CHECK(abapath_is_initialize(),
                    "Can not get the msg file path beacause it has not initialzie");
    return msgfile_path;
  }
};

template <typename T>
auto has_nan_inf(const T &value) -> ErrorCode {

  if constexpr (std::is_same_v<T, torch::Tensor>) {
    const bool has_nan = torch::isnan(value).any().template item<bool>();
    const bool has_inf = torch::isinf(value).any().template item<bool>();

    if (has_nan)
      return ErrorCode::NanError;
    if (has_inf)
      return ErrorCode::InfError;

  } else if constexpr (std::is_same_v<T, utils::data_t>) {
    if (std::isnan(value))
      return ErrorCode::NanError;
    if (std::isinf(value))
      return ErrorCode::InfError;
  } else {
  }
  return ErrorCode::Success;
}
template <typename... Args>
auto has_any_nan_inf(Args... args) -> ErrorCode {
  auto err = ErrorCode::Success;
  (..., (err == ErrorCode::Success ? (void)(err = has_nan_inf(args)) : (void)0));
  return err;
}
} // namespace umat::core

#endif // UTILS_PREPROCESS_H
