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

/**
 * @brief Initialization manager for UMAT library
 *
 * This struct manages the initialization of external dependencies
 * (PyTorch, ABAQUS paths) and provides access to configuration paths.
 * Uses singleton-like pattern with static members for one-time initialization.
 **/
struct MYUMAT_API Initialize {
  inline static std::string datafile_path = ""; ///< Path to data file for material properties
  inline static std::string msgfile_path = "";  ///< Path to message file for ABAQUS output
  inline static bool has_initialize_abapath =
      false;                                       ///< Flag indicating ABAQUS paths are initialized
  inline static bool has_initialize_torch = false; ///< Flag indicating PyTorch is initialized

  /**
   * @brief Check if PyTorch is initialized
   * @return True if PyTorch is initialized, false otherwise
   */
  inline static auto torch_is_initialize() -> bool { return has_initialize_torch; }

  /**
   * @brief Check if ABAQUS paths are initialized
   * @return True if ABAQUS paths are initialized, false otherwise
   */
  inline static auto abapath_is_initialize() -> bool { return has_initialize_abapath; }

  /**
   * @brief Initialize PyTorch configuration
   * Sets up PyTorch backend, device selection, and random seed
   */
  static auto initialize_torch_config() -> void;

  /**
   * @brief Initialize ABAQUS path configuration
   * @param jobname ABAQUS job name (input from Fortran)
   * @param lenjobname Length of jobname string
   * @param outdir ABAQUS output directory (input from Fortran)
   * @param lenoutdir Length of outdir string
   * Sets up paths for data files and message files based on ABAQUS environment
   */
  static auto initialize_abapath_config(char *jobname, int *lenjobname, char *outdir,
                                        int *lenoutdir) -> void;

  /**
   * @brief Get message file path
   * @return Path to message file for ABAQUS output
   * @throws std::runtime_error if ABAQUS paths are not initialized
   */
  static auto get_msgfile_path() -> std::string {
    STD_TORCH_CHECK(abapath_is_initialize(),
                    "Can not get the msg file path beacause it has not initialzie");
    return msgfile_path;
  }
};

/**
 * @brief Check for NaN or infinite values in input
 * @tparam T Type of input (torch::Tensor or utils::data_t)
 * @param value Input value to check
 * @return ErrorCode::Success if no NaN/inf, ErrorCode::NanError if NaN found,
 *         ErrorCode::InfError if infinite value found
 *
 * This function performs compile-time type checking to handle both
 * tensor and scalar inputs appropriately.
 */
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
    // For other types, no check is performed
  }
  return ErrorCode::Success;
}
/**
 * @brief Check multiple inputs for NaN or infinite values
 * @tparam Args Variadic template parameter pack
 * @param args Input values to check
 * @return First error encountered (NaN or Inf), or ErrorCode::Success if all inputs are valid
 *
 * This function checks each argument in sequence and returns immediately
 * upon finding the first NaN or infinite value.
 */
template <typename... Args>
auto has_any_nan_inf(Args... args) -> ErrorCode {
  auto err = ErrorCode::Success;
  (..., (err == ErrorCode::Success ? (void)(err = has_nan_inf(args)) : (void)0));
  return err;
}
} // namespace umat::core

#endif // UTILS_PREPROCESS_H
