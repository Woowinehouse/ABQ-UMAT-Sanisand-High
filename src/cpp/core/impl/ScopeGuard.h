#ifndef CORE_SCOPEGuard_H
#define CORE_SCOPEGuard_H
#include "functional"
#include "torch/torch.h"
#include "utils/export.h"

namespace umat::core::impl {
/**
 * @brief Tag for full object backup (backup all data)
 */
struct ObjectTag {}; ///< Backup all object data including tensors and metadata

/**
 * @brief Tag for gradient-only backup (backup only gradient information)
 */
struct GradTag {}; ///< Backup only gradient information for automatic differentiation

/**
 * @brief Implementation for full object backup
 * @tparam T Type of object to backup
 * @param tag ObjectTag for full backup
 * @param obj Reference to object to backup
 * @param backup_storage Optional storage for backup
 *
 * Saves complete state of object including all tensors and metadata.
 */
template <typename T>
auto backup_impl(ObjectTag, T &obj, c10::optional<T> &backup_storage) -> void {
  backup_storage = obj.backup_state();
}

/**
 * @brief Implementation for full object restore
 * @tparam T Type of object to restore
 * @param tag ObjectTag for full restore
 * @param obj Reference to object to restore
 * @param backup_storage Optional storage containing backup
 *
 * Restores object to previously saved state and clears backup storage.
 */
template <typename T>
auto restore_impl(ObjectTag, T &obj, c10::optional<T> &backup_storage) -> void {
  if (backup_storage.has_value()) {
    obj.restore_state(std::move(backup_storage));
    backup_storage.reset();
  }
}
/**
 * @brief Structure for storing tensor gradient state
 *
 * Used for gradient-only backup operations in automatic differentiation.
 */
struct TensorGradState {
  bool requires_grad{};              ///< Whether tensor requires gradient computation
  c10::optional<torch::Tensor> grad; ///< Optional gradient tensor
};
/**
 * @brief Implementation for gradient-only backup (stub implementation)
 * @tparam T Type of object
 * @param tag GradTag for gradient-only backup
 * @param obj Reference to object
 * @param backup_storage Optional storage for backup
 *
 * Currently a stub - gradient-only backup not fully implemented.
 */
template <typename T>
auto backup_impl(GradTag, T &obj, c10::optional<T> &backup_storage) {}

/**
 * @brief RAII scope guard for object state management
 *
 * Provides exception-safe rollback mechanism for object state changes.
 * Automatically restores object to original state when guard goes out of scope,
 * unless explicitly dismissed. Useful for plasticity integration algorithms
 * where operations may fail and require rollback.
 *
 * @tparam T Type of object to guard (must have backup_state() and restore_state() methods)
 */
template <typename T>
class ScopeGuard {
  private:
  T &obj_;                         ///< Reference to guarded object
  c10::optional<T> current_backup; ///< Optional backup of object state
  bool dismissed_ = false;         ///< Flag indicating guard has been dismissed

  public:
  ScopeGuard() = delete;

  /**
   * @brief Construct scope guard for object
   * @param obj Object to guard
   * @param backup_immediately Whether to backup immediately (default: true)
   */
  explicit ScopeGuard(T &obj, bool backup_immediately = true) : obj_(obj) {
    if (backup_immediately) {
      backup();
    }
  }

  /**
   * @brief Destructor - automatically restores object if not dismissed
   */
  ~ScopeGuard() { restore(); }

  /**
   * @brief Copy constructor (deleted to prevent duplicate restoration)
   */
  ScopeGuard(const ScopeGuard &) = delete;

  /**
   * @brief Copy assignment (deleted to prevent duplicate restoration)
   */
  auto operator=(const ScopeGuard &) -> ScopeGuard & = delete;

  /**
   * @brief Move constructor
   * @param other Other ScopeGuard to move from
   * Transfers ownership of backup and dismisses source guard
   */
  ScopeGuard(ScopeGuard &&other) noexcept
      : obj_(other.obj_), current_backup(std::move(other.current_backup)),
        dismissed_(other.dismissed_) {
    other.dismissed_ = true;
  }

  /**
   * @brief Restore object to backed-up state
   *
   * Only restores if guard hasn't been dismissed and backup exists.
   * Clears backup after restoration.
   */
  void restore() {
    if (!dismissed_ && current_backup.has_value()) {
      obj_.restore_state(std::move(current_backup.value()));
      current_backup.reset();
    }
  }

  /**
   * @brief Create new backup of current object state
   */
  void backup() { current_backup = obj_.backup_state(); }

  /**
   * @brief Dismiss the guard (prevent automatic restoration)
   */
  void dismiss() { dismissed_ = true; }

  /**
   * @brief Check if guard has a valid backup
   * @return True if backup exists, false otherwise
   */
  auto has_backup() const -> bool { return current_backup.has_value(); }

  /**
   * @brief Clear current backup without restoring
   */
  void clear_backup() { current_backup.reset(); }
};
/**
 * @brief Factory function for creating ScopeGuard
 * @tparam T Type of object
 * @param obj Object to guard
 * @param backup_immediately Whether to backup immediately (default: true)
 * @return ScopeGuard<T> for the object
 */
template <typename T>
auto make_ScopeGuard(T &obj, bool backup_immediately = true) -> ScopeGuard<T> {
  return ScopeGuard<T>(obj, backup_immediately);
}

/**
 * @brief Factory function for creating gradient-only guard (stub)
 * @tparam T Type of object
 * @param obj Object to guard
 * @param backup_immediately Whether to backup immediately (default: true)
 * @return ScopeGuard<T> for the object
 *
 * Currently returns empty guard - gradient-only functionality not implemented.
 */
template <typename T>
auto make_gradGuard(T &obj, bool backup_immediately = true) -> ScopeGuard<T> {
  return ScopeGuard<T>(obj, backup_immediately);
}
} // namespace umat::core::impl

#endif // CORE_SCOPEGuard_H
