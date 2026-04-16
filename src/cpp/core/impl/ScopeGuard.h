#ifndef CORE_SCOPEGuard_H
#define CORE_SCOPEGuard_H
#include "functional"
#include "torch/torch.h"
#include "utils/export.h"

namespace umat::core::impl {
// tag
struct ObjectTag {}; // backup all data
struct GradTag {};   // just only backup grad

// backup all data
template <typename T>
auto backup_impl(ObjectTag, T &obj, c10::optional<T> &backup_storage) -> void {
  backup_storage = obj.backup_state();
}
template <typename T>
auto restore_impl(ObjectTag, T &obj, c10::optional<T> &backup_storage) -> void {
  if (backup_storage.has_value()) {
    obj.restore_state(std::move(backup_storage));
    backup_storage.reset();
  }
}
struct TensorGradState {
  bool requires_grad{};
  c10::optional<torch::Tensor> grad;
};
//
template <typename T>
auto backup_impl(GradTag, T &obj, c10::optional<T> &backup_storage) {}
template <typename T>
class ScopeGuard {
  private:
  T &obj_;
  c10::optional<T> current_backup;
  bool dismissed_ = false;

  public:
  ScopeGuard() = delete;
  explicit ScopeGuard(T &obj, bool backup_immediately = true) : obj_(obj) {
    if (backup_immediately) {
      backup();
    }
  }
  ~ScopeGuard() { restore(); }
  // 禁止拷贝，避免重复恢复
  ScopeGuard(const ScopeGuard &) = delete;
  auto operator=(const ScopeGuard &) -> ScopeGuard & = delete;
  // 允许移动（可选）
  ScopeGuard(ScopeGuard &&other) noexcept
      : obj_(other.obj_), current_backup(std::move(other.current_backup)),
        dismissed_(other.dismissed_) {
    other.dismissed_ = true;
  }
  // 恢复到最新备份状态（仅当有备份且未取消时执行）
  void restore() {
    if (!dismissed_ && current_backup.has_value()) {
      obj_.restore_state(std::move(current_backup.value()));
      current_backup.reset();
    }
  }
  void backup() { current_backup = obj_.backup_state(); }
  // 手动取消恢复（可选，按需使用）
  void dismiss() { dismissed_ = true; }
  // 检查是否有有效备份
  auto has_backup() const -> bool { return current_backup.has_value(); }
  // 清空当前备份（变为空状态）
  void clear_backup() { current_backup.reset(); }
};
template <typename T>
auto make_ScopeGuard(T &obj, bool backup_immediately = true) -> ScopeGuard<T> {
  return ScopeGuard<T>(obj, backup_immediately);
}
template <typename T>
auto make_gradGuard(T &obj, bool backup_immediately = true) -> ScopeGuard<T> {}
} // namespace umat::core::impl

#endif // CORE_SCOPEGuard_H
