#pragma once

#include "TypeMap.h"
#include "base_config.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/optim/optimizer.h"
#include "torch/torch.h"
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>

namespace umat::utils {
template <typename T>
auto get_value(const T &value) -> utils::data_t {
  using U = std::decay_t<decltype(value)>;
  if constexpr (std::is_same_v<U, utils::data_t>) {
    return value;
  } else {
    return value.template item<utils::data_t>();
  }
}
template <typename T>
auto get_tensor(const T &value) -> torch::Tensor {
  using U = std::decay_t<decltype(value)>;
  if constexpr (std::is_same_v<U, utils::data_t>) {
    return torch::tensor(value);
  } else {
    return value;
  }
}
// 1. overloaded 通用组合器（C++17 必备）
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts &&...) -> overloaded<Ts...>;
//
template <typename... Vars>
constexpr auto get_first_ErrorCode(Vars &&...vars) -> std::optional<ErrorCode> {
  std::optional<ErrorCode> err = {};
  // 折叠表达式：遍历所有 variant，找到第一个 ErrorCode 立即返回
  (void)((std::holds_alternative<ErrorCode>(vars)
              ? (err = std::get<ErrorCode>(std::forward<Vars>(vars)), true)
              : false) ||
         ...);
  return err;
}
template <typename... Vars>
constexpr auto has_any_monostate(Vars &&...vars) -> bool {
  return (std::holds_alternative<std::monostate>(std::forward<Vars>(vars)) || ...);
}
template <typename Target, typename... Vars>
constexpr auto has_any_type(Vars &&...vars) -> bool {
  return (std::holds_alternative<Target>(std::forward<Vars>(vars)) || ...);
}
//
template <typename... Vs>
constexpr bool has_ErrorCode = (... || std::is_same_v<std::decay_t<Vs>, ErrorCode>);
template <typename... Vs>
constexpr bool has_monostate = (... || std::is_same_v<std::decay_t<Vs>, std::monostate>);
// 严格匹配
template <typename Target, typename Variant>
struct variant_contains_type : std::false_type {};
template <typename Target, typename... Vs>
struct variant_contains_type<Target, std::variant<Vs...>>
    : std::disjunction<std::is_same<Target, Vs>...> {};

template <typename Target, typename Variant>
constexpr bool variant_contains_type_v =
    variant_contains_type<Target, std::decay_t<Variant>>::value;
template <typename Target, typename... Variants>
constexpr bool all_variants_contain_type_v = (... && variant_contains_type_v<Target, Variants>);
//
template <typename Target, typename Variant>
struct variant_contains_type_decay;
template <typename Target, typename... Vs>
struct variant_contains_type_decay<Target, std::variant<Vs...>> {
  static constexpr bool value = (... || std::is_same_v<std::decay_t<Target>, std::decay_t<Vs>>);
};
template <typename Target, typename Variant>
constexpr bool variant_contains_type_decay_v = variant_contains_type_decay<Target, Variant>::value;
/**
 * @brief: 从类型中移除指定类型
 * @tparam T
 * @tparam List
 *
 **/
template <typename T, typename Tuple>
struct is_in_tuple {
  static constexpr bool value = false;
};

template <typename T>
struct is_in_tuple<T, std::tuple<>> {
  static constexpr bool value = false;
};

template <typename T, typename First, typename... Rest>
struct is_in_tuple<T, std::tuple<First, Rest...>> {
  static constexpr bool value = std::is_same_v<std::decay_t<T>, std::decay_t<First>> ||
                                is_in_tuple<T, std::tuple<Rest...>>::value;
};
template <typename T, typename Tuple>
constexpr bool is_in_tuple_v = is_in_tuple<T, Tuple>::value;

// ===================== 移除 variant 中的指定类型 =====================
// 辅助模板：过滤类型列表
template <typename ToRemoveTuple, typename TypeList>
struct filter_types_impl;

template <typename ToRemoveTuple, typename... Ts>
struct filter_types_impl<ToRemoveTuple, std::tuple<Ts...>> {
  // 检查类型是否应该保留
  template <typename T>
  static constexpr bool should_keep = !is_in_tuple_v<T, ToRemoveTuple>;

  // 递归收集保留的类型
  template <typename... KeepTs>
  struct collector;

  // 基础情况：空列表
  template <typename... KeepTs>
  struct collector<std::tuple<>, KeepTs...> {
    using type = std::tuple<KeepTs...>;
  };

  // 递归情况
  template <typename First, typename... Rest, typename... KeepTs>
  struct collector<std::tuple<First, Rest...>, KeepTs...> {
    using type =
        typename std::conditional_t<should_keep<First>,
                                    typename collector<std::tuple<Rest...>, KeepTs..., First>::type,
                                    typename collector<std::tuple<Rest...>, KeepTs...>::type>;
  };

  using type = typename collector<std::tuple<Ts...>>::type;
};

// 将tuple转换为variant
template <typename Tuple>
struct tuple_to_variant;

template <typename... Ts>
struct tuple_to_variant<std::tuple<Ts...>> {
  using type = std::variant<Ts...>;
};

// 主模板：移除variant中的指定类型
template <typename ToRemoveTuple, typename Variant>
struct remove_types_from_variant;

template <typename ToRemoveTuple, typename... Ts>
struct remove_types_from_variant<ToRemoveTuple, std::variant<Ts...>> {
  // 首先过滤类型
  using filtered_tuple = typename filter_types_impl<ToRemoveTuple, std::tuple<Ts...>>::type;
  // 然后将tuple转换为variant
  using type = typename tuple_to_variant<filtered_tuple>::type;
};

template <typename Variant, typename... ToRemove>
using remove_type_from_variant_t =
    typename remove_types_from_variant<std::tuple<ToRemove...>, Variant>::type;

template <typename... ToRemove, typename Variant>
auto strip_types(Variant var) {
  using StrippedVariant = remove_type_from_variant_t<std::decay_t<Variant>, ToRemove...>;
  return std::visit(
      [](auto &&val) -> StrippedVariant {
        using ValType = std::decay_t<decltype(val)>;
        if constexpr (is_in_tuple_v<ValType, std::tuple<ToRemove...>>) {
          // 理论上不会走到这，因为外层已经提前判断过
          return {};
        } else {
          return std::forward<decltype(val)>(val);
        }
      },
      var);
}
//
template <typename Ret, typename Args, typename Func>
auto VisitorWithError(Func &&funcs, Args &&args) -> Ret {
  static_assert(variant_contains_type_v<ErrorCode, Args>, "variant must has ErrorCode");
  static_assert(variant_contains_type_v<std::monostate, Args>, "variant must has std::monostate");
  return std::visit(overloaded{std::forward<Func>(funcs), [](ErrorCode err) -> Ret { return err; },
                               [](std::monostate) -> Ret { return ErrorCode::UnKnownError; }},
                    std::forward<Args>(args));
}
/**
 * @brief
 * @tparam Ret
 * @tparam Vars
 * @tparam Func
 * @param  func
 * @param  vars
 *
 * @return Ret
 **/
template <typename Ret, typename... Vars, typename Func>
auto Muti_VisitorWithError(Func &&func, Vars &&...vars) -> Ret {
  static_assert(all_variants_contain_type_v<ErrorCode, Vars...>, "variant must has ErrorCode");
  static_assert(all_variants_contain_type_v<std::monostate, Vars...>,
                "variant must has std::monostate");
  static_assert(variant_contains_type_v<ErrorCode, Ret>, "variant must has ErrorCode");
  static_assert(variant_contains_type_v<std::monostate, Ret>, "variant must has std::monostate");
  auto err = get_first_ErrorCode(std::forward<Vars>(vars)...);
  if (err.has_value()) {
    return *err;
  }
  // 第二步：检查所有variant是否包含monostate
  if (has_any_monostate(std::forward<Vars>(vars)...)) {
    return ErrorCode::UnKnownError;
  }

  auto get_safe_var = [&](auto &&vs) {
    using Original = std::decay_t<decltype(vs)>;
    using Safe = remove_type_from_variant_t<Original, ErrorCode, std::monostate>;
    auto safe_var = strip_types<ErrorCode, std::monostate>(std::forward<decltype(vs)>(vs));
    return safe_var;
  };
  return std::visit(
      overloaded{
          // 分支1：业务逻辑（用户传入的有效值处理）
          std::forward<Func>(func),
          //
      },
      get_safe_var(std::forward<Vars>(vars))...);
}
} // namespace umat::utils
