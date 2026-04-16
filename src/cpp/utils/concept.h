#ifndef UTILS_CONCEPT_H
#define UTILS_CONCEPT_H
#include "utils/base_config.h"
#include <torch/types.h>
namespace at {
class Tensor;
}
namespace umat::utils {
template <typename Dtype>
concept Scalartype = (std::is_same_v<std::decay_t<Dtype>, utils::data_t> ||
                      std::is_same_v<std::decay_t<Dtype>, torch::Tensor>);
}
#endif // UTILS_CONCEPT_H