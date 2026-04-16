#include "core/auxiliary.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarTypeToTypeMeta.h"
#include "torch/types.h"
#include <iostream>
#include <string>
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif

namespace umat::core {
/**
 * @brief
 *
 **/
auto Initialize::initialize_torch_config() -> void {
  if (!has_initialize_torch) {
    // 初始化张量精度
    torch::set_default_dtype(c10::scalarTypeToTypeMeta(torch::kFloat64));

#ifdef _DEBUG
    std::cout << "Torch initialized with default dtype: " << torch::get_default_dtype()
              << std::endl;
#endif
    // 初始化完成
    has_initialize_torch = true;
  }
}
auto Initialize::initialize_abapath_config(char *jobname, int *lenjobname, char *outdir,
                                           int *lenoutdir) -> void {
  if (!has_initialize_abapath) {
    // check intput path
    std::string outdir_str(outdir, *lenoutdir);
    std::string jobname_str(jobname, *lenjobname);
    std::string full_path = outdir_str + "\\" + jobname_str;
    datafile_path = full_path + ".dat";
    msgfile_path = full_path + ".txt";
#ifdef _DEBUG
    std::cout << "data file path : " << datafile_path.c_str() << std::endl;
    std::cout << "msg file path : " << msgfile_path.c_str() << std::endl;
#endif
    has_initialize_abapath = true;
  }
}
} // namespace umat::core
