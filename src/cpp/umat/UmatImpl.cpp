/*****************************************************************************
 *  Project Name
 *  Copyright (C) 2026 Your Name <your.email@example.com>
 *
 *  This file is part of Project Name.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the
 *  "Software"), to deal in the Software without restriction, including
 *  without limitation the rights to use, copy, modify, merge, publish,
 *  distribute, sublicense, and/or sell copies of the Software, and to
 *  permit persons to whom the Software is furnished to do so, subject to
 *  the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included
 *  in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 *  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 *  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *  @file     umat.cpp
 *  @brief    简要说明
 *  @details  详细描述
 *
 *  @author   Wu Wenhao
 *  @email    617082766@qq.com
 *  @version  1.0.0.1
 *  @date     2026/02/14
 *  @license  MIT License
 *---------------------------------------------------------------------------*
 *  Remark         : 说明备注
 * -------------------------------------------------------------------
 * Remark         : A state variable array of size NSTATV to be
 * updated by the UMAT,which includes following variables:
 * statev(0) = void_ratio : current void ratio
 * statev(1) = p0
 * statev(2) = alpha[1] : alpha a11
 * statev(3) = alpha[2] : alpha a22
 * statev(4) = alpha[3] : alpha a33
 * statev(5) = alpha[4] : alpha a12,a21
 * statev(6) = alpha[5] : alpha a13,a31
 * statev(7) = alpha[6] : alpha a23,a32
 * statev(8) = alpha_in[1] : alpha_in a_in11
 * statev(9) = alpha_in[2] : alpha_in a_in22
 * statev(10)= alpha_in[3] : alpha_in a_in33
 * statev(11)= alpha_in[4] : alpha_in a_in12,a_in21
 * statev(12)= alpha_in[5] : alpha_in a_in13,a_in31
 * statev(13)= alpha_in[6] : alpha_in a_in23,a_in32
 * statev(14)= confining pressure
 * statev(15)= shear stress
 * statev(16)= ratio stress
 * statev(17)= the total of volumetric strain
 * statev(18)= the increment of shear strain
 * statev(19)= Dkp
 * statev(20)= dilatancy
 * statev(21)= axial stress
 * statev(22)=
 * statev(23)=
 * statev(24)=
 * statev(25)=
 * statev(26)=
 * statev(27)=
 * statev(28)=
 * -------------------------------------------------------------------
 *---------------------------------------------------------------------------*
 *  Change History :
 *  <Date>     | <Version> | <Author>       | <Description>
 *  2026/02/14 | 1.0.0.1   | Wu wenhao      | Create file
 *****************************************************************************/
#include "umat/UmatImpl.h"
#include "core/ShareVar.h"
#include "core/StateVar.h"
#include "core/StressTensor.h"
#include "core/auxiliary.h"
#include "ops/Elastic.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "umat/math.h"
#include "utils/TypeMap.h"
#include "utils/base_config.h"
#include "utils/config.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>
#include <torch/torch.h>

namespace umat {
using namespace utils;
using namespace core;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-dllimport"
#endif
// 调用fortran 函数
#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif
auto UmatImpl::call(double *stress, double *statev, double *ddsdde, double *strain, double *dstrain,
                    int ntens, int nstatv, double *pnewdt, int noel, int npt, int /*kspt*/,
                    int /*jstep*/, int /*kinc*/, int number) -> ErrorCode {
  // 创建err
#ifdef STRICT_CHECK_ENABLED
  ErrorCode err = ErrorCode::Success;
#else
  ErrorCode *err = nullptr;
#endif
#ifdef DEBUG_SPAN_ENABLED
  // 方便调试时查看变量
  auto stress_view = std::span<data_t>(stress, ntens);
  auto strain_view = std::span<data_t>(strain, ntens);
  auto dstrain_view = std::span<data_t>(dstrain, ntens);
  auto statev_view = std::span<data_t>(statev, nstatv);
  auto ddsdde_view = std::span<data_t>(ddsdde, static_cast<size_t>((ntens) * (ntens)));
#endif
  // try {
  auto state = size_to_stress_state(ntens);
  auto sigma = -make_StressTensor(stress, state);
  auto depsln = -make_StressTensor(dstrain, state, 2.0);
  auto strain_tensor = -make_StressTensor(strain, state, 2.0);
  auto alpha = make_StressTensor(statev, 2, state);
  auto p0 = make_StressTensor(statev[1], state);
  auto alpha_ini = make_StressTensor(statev, 8, state);
  // constructor
  auto shvar = core::make_ShareVar(sigma, alpha, p0);
  auto stvar = make_Statevar(statev[0], alpha_ini, *pnewdt);

  /**
   * @brief
   *
   **/
  if (shvar.is_lowstress()) {
    //@brief the stress is too low
  }
  // @brief elastic trial
  auto mean_etr = ops::pressure_with_depsln(shvar, stvar, depsln, &err);
  auto df_etr = ops::ftol_with_depsln(shvar, stvar, depsln, &err);
  auto dsdetl = [&]() -> Tensor {
    if (mean_etr >= 0.0 && df_etr <= FTOLR) {
      return math::elastic_update(shvar, stvar, depsln, &err);
      //
    } else if (df_etr > FTOLR || (df_etr < FTOLR && mean_etr < 0.0)) {
      auto alout = math::intchc(shvar, stvar, depsln, &err);
      if (fabs(alout) < 1e-12) {
        return math::onyield(shvar, stvar, depsln, noel, npt, &err);
      }
      auto depsln_ela = alout * depsln;
      auto depsln_pla = depsln - depsln_ela;
      auto stiff_ela = math::elastic_update(shvar, stvar, depsln_ela, &err);
      /**
       * @brief : calculate dsigma for plastic correction
       *
       **/
      auto stiff_pla = math::onyield(shvar, stvar, depsln_pla, noel, npt, &err);
      return alout * stiff_ela + (1.0 - alout) * stiff_pla;
    }
    err = ErrorCode::UnKnownError;
    return torch::empty({});
  }();
  //@brief update output variables
  convert_tensor_to_array(stress, 0, ntens, -(shvar.get_stress()), 1.0);
  statev[0] = stvar.get_voidr();
  statev[1] = shvar.get_p0()->item<data_t>();
  *pnewdt = stvar.get_pnewdt();
  convert_tensor_to_array(statev, 2, nstatv, shvar.get_alpha(), 1.0);
  convert_tensor_to_array(statev, 8, nstatv, stvar.get_alphaIni(), 1.0);
  convert_tensor4_to_array(ddsdde, ntens, dsdetl, state);
  return ErrorCode::Success;
  // }
  // #ifdef STRICT_CHECK_ENABLED
  //   catch (const std::exception &e) {
  //     std::ofstream msg_file(core::Initialize::get_msgfile_path().c_str(), std::ios::app);
  //     // 写入异常日志到 .msg
  //     msg_file << "\n==============================================================" <<
  //     std::endl; msg_file << "UMAT 异常捕获 计算失败" << std::endl; msg_file << "" << std::endl;
  //     msg_file << "error message: " << e.what() << std::endl;
  //     msg_file << "==============================================================\n" <<
  //     std::endl;
  //     // 立即写入磁盘，防止日志丢失
  //     msg_file.flush();
  //     //
  //     msg_file.close();
  //     // return a empty tensor
  //     return ErrorCode::UnKnownError;
  //   }
  // #endif
  //   catch (...) {
  //     std::ofstream msg_file(core::Initialize::get_msgfile_path(), std::ios::app);
  //     msg_file << "\n==============================================================" <<
  //     std::endl; msg_file << "UMAT 未知异常 [elastic_update 崩溃]" << std::endl; msg_file <<
  //     "==============================================================\n" << std::endl;

  //     msg_file.flush();
  //     msg_file.close();
  //     return ErrorCode::UnKnownError;
  //   }
}

/**
 * @brief
 * @param  sigma
 * @param  ntens
 * @param  noel
 * @param  npt
 *
 * @return ErrorCode
 **/
auto Sigini_impl::call(double *sigma, int *ntens, int *noel, int *npt, char *jobname,
                       int *lenjobname, char *outdir, int *lenoutdir) -> size_t {
  core::Initialize::initialize_torch_config();
  core::Initialize::initialize_abapath_config(jobname, lenjobname, outdir, lenoutdir);
  if (*ntens < 0) {
    return convert_ErrorCode_to_num(ErrorCode::InputError);
  }
  std::span<data_t, std::dynamic_extent> sigma_view(sigma, static_cast<size_t>(*ntens));
  switch (sigma_view.size()) {
  case 3:
    sigma_view[0] = -100.0;
    sigma_view[1] = -100.0;
    sigma_view[2] = 0.0;
    break;
  case 4:
    sigma_view[0] = -100.0;
    sigma_view[1] = -100.0;
    sigma_view[2] = -100.0;
    sigma_view[3] = -0.0;
    break;
  case 6:
    sigma_view[0] = -100.0;
    sigma_view[1] = -100.0;
    sigma_view[2] = -100.0;
    sigma_view[3] = -0.0;
    sigma_view[4] = -0.0;
    sigma_view[5] = -0.0;
    break;
  default:
    return convert_ErrorCode_to_num(ErrorCode::InputError);
  }
#ifdef DEBUG_SPAN_ENABLED
  if (*noel == 1 && *npt == 1) {
    std::cout << "sigma: " << "[";
    for (int i = 0; i < 6; ++i) {
      std::cout << sigma_view[i] << ", ";
    }
    std::cout << "]" << std::endl;
  }
#endif
  return convert_ErrorCode_to_num(ErrorCode::Success);
}
auto Sdvini_impl::call(double *statev, int *nstatv, int * /*noel*/, int * /*npt*/) -> size_t {
  std::span<data_t> statev_view(statev, *nstatv);
  statev_view[0] = 0.8;
  statev_view[1] = 100;
  std::fill_n(statev_view.begin() + 2, statev_view.size() - 2, 0.0);
  return convert_ErrorCode_to_num(ErrorCode::Success);
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
} // namespace umat