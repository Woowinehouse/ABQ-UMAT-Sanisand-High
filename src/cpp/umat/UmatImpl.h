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
#ifndef UMAT_UMATIMPL_H
#define UMAT_UMATIMPL_H

#include "core/ShareVar.h"
#include "core/StateVar.h"
#include "torch/torch.h"
#include "utils/TypeMap.h"
#include "utils/export.h"
#include <omp.h>

namespace umat {

struct MYUMAT_API Sigini_impl {
  [[nodiscard]] static auto call(double *sigma, int *ntens, int *noel, int *npt, char *jobname,
                                 int *lenjobname, char *outdir, int *lenoutdir) -> size_t;
};

struct MYUMAT_API Sdvini_impl {
  [[nodiscard]] static auto call(double *statev, int *nstatv, int *noel, int *npt) -> size_t;
};
struct MYUMAT_API UmatImpl {
  using ShareVar = core::ShareVar;
  using StateVar = core::StateVar;
  using Tensor = torch::Tensor;
  [[nodiscard]] auto static call(double *stress, double *statev, double *ddsdde, double *strain,
                                 double *dstrain, int ntens, int nstatv, double *pnewdt, int noel,
                                 int npt, int kspt, int jstep, int kinc, int number) -> ErrorCode;
};

#ifdef __cplusplus
extern "C" {
#endif
MYUMAT_API inline auto siginiIntf(double *sigma, int *ntens, int *noel, int *npt, char *jobname,
                                  int *lenjobname, char *outdir, int *lenoutdir) -> size_t {
  return Sigini_impl::call(sigma, ntens, noel, npt, jobname, lenjobname, outdir, lenoutdir);
}
/**
 * @brief
 * @param  statev
 * @param  nstatv
 * @param  noel
 * @param  npt
 *
 * @return size_t
 **/
MYUMAT_API inline auto sdviniIntf(double *statev, int *nstatv, int *noel, int *npt) -> size_t {
  return Sdvini_impl::call(statev, nstatv, noel, npt);
}
/**
 * @brief
 * @param  stress
 * @param  statev
 * @param  ddsdde
 * @param  strain
 * @param  dstrain
 * @param  ntens
 * @param  nstatv
 * @param  pnewdt
 * @param  noel
 * @param  npt
 * @param  number
 *
 **/
void Fortran_umat_intf(double *stress, double *statev, double *ddsdde, double *strain,
                       double *dstrain, int ntens, int nstatv, double *pnewdt, int noel, int npt,
                       int number);
/**
 * @brief : This subroutine implements the user material subroutine (UMAT) for
 * the SANISAND constitutive model in ABAQUS. It calculates stress
 * updates and consistent tangent moduli for soil materials under
 * various loading conditions. The algorithm incorporates elastic
 * predictor - plastic corrector methodology, fabric evolution, and
 * state-dependent hardening. The subroutine handles both elastic and
 * plastic loading paths, including return mapping to the yield surface.
 * @param  stress : Cauchy stress tensor (ntens)
 * @param  statev : State variable array (nstatv)
 * @param  ddsdde : Jacobian matrix (ntens x ntens)
 * @param  strain : Total strains at beginning of increment (ntens)
 * @param  dstrain: Strain increments (ntens)
 * @param  ntens  : Size of stress/strain array (ndi + nshr)
 * @param  nstatv : Number of state variables
 * @param  props  : Material properties array (nprops)
 * @param  nprops : Material properties array (nprops)
 * @param  coords : Spatial coordinates of integration point
 * @param  drot   : Rotation increment matrix (3x3)
 * @param  pnewdt : Ratio of suggested new time increment
 * @param  noel   : Element number
 * @param  npt    : Integration point number
 * @param  layer  : Layer number (for composite shells and layered solids)
 * @param  kspt   : Section point number within the current layer
 * @param  jstep  : Step number
 * @param  kinc   : Increment number
 *
 * @return auto
 **/
MYUMAT_API inline auto CXX_umat_Intf(double *stress, double *statev, double *ddsdde, double *strain,
                                     double *dstrain, int ntens, int nstatv, double *pnewdt,
                                     int noel, int npt, int kspt, int jstep, int kinc, int number)
    -> size_t {
  // 调用C++ UMAT 实现
  auto err =
      static_cast<size_t>(UmatImpl::call(stress, statev, ddsdde, strain, dstrain, ntens, nstatv,
                                         pnewdt, noel, npt, kspt, jstep, kinc, number));
  return err;
}
#ifdef __cplusplus
} // extern "C"
#endif
} // namespace umat
#endif // UMAT_UMATIMPL_H
