/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2021 Tomislav Maric, TU Darmstadt
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Description
    Utility functions data exchange between libtorch and OpenFOAM.

SourceFiles
    torchDifferentialOperators.

\*---------------------------------------------------------------------------*/

#include "torchDifferentialOperators.H"

using namespace torch::indexing;

namespace Foam {
namespace AI {
torch::Tensor
div(const torch::Tensor& vel_vec,
    const torch::Tensor& input)
{

  const auto u = vel_vec.index({ Slice(), 0 });
  const auto v = vel_vec.index({ Slice(), 1 });
  const auto w = vel_vec.index({ Slice(), 2 });

  const auto u_grad = torch::autograd::grad(
    { u }, { input }, { torch::ones_like(u) }, true, true);
  const auto v_grad = torch::autograd::grad(
    { v }, { input }, { torch::ones_like(v) }, true, true);
  const auto w_grad = torch::autograd::grad(
    { w }, { input }, { torch::ones_like(w) }, true, true);

  const auto div_vel = u_grad[0].index({ Slice(), 0 }) +
                 v_grad[0].index({ Slice(), 1 }) +
                 w_grad[0].index({ Slice(), 2 });

  return div_vel;
}

torch::Tensor
div(const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& input)
{

  const auto u_grad = torch::autograd::grad(
    { u }, { input }, { torch::ones_like(u) }, true, true);
  const auto v_grad = torch::autograd::grad(
    { v }, { input }, { torch::ones_like(v) }, true, true);
  const auto w_grad = torch::autograd::grad(
    { w }, { input }, { torch::ones_like(w) }, true, true);

  const auto div_vel = u_grad[0].index({ Slice(), 0 }) +
                 v_grad[0].index({ Slice(), 1 }) +
                 w_grad[0].index({ Slice(), 2 });

  return div_vel;
}

torch::Tensor
laplacian(const torch::Tensor& var,
          const torch::Tensor& input)
{

  const auto var_grad = torch::autograd::grad(
    { var },
    { input },
    { torch::ones_like(var) },
    true,
    true);

  // compute second derivatives required for laplacian
  const auto grad_x_var_grad = torch::autograd::grad(
    { var_grad[0].index({ Slice(), 0 }) },
    { input },
    { torch::ones_like(var_grad[0].index({ Slice(), 0 })) },
    true,
    true);
  const auto grad_y_var_grad = torch::autograd::grad(
    { var_grad[0].index({ Slice(), 1 }) },
    { input },
    { torch::ones_like(var_grad[0].index({ Slice(), 1 })) },
    true,
    true);
  const auto grad_z_var_grad = torch::autograd::grad(
    { var_grad[0].index({ Slice(), 2 }) },
    { input },
    { torch::ones_like(var_grad[0].index({ Slice(), 2 })) },
    true,
    true);

  // compute laplacian
  const auto laplacian_var = grad_x_var_grad[0].index({ Slice(), 0 }) +
                       grad_y_var_grad[0].index({ Slice(), 1 }) +
                       grad_z_var_grad[0].index({ Slice(), 2 });

  return laplacian_var;
}

} // namespace AI
} // namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //