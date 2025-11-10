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

#ifndef torchDifferentialOperators_H
#define torchDifferentialOperators_H

#include <torch/torch.h>

#include "vector.H"

namespace Foam {
namespace AI {

/** \brief Compute the divergence of a vector field.
 *  \param[in] vel_vec The velocity vector field.
 *  \param[in] input The input field.
 *  \return The divergence of the input field.
 */
torch::Tensor
div(const torch::Tensor& vel_vec,
    const torch::Tensor& input);

/** \brief Compute the divergence of a vector field given as three scalar fields (u, v, w).
 *  \param[in] The scalar field: u
 *  \param[in] The scalar field: v
 *  \param[in] The scalar field: w
 *  \param[in] input The input field.
 *  \return The divergence of the input field.
 * \overload
 */
torch::Tensor
div(const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& input);

/** \brief Compute the laplacian of a scalar field.
 *  \param[in] var The scalar field.
 *  \param[in] input The input field.
 *  \return The laplacian of the input field.
 */
torch::Tensor
laplacian(const torch::Tensor& var,
          const torch::Tensor& input,
          const torch::Tensor& grad_output);
} // namespace AI
} // namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
