/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Hewlett Packard Enterprise
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "dataset.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;

namespace SmartRedis {

class PyDataset;

class PyDataset
{
    public:

        /*!
        *   \brief PyDataset constructor
        *   \param name The name of the dataset
        */
        PyDataset(const std::string& name);

        /*!
        *   \brief PyDataset constructor from a
        *          SmartRedis::DataSet object
        *   \param dataset A SmartRedis::DataSet pointer to
        *                  a SmartRedis::DataSet allocated on
        *                  the heap.  The SmartRedis::DataSet
        *                  will be deleted upton PyDataset
        *                  deletion.
        */
        PyDataset(DataSet* dataset);

        /*!
        *   \brief PyDataset destructor
        */
        ~PyDataset();

        /*!
        *   \brief Add a tensor to the PyDataset
        *   \param name The name of the tensor
        *   \param data A py::array containing the tensor data
        *   \param type A std::string corresponding to the tensor
        *               data type
        */
        void add_tensor(const std::string& name,
                        py::array data,
                        std::string& type);

        /*!
        *   \brief Retrieve a tensor from the PyDataset
        *   \param name The name of the tensor
        *   \return py::array containing the tensor data
        */
        py::array get_tensor(const std::string& name);

        /*!
        *   \brief Add a metadata scalar to the PyDataset.
        *          If the field already exists, the value
        *          will be appended to the field.
        *   \param name The name of scalar field
        *   \param data A py::array containing the
        *               scalar value.  The array
        *               must only be of length 1.
        *   \param type The type associated with the
        *               scalar.
        */
        void add_meta_scalar(const std::string& name,
                            py::array data,
                            std::string& type);

        /*!
        *   \brief Add a metadata string to the PyDataset.
        *          If the field already exists, the value
        *          will be appended to the field.
        *   \param name The name of string field
        *   \param data The string to add
        */
        void add_meta_string(const std::string& name,
                            const std::string& data);

        /*!
        *   \brief Get a metadata scalar field from the
        *          PyDataset
        *   \param name The name of the field
        *   \returns A py::array with all of the field values
        */
        py::array get_meta_scalars(const std::string& name);

        /*!
        *   \brief Get a metadata scalar field from the
        *          PyDataset
        *   \param name The name of the field
        *   \returns A py::array with all of the field values
        */
        py::list get_meta_strings(const std::string& name);

        /*!
        *   \brief Get the name of the PyDataset
        *   \returns std::string of the PyDataset name
        */
        std::string get_name();

        /*!
        *   \brief Retrieve a pointer to the underlying
        *          SmartRedis::DataSet object
        *   \returns DataSet pointer within PyDataset
        */
        DataSet* get();

    private:

        DataSet* _dataset;
};

} //namespace SmartRedis