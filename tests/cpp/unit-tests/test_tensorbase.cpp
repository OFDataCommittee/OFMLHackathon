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

#include "../../../third-party/catch/catch.hpp"
#include "tensorbase.h"
#include "tensorpack.h"
#include "srexception.h"

using namespace SmartRedis;

SCENARIO("Testing TensorBase through TensorPack", "[TensorBase]")
{
    TensorType tensor_type = GENERATE(TensorType::dbl, TensorType::flt,
                                      TensorType::int64, TensorType::int32,
                                      TensorType::int16, TensorType::int8,
                                      TensorType::uint16, TensorType::uint8);

    GIVEN("A TensorPack object")
    {
        TensorPack tp;

        WHEN("A valid tensor is added to the tensorpack")
        {
            std::string name = "test_tensor";
            std::vector<size_t> dims = {1, 2, 3};
            size_t tensor_size = dims.at(0) * dims.at(1) * dims.at(2);
            std::vector<double> tensor(tensor_size, 0);
            for (size_t i=0; i<tensor_size; i++)
                tensor[i] = 2.0*rand()/RAND_MAX - 1.0;
            void* data = tensor.data();
            tp.add_tensor(name, data, dims,
                          tensor_type,
                          MemoryLayout::contiguous);

            THEN("The tensor and its data members can be retrieved")
            {
                TensorBase* tensor_ptr = tp.get_tensor(name);

                CHECK(tensor_ptr->name() == name);
                CHECK(tensor_ptr->type() == tensor_type);
                CHECK(tensor_ptr->type_str() ==
                      TENSOR_STR_MAP.at(tensor_type));
                CHECK(tensor_ptr->dims() == dims);
                CHECK(tensor_ptr->num_values() == tensor_size);
            }

            AND_THEN("The tensor can be copied")
            {
                TensorPack tp_cpy(tp);

                TensorBase* tensor_ptr = tp.get_tensor(name);
                TensorBase* tensor_cpy_ptr = tp_cpy.get_tensor(name);

                CHECK(tensor_ptr->name() == tensor_cpy_ptr->name());
                CHECK(tensor_ptr->type() == tensor_cpy_ptr->type());
                CHECK(tensor_ptr->type_str() == tensor_cpy_ptr->type_str());
                CHECK(tensor_ptr->dims() == tensor_cpy_ptr->dims());
                CHECK(tensor_ptr->num_values() ==
                      tensor_cpy_ptr->num_values());
                CHECK(tensor_ptr->buf() == tensor_cpy_ptr->buf());
            }

            AND_THEN("The TensorPack that contains the Tensor can be copied")
            {
                TensorPack tp_cpy;
                tp_cpy = tp;

                // ensure the TensorPacks have the same "_all_tensors"
                // data members by iterating over each
                TensorPack::const_tensorbase_iterator it =
                    tp.tensor_cbegin();
                TensorPack::const_tensorbase_iterator it_end =
                    tp.tensor_cend();
                TensorPack::const_tensorbase_iterator it_cpy =
                    tp_cpy.tensor_cbegin();
                TensorPack::const_tensorbase_iterator it_cpy_end =
                    tp_cpy.tensor_cend();

                while (it != it_end) {
                    REQUIRE(it_cpy != it_cpy_end);
                    CHECK((*it)->name() == (*it_cpy)->name());
                    CHECK((*it)->type() == (*it_cpy)->type());
                    CHECK((*it)->type_str() == (*it_cpy)->type_str());
                    CHECK((*it)->dims() == (*it_cpy)->dims());
                    CHECK((*it)->num_values() == (*it_cpy)->num_values());
                    CHECK((*it)->buf() == (*it_cpy)->buf());
                    it++;
                    it_cpy++;
                }
                // verify equivalency of length
                CHECK(it_cpy == it_cpy_end);

                // ensure the tensor inside the TensorPack was copied correctly
                TensorBase* t = tp.get_tensor(name);
                TensorBase* t_cpy = tp_cpy.get_tensor(name);
                CHECK(t->name() == t_cpy->name());
                CHECK(t->type() == t_cpy->type());
                CHECK(t->type_str() == t_cpy->type_str());
                CHECK(t->dims() == t_cpy->dims());
                CHECK(t->num_values() == t_cpy->num_values());
                CHECK(t->buf() == t_cpy->buf());
            }
        }

        AND_WHEN("A tensor with data as nullptr is added to the tensorpack")
        {
            std::string name = "test_tensor";
            std::vector<size_t> dims = {1, 1, 1};
            std::vector<double> tensor;
            void* data = tensor.data();

            THEN("A runtime error is thrown during TensorType initialization")
            {
                CHECK_THROWS_AS(
                    tp.add_tensor(name, data, dims,
                                  tensor_type,
                                  MemoryLayout::contiguous),
                    _smart_runtime_error
                );
            }
        }

        AND_WHEN("A tensor with an empty string as its"
                 "name is added to the tensorpack")
        {
            std::string name = "";
            std::vector<size_t> dims = {1, 2, 3};
            size_t tensor_size = dims.at(0) * dims.at(1) * dims.at(2);
            std::vector<double> tensor(tensor_size, 0);
            for (size_t i=0; i<tensor_size; i++)
                tensor[i] = 2.0*rand()/RAND_MAX - 1.0;
            void* data = tensor.data();

            THEN("A runtime error is thrown")
            {
                CHECK_THROWS_AS(
                    tp.add_tensor(name, data, dims,
                                  tensor_type,
                                  MemoryLayout::contiguous),
                    _smart_runtime_error
                );
            }
        }

        AND_WHEN("A tensor with '.meta' as its name"
                 "is added to the tensorpack")
        {
            std::string name = ".meta";
            std::vector<size_t> dims = {1, 2, 3};
            size_t tensor_size = dims.at(0) * dims.at(1) * dims.at(2);
            std::vector<double> tensor(tensor_size, 0);
            for (size_t i=0; i<tensor_size; i++)
                tensor[i] = 2.0*rand()/RAND_MAX - 1.0;
            void* data = tensor.data();

            THEN("A runtime error is thrown")
            {
                CHECK_THROWS_AS(
                    tp.add_tensor(name, data, dims,
                                 tensor_type,
                                 MemoryLayout::contiguous),
                    _smart_runtime_error
                );
            }
        }

        AND_WHEN("A tensor with zero dimensions is added to the tensorpack")
        {
            std::string name = "test_tensor";
            std::vector<size_t> dims = {};
            std::vector<double> tensor(5, 0);
            for (size_t i=0; i<5; i++)
                tensor[i] = 2.0*rand()/RAND_MAX - 1.0;
            void* data = tensor.data();

            THEN("A runtime error is thrown")
            {
                CHECK_THROWS_AS(
                    tp.add_tensor(name, data, dims,
                                  tensor_type,
                                  MemoryLayout::contiguous),
                    _smart_runtime_error
                );
            }
        }

        AND_WHEN("A tensor with a dimension that is"
                 "<= 0 is added to the tensorpack")
        {
            std::string name = "test_tensor";
            std::vector<size_t> dims = {1, 0, 3};
            std::vector<double> tensor(5, 0);
            for (size_t i=0; i<5; i++)
                tensor[i] = 2.0*rand()/RAND_MAX - 1.0;
            void* data = tensor.data();

            THEN("A runtime error is thrown")
            {
                CHECK_THROWS_AS(
                    tp.add_tensor(name, data, dims,
                                  tensor_type,
                                  MemoryLayout::contiguous),
                    _smart_runtime_error);
            }
        }
    }
}