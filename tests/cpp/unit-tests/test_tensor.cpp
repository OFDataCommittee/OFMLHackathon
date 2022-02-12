/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

#include "../../../third-party/catch/single_include/catch2/catch.hpp"
#include "tensor.h"

using namespace SmartRedis;

SCENARIO("Testing Tensor", "[Tensor]")
{

    GIVEN("Two Tensors")
    {
        // Create first tensor
        std::string name = "test_tensor";
        std::vector<size_t> dims = {1, 2, 3};
        SRTensorType type = SRTensorTypeFloat;
        size_t tensor_size = dims.at(0) * dims.at(1) * dims.at(2);
        std::vector<float> tensor(tensor_size, 0);
        for (size_t i=0; i<tensor_size; i++)
            tensor[i] = 2.0*rand()/RAND_MAX -1.0;
        void* data = tensor.data();
        SRMemoryLayout mem_layout = SRMemLayoutContiguous;
        Tensor<float> t(name, data, dims, type, mem_layout);

        // Create second tensor
        std::string name_2 = "test_tensor_2";
        std::vector<size_t> dims_2 = {3, 2, 2};
        SRTensorType type_2 = SRTensorTypeFloat;
        size_t tensor_size_2 = dims_2.at(0) * dims_2.at(1) * dims_2.at(2);
        std::vector<float> tensor_2(tensor_size_2, 0);
        for (size_t i=0; i<tensor_size_2; i++)
            tensor_2[i] = 2.0*rand()/RAND_MAX -1.0;
        void* data_2 = tensor_2.data();
        SRMemoryLayout mem_layout_2 = SRMemLayoutContiguous;
        Tensor<float> t_2(name_2, data_2, dims_2, type_2, mem_layout_2);

        WHEN("A tensor is copied with the assignment operator")
        {
            t_2 = t;

            THEN("The two tensors are the same")
            {
                float* t_data = (float*)t.data();
                float* t_2_data = (float*)t_2.data();
                for (int i=0; i<tensor_size; i++)
                    CHECK(*t_data++ == *t_2_data++);

                CHECK(t.name() == t_2.name());
                CHECK(t.type() == t_2.type());
                CHECK(t.type_str() == t_2.type_str());
                CHECK(t.dims() == t_2.dims());
                CHECK(t.num_values() == t_2.num_values());
                CHECK(t.buf() == t_2.buf());
            }
        }

        AND_WHEN("A third Tensor is constructed with the copy constructor")
        {
            Tensor<float> t_3(t);

            THEN("The Tensor is copied correctly")
            {
                float* point = (float*)t_3.data();
                for (int i=0; i<tensor_size; i++)
                    CHECK(tensor[i] == *point++);

                CHECK(t_3.name() == t.name());
                CHECK(t_3.type() == t.type());
                CHECK(t_3.dims() == t.dims());
                CHECK(t_3.num_values() == t.num_values());
                CHECK(t_3.buf() == t.buf());
            }
        }

        AND_WHEN("A Tensor is constructed with the move assignment operator")
        {
            t_2 = std::move(t);

            THEN("The Tensor is moved correctly")
            {
                float* point = (float*)t_2.data();
                for (int i=0; i<tensor_size; i++)
                    CHECK(tensor[i] == *point++);

                CHECK(t_2.name() == name);
                CHECK(t_2.type() == type);
                CHECK(t_2.dims() == dims);
                CHECK(t.data() == 0);
            }
        }

        AND_WHEN("A third Tensor is constructed with the move constructor")
        {
            Tensor<float> t_3(std::move(t));

            THEN("The Tensor is moved correctly")
            {
                float* point = (float*)t_3.data();
                for (int i=0; i<tensor_size; i++)
                    CHECK(tensor[i] == *point++);

                CHECK(t_3.name() == name);
                CHECK(t_3.type() == type);
                CHECK(t_3.dims() == dims);
                CHECK(t.data() == 0);
            }
        }
    }
}