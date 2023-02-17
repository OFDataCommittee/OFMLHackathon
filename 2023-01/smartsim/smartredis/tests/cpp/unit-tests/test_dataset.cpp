/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Hewlett Packard Enterprise
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
#include "dataset.h"
#include "srexception.h"
#include <cxxabi.h>
#include "logger.h"

unsigned long get_time_offset();

using namespace SmartRedis;

const char *currentExceptionTypeName() {
    return abi::__cxa_current_exception_type()->name();
}

SCENARIO("Testing DataSet object", "[DataSet]")
{
    std::cout << std::to_string(get_time_offset()) << ": Testing DataSet object" << std::endl;
    std::string context("test_dataset");
    log_data(context, LLDebug, "***Beginning DataSet testing***");

    GIVEN("A DataSet object")
    {
        std::string dataset_name;
        dataset_name = "dataset_name";
        SmartRedis::DataSet dataset(dataset_name);

        WHEN("A tensor is added to the DataSet")
        {
            std::string tensor_name = "test_tensor";
            std::vector<size_t> dims = {1, 2, 3};
            SRTensorType type = SRTensorTypeFloat;
            size_t tensor_size = dims.at(0) * dims.at(1) * dims.at(2);
            std::vector<float> tensor(tensor_size, 2.0);
            void* data = tensor.data();
            SRMemoryLayout mem_layout = SRMemLayoutContiguous;
            dataset.add_tensor(tensor_name, data, dims, type, mem_layout);

            THEN("The tensor name can be retrieved")
            {
                std::vector<std::string> tensor_names =
                    dataset.get_tensor_names();
                CHECK(tensor_names[0] == tensor_name);
            }

            THEN("The tensor can be retrieved")
            {
                void* retrieved_data;
                std::vector<size_t> retrieved_dims;
                SRTensorType retrieved_type;
                dataset.get_tensor(tensor_name, retrieved_data, retrieved_dims,
                                   retrieved_type, mem_layout);

                // Ensure the retrieved tensor is correct
                CHECK(retrieved_dims == dims);
                CHECK(retrieved_type == type);
                float* data_ptr = (float*)data;
                float* ret_data_ptr = (float*)retrieved_data;
                for (int i=0; i<tensor_size; i++) {
                    CHECK(*data_ptr == *ret_data_ptr);
                    data_ptr++;
                    ret_data_ptr++;
                }
            }

            THEN("The tensor can be retrieved with the c-style interface")
            {
                void* retrieved_data;
                size_t* retrieved_dims;
                size_t retrieved_n_dims;
                SRTensorType retrieved_type;
                dataset.get_tensor(tensor_name, retrieved_data,
                                   retrieved_dims, retrieved_n_dims,
                                   retrieved_type, mem_layout);

                // Ensure the retrieved tensor is correct
                CHECK(retrieved_type == type);
                size_t* curr_retrieved_dim = retrieved_dims;
                for (size_t i=0; i<retrieved_n_dims; i++) {
                    CHECK(*curr_retrieved_dim == dims.at(i));
                    curr_retrieved_dim++;
                }
                float* data_ptr = (float*)data;
                float* ret_data_ptr = (float*)retrieved_data;
                for (int i=0; i<tensor_size; i++)
                    CHECK(*data_ptr++ == *ret_data_ptr++);
            }

            THEN("The tensor can be retrieved by unpacking")
            {
                void* retrieved_data =
                    malloc(dims[0] * dims[1] * dims[2] * sizeof(float));
                dataset.unpack_tensor(tensor_name, retrieved_data,
                                      dims, type, mem_layout);

                // Ensure the unpacked tensor is correct
                float* data_ptr = (float*)data;
                float* ret_data_ptr = (float*)retrieved_data;
                for (int i=0; i<tensor_size; i++) {
                    CHECK(*data_ptr == *ret_data_ptr);
                    data_ptr++;
                    ret_data_ptr++;
                }

                free(retrieved_data);
            }

            THEN("The tensor cannot be retrieved if the "
                 "specified name does not exist")
            {
                void* retrieved_data;
                std::vector<size_t> retrieved_dims;
                SRTensorType retrieved_type;

                CHECK_THROWS_AS(
                    dataset.get_tensor("does_not_exist", retrieved_data,
                                       retrieved_dims, retrieved_type,
                                       mem_layout),
                    SmartRedis::KeyException
                );
            }
        }

        AND_WHEN("A meta scalar is added to the DataSet")
        {
            std::string meta_scalar_name = "flt_meta_scalars";
            float meta_scalar = 10.0;
            SRMetaDataType type = SRMetadataTypeFloat;

            CHECK(dataset.has_field(meta_scalar_name) == false);
            dataset.add_meta_scalar(meta_scalar_name, &meta_scalar, type);
            CHECK(dataset.has_field(meta_scalar_name) == true);

            THEN("The meta scalar can be retrieved")
            {
                float* retrieved_data;
                size_t retrieved_length;
                dataset.get_meta_scalars(meta_scalar_name,
                                        (void*&)retrieved_data,
                                         retrieved_length, type);

                CHECK(retrieved_length == 1);
                CHECK(*retrieved_data == meta_scalar);
            }

            AND_THEN("The meta scalar can be cleared")
            {
                float* retrieved_data;
                size_t retrieved_length;
                dataset.clear_field(meta_scalar_name);

                // The meta scalar no longer exists
                CHECK_THROWS_AS(
                    dataset.get_meta_scalars(meta_scalar_name,
                                            (void*&)retrieved_data,
                                             retrieved_length, type),
                    SmartRedis::RuntimeException
                );
            }
        }

        AND_WHEN("A meta string is added to the DataSet")
        {
            std::string meta_str_name = "meta_string_name";
            std::string meta_str_val = "100";
            dataset.add_meta_string(meta_str_name, meta_str_val);

            THEN("The meta string can be retrieved")
            {
                char** meta_str_data;
                size_t n_strings;
                size_t* lengths;
                dataset.get_meta_strings(meta_str_name, meta_str_data,
                                         n_strings, lengths);
                CHECK(*meta_str_data == meta_str_val);
            }

            THEN("The meta string can be retrieved by name")
            {
                std::vector<std::string> meta_str_data =
                    dataset.get_meta_strings(meta_str_name);
                CHECK(meta_str_data[0] == meta_str_val);
            }
        }
    }
    log_data(context, LLDebug, "***End DataSet testing***");
}

SCENARIO("Testing DataSet inspection", "[DataSet]")
{
    std::cout << std::to_string(get_time_offset()) << ": Testing DataSet inspection" << std::endl;
    std::string context("test_dataset_inspection");
    log_data(context, LLDebug, "***Beginning DataSet Inspection testing***");

    GIVEN("A DataSet object")
    {
        std::string dataset_name;
        dataset_name = "dataset_name";
        SmartRedis::DataSet dataset(dataset_name);

        WHEN("Meta data is added to the DataSet")
        {
            std::string float_scalar_name = "flt_meta_scalar";
            float meta_scalar = 10.0;
            SRMetaDataType type = SRMetadataTypeFloat;
            CHECK(dataset.has_field(float_scalar_name) == false);
            dataset.add_meta_scalar(float_scalar_name, &meta_scalar, type);
            CHECK(dataset.has_field(float_scalar_name) == true);

            std::string i32_scalar_name = "i32_meta_scalar";
            int32_t int_32_scalar = 42;
            type = SRMetadataTypeInt32;
            CHECK(dataset.has_field(i32_scalar_name) == false);
            dataset.add_meta_scalar(i32_scalar_name, &int_32_scalar, type);
            CHECK(dataset.has_field(i32_scalar_name) == true);

            std::string string_scalar_name = "string_meta_scalar";
            std::string string_scalar = "Hello, world";
            CHECK(dataset.has_field(string_scalar_name) == false);
            dataset.add_meta_string(string_scalar_name, string_scalar);
            CHECK(dataset.has_field(string_scalar_name) == true);

            THEN("The metadata can be inspected")
            {
                auto names = dataset.get_metadata_field_names();
                CHECK(names.size() == 3);
                CHECK(std::find(names.begin(), names.end(), float_scalar_name) != names.end());
                CHECK(std::find(names.begin(), names.end(), i32_scalar_name) != names.end());
                CHECK(std::find(names.begin(), names.end(), string_scalar_name) != names.end());

                CHECK(SRMetadataTypeFloat == dataset.get_metadata_field_type(float_scalar_name));
                CHECK(SRMetadataTypeInt32 == dataset.get_metadata_field_type(i32_scalar_name));
                CHECK(SRMetadataTypeString == dataset.get_metadata_field_type(string_scalar_name));
            }
        }

        AND_WHEN("A tensor is added to the DataSet")
        {
            std::string tensor_name = "test_tensor";
            std::vector<size_t> dims = {1, 2, 3};
            SRTensorType type = SRTensorTypeFloat;
            size_t tensor_size = dims.at(0) * dims.at(1) * dims.at(2);
            std::vector<float> tensor(tensor_size, 2.0);
            void* data = tensor.data();
            SRMemoryLayout mem_layout = SRMemLayoutContiguous;
            dataset.add_tensor(tensor_name, data, dims, type, mem_layout);

            THEN("The tensor's type can be inspected")
            {
                CHECK(SRTensorTypeFloat == dataset.get_tensor_type(tensor_name));
            }
        }
    }
    log_data(context, LLDebug, "***End DataSet testing***");
}
