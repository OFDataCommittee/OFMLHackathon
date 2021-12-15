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

#include "client.h"
#include "dataset.h"
#include "client_test_utils.h"
#include "dataset_test_utils.h"

template <typename T_send, typename T_recv>
void get_multiple_tensors(
            void (*fill_array)(T_send***, int, int, int),
            std::vector<size_t> dims,
            SRTensorType type,
            std::string key_suffix,
            std::string dataset_name)
{
    T_send*** t_send_1 =
        allocate_3D_array<T_send>(dims[0], dims[1], dims[2]);
    fill_array(t_send_1, dims[0], dims[1], dims[2]);

    T_send*** t_send_2 =
        allocate_3D_array<T_send>(dims[0], dims[1], dims[2]);
    fill_array(t_send_2, dims[0], dims[1], dims[2]);

    T_send*** t_send_3 =
        allocate_3D_array<T_send>(dims[0], dims[1], dims[2]);
    fill_array(t_send_3, dims[0], dims[1], dims[2]);

    //Create Client and DataSet
    SmartRedis::Client client(use_cluster());
    SmartRedis::DataSet sent_dataset(dataset_name);

    //Add metadata to the DataSet
    DATASET_TEST_UTILS::fill_dataset_with_metadata(sent_dataset);

    //Add tensors to the DataSet
    std::string t_name_1 = "tensor_1";
    std::string t_name_2 = "tensor_2";
    std::string t_name_3 = "tensor_3";

    sent_dataset.add_tensor(t_name_1, t_send_1, dims, type, SRMemLayoutNested);
    sent_dataset.add_tensor(t_name_2, t_send_2, dims, type, SRMemLayoutNested);
    sent_dataset.add_tensor(t_name_3, t_send_3, dims, type, SRMemLayoutNested);

    //Put the DataSet into the database
    client.put_dataset(sent_dataset);

    if(!client.tensor_exists(dataset_name))
        throw std::runtime_error("The DataSet "\
                                 "confirmation key is not set.");

    SmartRedis::DataSet retrieved_dataset = client.get_dataset(dataset_name);

    DATASET_TEST_UTILS::check_tensor_names(retrieved_dataset,
                                    {t_name_1, t_name_2, t_name_3});

    //Check that the tensors are the same
    DATASET_TEST_UTILS::check_nested_3D_tensor(retrieved_dataset,
                                               t_name_1,
                                               type, t_send_1, dims);
    DATASET_TEST_UTILS::check_nested_3D_tensor(retrieved_dataset,
                                               t_name_2,
                                               type, t_send_2, dims);
    DATASET_TEST_UTILS::check_nested_3D_tensor(retrieved_dataset,
                                               t_name_3,
                                               type, t_send_3, dims);

    //Check that the metadata values are correct for the metadata
    DATASET_TEST_UTILS::check_dataset_metadata(retrieved_dataset);

    //Check that if we retrieve a tensor multiple times from a
    //DataSet, and in the course of retrieveing multiple times
    //one of the tensors is altered that it does not affect future
    //tensors retrievals.
    double*** r_t1_data;
    std::vector<size_t> r_t1_dims;
    SRTensorType r_t1_type = SRTensorTypeInvalid;

    retrieved_dataset.get_tensor(t_name_1, (void*&)r_t1_data,
                                 r_t1_dims, r_t1_type,
                                 SRMemLayoutNested);

    for(size_t i=0; i<r_t1_dims[0]; i++)
        for(size_t j=0; j<r_t1_dims[1]; j++)
            for(size_t k=0; k<r_t1_dims[2]; k++)
                r_t1_data[i][j][k] *= 0.5;

    double*** r_t2_data;
    std::vector<size_t> r_t2_dims;
    SRTensorType r_t2_type = SRTensorTypeInvalid;

    retrieved_dataset.get_tensor(t_name_1, (void*&)r_t2_data,
                                 r_t2_dims, r_t2_type,
                                 SRMemLayoutNested);

    assert(is_equal_3D_array(t_send_1, r_t2_data, dims[0], dims[1], dims[2]));
    assert(!is_equal_3D_array(r_t1_data, r_t2_data, dims[0], dims[1], dims[2]));
}

int main(int argc, char* argv[])
{
    //Declare the dimensions for the 3D arrays
    std::vector<size_t> dims{5,4,17};

    std::string dataset_name = "dataset_multiple_gets";
    get_multiple_tensors<double,double>(
                    &set_3D_array_floating_point_values<double>,
                    dims, SRTensorTypeDouble,
                    "_dbl", dataset_name);

    return 0;
}
