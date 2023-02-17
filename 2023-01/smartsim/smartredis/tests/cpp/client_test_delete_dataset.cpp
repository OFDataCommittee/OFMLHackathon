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

#include "client.h"
#include "dataset.h"
#include "client_test_utils.h"
#include "dataset_test_utils.h"

template <typename T_send, typename T_recv>
void dataset_delete(
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

    //Create Client and DataSets
    DATASET_TEST_UTILS::DatasetTestClient client(use_cluster(), "client_test_delete_dataset");
    SmartRedis::DataSet* dataset = new SmartRedis::DataSet(dataset_name);

    //Add tensors to the DataSet
    std::string t_name_1 = "tensor_1";
    std::string t_name_2 = "tensor_2";
    std::string t_name_3 = "tensor_3";

    dataset->add_tensor(t_name_1, t_send_1, dims, type, SRMemLayoutNested);
    dataset->add_tensor(t_name_2, t_send_2, dims, type, SRMemLayoutNested);
    dataset->add_tensor(t_name_3, t_send_3, dims, type, SRMemLayoutNested);

    //Add metadata to the DataSet
    DATASET_TEST_UTILS::fill_dataset_with_metadata(*dataset);

    //Put the DataSet into the database
    client.put_dataset(*dataset);
    delete dataset;

    //Check that the DataSet was actually properly set in the database
    //and can be retrieved.
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

    client.delete_dataset(dataset_name);

    std::string key;
    key = "{"+dataset_name+"}.meta";
    if(client.key_exists(key))
        throw std::runtime_error("The DataSet metadata "\
                                 "was not deleted.");

    key = "{"+dataset_name+"}."+t_name_1;
    if(client.key_exists(key))
        throw std::runtime_error("The DataSet tensor " + key +
                                 " was not deleted.");

    key = "{"+dataset_name+"}."+t_name_2;
    if(client.key_exists(key))
        throw std::runtime_error("The DataSet tensor " + key +
                                 " was not deleted.");

    key = "{"+dataset_name+"}."+t_name_3;
    if(client.key_exists(key))
        throw std::runtime_error("The DataSet tensor " + key +
                                 " was not deleted.");

    std::string ack_key = "{" + dataset_name + "}" + ".meta";
    std::string ack_field = client.ack_field();
    if(client.hash_field_exists(ack_key, ack_field))
        throw std::runtime_error("The dataset ack field still exists "\
                                 "after deletion");
}

int main(int argc, char* argv[]) {

  //Declare the dimensions for the 3D arrays
  std::vector<size_t> dims{5,4,17};

  std::string dataset_name;

  dataset_name = "3D_dbl_dataset_delete";
  dataset_delete<double,double>(
                  &set_3D_array_floating_point_values<double>,
                  dims, SRTensorTypeDouble, "_dbl", dataset_name);

  return 0;
}
