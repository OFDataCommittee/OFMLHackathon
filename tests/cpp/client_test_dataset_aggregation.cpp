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

#include "client.h"
#include "dataset.h"
#include "client_test_utils.h"
#include "dataset_test_utils.h"

template <typename T>
SmartRedis::DataSet generate_dataset(void (*fill_array)(T***, int, int, int),
                                     std::vector<size_t> dims,
                                     SRTensorType type,
                                     std::string dataset_name)
{
    T*** tensor_1 =
        allocate_3D_array<T>(dims[0], dims[1], dims[2]);
    fill_array(tensor_1, dims[0], dims[1], dims[2]);

    T*** tensor_2 =
        allocate_3D_array<T>(dims[0], dims[1], dims[2]);
    fill_array(tensor_2, dims[0], dims[1], dims[2]);

    T*** tensor_3 =
        allocate_3D_array<T>(dims[0], dims[1], dims[2]);
    fill_array(tensor_3, dims[0], dims[1], dims[2]);

    //Create DataSet
    SmartRedis::DataSet dataset(dataset_name);

    //Add metadata to the DataSet
    DATASET_TEST_UTILS::fill_dataset_with_metadata(dataset);

    //Add tensors to the DataSet
    std::string t_name_1 = "tensor_1";
    std::string t_name_2 = "tensor_2";
    std::string t_name_3 = "tensor_3";

    dataset.add_tensor(t_name_1, tensor_1, dims, type, SRMemLayoutNested);
    dataset.add_tensor(t_name_2, tensor_2, dims, type, SRMemLayoutNested);
    dataset.add_tensor(t_name_3, tensor_3, dims, type, SRMemLayoutNested);

    return dataset;
}

template <typename T>
void check_dataset(SmartRedis::DataSet& dataset_1,
                   SmartRedis::DataSet& dataset_2)
{
    if(dataset_1.get_name() != dataset_2.get_name()) {
        throw std::runtime_error("The dataset name " + dataset_1.get_name() +
                                 " does not match the other dataset name " +
                                 dataset_2.get_name());
    }

    std::vector<std::string> d2_tensor_names = dataset_2.get_tensor_names();

    DATASET_TEST_UTILS::check_tensor_names(dataset_1, d2_tensor_names);

    // Check that the tensors are the same in the datasets
    for (size_t i = 0; i < d2_tensor_names.size(); i++) {
        std::string d2_tensor_name = d2_tensor_names[i];
        void* d2_tensor_data;
        std::vector<size_t> d2_tensor_dims;
        SRTensorType d2_tensor_type;
        SRMemoryLayout d2_tensor_layout = SRMemLayoutNested;

        dataset_2.get_tensor(d2_tensor_name,
                             d2_tensor_data,
                             d2_tensor_dims,
                             d2_tensor_type,
                             d2_tensor_layout);

        DATASET_TEST_UTILS::check_nested_3D_tensor(dataset_1,
                                                   d2_tensor_name,
                                                   d2_tensor_type,
                                                   (T***)d2_tensor_data,
                                                   d2_tensor_dims);
    }

    //Check that the metadata values are correct for the metadata
    DATASET_TEST_UTILS::check_dataset_metadata(dataset_1);
    DATASET_TEST_UTILS::check_dataset_metadata(dataset_2);

    return;
}

int main(int argc, char* argv[]) {

    // Create client for dataset and aggregation list actions
    SmartRedis::Client client(use_cluster());

    // Set a fill function for dataset creation
    void (*fill_function)(double***, int, int, int) =
        &set_3D_array_floating_point_values<double>;

    //Declare the dimensions for the 3D arrays
    std::vector<size_t> dims{5,4,17};

    SmartRedis::DataSet dataset_1 = generate_dataset(fill_function, dims,
                                                     SRTensorTypeDouble,
                                                     "dataset_1");

    SmartRedis::DataSet dataset_2 = generate_dataset(fill_function, dims,
                                                     SRTensorTypeDouble,
                                                     "dataset_2");

    SmartRedis::DataSet dataset_3 = generate_dataset(fill_function, dims,
                                                     SRTensorTypeDouble,
                                                     "dataset_3");

    SmartRedis::DataSet dataset_4 = generate_dataset(fill_function, dims,
                                                     SRTensorTypeDouble,
                                                     "dataset_4");

    std::string list_name = "dataset_test_list";

    // Make sure the list is cleared
    client.delete_list(list_name);

    // Put two datasets into the list
    client.put_dataset(dataset_1);
    client.put_dataset(dataset_2);
    client.append_to_list(list_name, dataset_1);
    client.append_to_list(list_name, dataset_2);

    // Put the final two datasets into the list
    client.put_dataset(dataset_3);
    client.put_dataset(dataset_4);
    client.append_to_list(list_name, dataset_3);
    client.append_to_list(list_name, dataset_4);


    int actual_length = 4;

    // Confirm that poll for list length works correctly
    bool poll_result = client.poll_list_length(list_name, actual_length, 100, 5);
    if (poll_result == false) {
        throw std::runtime_error("Polling for list length of " +
                                 std::to_string(actual_length)  +
                                 " returned false for known length of " +
                                 std::to_string(actual_length) + ".");
    }

    poll_result = client.poll_list_length(list_name, actual_length + 1, 100, 5);
    if (poll_result == true) {
        throw std::runtime_error("Polling for list length of " +
                                 std::to_string(actual_length + 1)  +
                                 " returned true for known length of " +
                                 std::to_string(actual_length) + ".");
    }


    // Confirm that poll for greater than or equal list length works correctly
    poll_result = client.poll_list_length_gte(list_name, actual_length - 1, 100, 5);
    if (poll_result == false) {
        throw std::runtime_error("Polling for list length greater "\
                                 "than or equal to " +
                                 std::to_string(actual_length - 1) +
                                 " returned false for known length of " +
                                 std::to_string(actual_length) + ".");
    }

    poll_result = client.poll_list_length_gte(list_name, actual_length, 100, 5);
    if (poll_result == false) {
        throw std::runtime_error("Polling for list length greater "\
                                 "than or equal to " +
                                 std::to_string(actual_length) +
                                 " returned false for known length of " +
                                 std::to_string(actual_length) + ".");
    }

    poll_result = client.poll_list_length_gte(list_name, actual_length + 1, 100, 5);
    if (poll_result == true) {
        throw std::runtime_error("Polling for list length greater "\
                                 "than or equal to " +
                                 std::to_string(actual_length + 1) +
                                 " returned true for known length of " +
                                 std::to_string(actual_length) + ".");
    }

    // Confirm that poll for less than or equal list length works correctly
    poll_result = client.poll_list_length_lte(list_name, actual_length - 1, 100, 5);
    if (poll_result == true) {
        throw std::runtime_error("Polling for list length less "\
                                 "than or equal to " +
                                 std::to_string(actual_length - 1) +
                                 " returned true for known length of " +
                                 std::to_string(actual_length) + ".");
    }

    poll_result = client.poll_list_length_lte(list_name, actual_length, 100, 5);
    if (poll_result == false) {
        throw std::runtime_error("Polling for list length less "\
                                 "than or equal to " +
                                 std::to_string(actual_length) +
                                 " returned false for known length of " +
                                 std::to_string(actual_length) + ".");
    }

    poll_result = client.poll_list_length_lte(list_name, actual_length + 1, 100, 5);
    if (poll_result == false) {
        throw std::runtime_error("Polling for list length less "\
                                 "than or equal to " +
                                 std::to_string(actual_length + 1) +
                                 " returned false for known length of " +
                                 std::to_string(actual_length) + ".");
    }

    // Check the list length
    int list_length = client.get_list_length(list_name);

    if (list_length != actual_length) {
        throw std::runtime_error("The list length of " +
                                 std::to_string(list_length) +
                                 " does not match expected value of " +
                                 std::to_string(actual_length) + ".");
    }

    // Retrieve datasets via the aggregation list
    std::vector<SmartRedis::DataSet> datasets =
        client.get_datasets_from_list(list_name);

    if (datasets.size() != list_length) {
        throw std::runtime_error("The number of datasets received " +
                                 std::to_string(datasets.size()) +
                                 " does not match expected value of " +
                                 std::to_string(list_length) + ".");
    }

    // Verify the datasets contain the correct information
    check_dataset<double>(dataset_1, datasets[0]);
    check_dataset<double>(dataset_2, datasets[1]);
    check_dataset<double>(dataset_3, datasets[2]);
    check_dataset<double>(dataset_4, datasets[3]);

    return 0;
}
