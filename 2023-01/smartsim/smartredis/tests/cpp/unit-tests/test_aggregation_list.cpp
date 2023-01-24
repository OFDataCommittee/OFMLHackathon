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
#include "client.h"
#include "dataset.h"
#include "../client_test_utils.h"
#include "../dataset_test_utils.h"
#include "srexception.h"
#include <sstream>

using namespace SmartRedis;

// helper function that determines whether two
// vectors of type T contain the same elements
template<typename T>
bool is_same_data(T* t1, T* t2, size_t length)
{
    for(size_t i=0; i<length; i++) {
        if(*t1 != *t2)
            return false;
        t1++;
        t2++;
    }
    return true;
}


// Generate a DataSet filled with random data tensors
// and fixed metadata
template <typename T>
DataSet generate_random_dataset(const std::string& dataset_name,
                                SRTensorType type,
                                const size_t n_tensors)
{
    //Create DataSet
    SmartRedis::DataSet dataset(dataset_name);

    //Add metadata to the DataSet
    DATASET_TEST_UTILS::fill_dataset_with_metadata(dataset);

    // Loop through and create n_tensors
    for (size_t i = 0; i < n_tensors; i++) {

        // Make the tensor name
        std::string t_name = "tensor_" + std::to_string(i);

        // Create random tensor dimensions
        std::vector<size_t> dims(3, 0);
        dims[0] = n_tensors + 1;
        dims[1] = n_tensors + 2;
        dims[2] = n_tensors + 3;

        // Allocate memory and fill with random values
        T*** t_data = allocate_3D_array<T>(dims[0], dims[1], dims[2]);
        set_3D_array_floating_point_values<T>(t_data, dims[0], dims[1], dims[2]);

        // Add the tensor to teh dataset
        dataset.add_tensor(t_name, t_data, dims, type, SRMemLayoutNested);
    }

    return dataset;
}

// Check that the two datasets are the same
template <typename T>
bool is_same_dataset(DataSet& dataset_1, DataSet& dataset_2)
{
    if (dataset_1.get_name() != dataset_2.get_name()) {
        return false;
    }

    DATASET_TEST_UTILS::check_dataset_metadata(dataset_1);
    DATASET_TEST_UTILS::check_dataset_metadata(dataset_2);

    std::vector<std::string> t1_names = dataset_1.get_tensor_names();
    std::vector<std::string> t2_names = dataset_2.get_tensor_names();

    if (t1_names.size() != t2_names.size())
        return false;

    for (size_t i = 0; i < t1_names.size(); i++) {

        std::string t1_name = t1_names[i];
        void* t1_data = NULL;
        std::vector<size_t> t1_dims;
        SRTensorType t1_type;
        SRMemoryLayout t1_layout = SRMemLayoutNested;

        dataset_1.get_tensor(t1_name, t1_data, t1_dims,
                             t1_type, t1_layout);

        std::string t2_name = t2_names[i];
        void* t2_data = NULL;
        std::vector<size_t> t2_dims;
        SRTensorType t2_type;
        SRMemoryLayout t2_layout = SRMemLayoutNested;

        dataset_1.get_tensor(t2_name, t2_data, t2_dims,
                             t2_type, t2_layout);

        if (t1_type != t2_type) {
            return false;
        }

        if (t1_dims != t2_dims) {
            return false;
        }

        if (is_equal_3D_array((double***)t1_data, (double***)t2_data,
                               t1_dims[0], t1_dims[1], t1_dims[2]) == false) {
            return false;
        }
    }

    return true;
}

SCENARIO("Testing Dataset aggregation via our client", "[List]")
{
    GIVEN("A Client object and vector of DataSet objects")
    {
        Client client(use_cluster());

        std::vector<DataSet> datasets;

        for (size_t i = 0; i < 7; i++) {
            std::string dataset_name = "dataset_" + std::to_string(i);
            size_t n_tensors = i + 2;
            DataSet dataset = generate_random_dataset<double>(dataset_name,
                                                              SRTensorTypeDouble,
                                                              n_tensors);
            datasets.push_back(dataset);
        }

        std::string list_name = "unit_test_list";

        WHEN("The DataSet objects are put into the database")
        {
            for (size_t i = 0; i < datasets.size(); i++) {
                client.put_dataset(datasets[i]);
            }

            AND_WHEN("The DataSet objects are added aggregation list")
            {

                // Make sure that the list does not exist
                client.delete_list(list_name);

                for (size_t i = 0; i < datasets.size(); i++) {
                    client.append_to_list(list_name, datasets[i]);
                }

                AND_THEN("The aggregation list length can be retrieved "\
                         "and is the correct value")
                {
                    int list_length = client.get_list_length(list_name);
                    CHECK((size_t)list_length == datasets.size());
                }

                AND_THEN("Polling for the correct list length exits "\
                         "immediately")
                {
                    int list_length = datasets.size();
                    CHECK(client.poll_list_length(list_name,
                                                  list_length, 1, 100));
                }

                AND_THEN("Polling for a list length size too large "\
                         "returns false")
                {
                    int list_length = datasets.size() + 1;
                    CHECK(client.poll_list_length(list_name,
                                                  list_length, 5, 5) == false);
                }

                AND_THEN("Polling for a negative list length throws an error")
                {
                    int list_length = -1;
                    CHECK_THROWS_AS(client.poll_list_length(list_name,
                                                           list_length, 5, 5),
                                                           ParameterException);
                }

                AND_THEN("Polling for a greater than or equal length exits "\
                         "immediately when given the correct length")
                {
                    int list_length = datasets.size();
                    CHECK(client.poll_list_length_gte(list_name,
                                                      list_length, 1, 100));
                }

                AND_THEN("Polling for a greater than or equal length exits "\
                         "with false when given a larger length")
                {
                    int list_length = datasets.size() + 1;
                    CHECK(client.poll_list_length_gte(list_name, list_length,
                                                      5, 5) == false);
                }

                AND_THEN("Polling for a greater than or equal length exits "\
                         "with true when given a smaller length")
                {
                    int list_length = datasets.size() - 1;
                    CHECK(client.poll_list_length_gte(list_name, list_length,
                                                      5, 5) == true);
                }

                AND_THEN("Polling for a negative list length throws an error")
                {
                    int list_length = -1;
                    CHECK_THROWS_AS(client.poll_list_length_gte(list_name,
                                                           list_length, 5, 5),
                                                           ParameterException);
                }

                AND_THEN("Polling for a less than or equal length exits "\
                         "with true when given the correct length")
                {
                    int list_length = datasets.size();
                    CHECK(client.poll_list_length_lte(list_name,
                                                      list_length, 1, 100));
                }

                AND_THEN("Polling for a less than or equal length exits "\
                         "with true when given a larger length")
                {
                    int list_length = datasets.size() + 1;
                    CHECK(client.poll_list_length_lte(list_name, list_length,
                                                      5, 5) == true);
                }

                AND_THEN("Polling for a less than or equal length exits "\
                         "with false when given a smaller length")
                {
                    int list_length = datasets.size() - 1;
                    CHECK(client.poll_list_length_lte(list_name, list_length,
                                                      5, 5) == false);
                }

                AND_THEN("Polling for a negative list length throws an error")
                {
                    int list_length = -1;
                    CHECK_THROWS_AS(client.poll_list_length_lte(list_name,
                                                           list_length, 5, 5),
                                                           ParameterException);
                }

                AND_THEN("The DataSet objects can be retrieved via  "\
                     "the aggregation list and match the original "\
                     "DataSet objects")
                {
                    std::vector<DataSet> retrieved_datasets =
                        client.get_datasets_from_list(list_name);

                    CHECK(retrieved_datasets.size() == datasets.size());

                    // This assumes datasets are in the same order,
                    // and they should be in the current API
                    for (size_t i = 0; i < datasets.size(); i++) {
                        CHECK(is_same_dataset<double>(datasets[i],
                            retrieved_datasets[i]));
                    }
                }

                AND_THEN("A subset of DataSet objects can be retrieved via  "\
                         "the aggregation list and match the original "\
                         "DataSet objects")
                {
                    int start_index = 1;
                    int end_index = 3;
                    std::vector<DataSet> retrieved_datasets =
                        client.get_dataset_list_range(list_name,
                                                      start_index,
                                                      end_index);

                    int n_values = end_index - start_index + 1;
                    CHECK(retrieved_datasets.size() == n_values);

                    // This assumes datasets are in the same order,
                    // and they should be in the current API
                    for (size_t i = start_index; i <= end_index; i++) {
                        CHECK(is_same_dataset<double>(datasets[i],
                            retrieved_datasets[i-start_index]));
                    }
                }

                AND_WHEN("An empty string list is attempted to be retrieved")
                {
                    THEN("A ParameterException is thrown")
                    {
                        std::string empty_string;
                        CHECK_THROWS_AS(client.get_datasets_from_list(empty_string),
                                        ParameterException);
                        CHECK_THROWS_AS(client.get_dataset_list_range(empty_string, 0, -1),
                                        ParameterException);
                    }
                }

                AND_WHEN("The aggregation list is attempted to be "\
                         "copied from an empty string name")
                {
                    THEN("A SmartRedis::ParameterException is thrown")
                    {
                        CHECK_THROWS_AS(client.copy_list("", "error_list"),
                                        ParameterException);
                    }
                }

                AND_WHEN("The aggregation list is attempted to be "\
                         "copied to an empty string name")
                {
                    THEN("A SmartRedis::ParameterException is thrown")
                    {
                        CHECK_THROWS_AS(client.copy_list(list_name, ""),
                                        ParameterException);
                    }
                }

                AND_WHEN("The aggregation list is attempted to be "\
                         "copied to the same name")
                {
                    THEN("The list remains unchanged")
                    {
                        client.copy_list(list_name, list_name);

                        std::vector<DataSet> retrieved_datasets =
                            client.get_datasets_from_list(list_name);

                        CHECK(retrieved_datasets.size() == datasets.size());
                    }
                }

                AND_WHEN("An empty aggregation list is copied")
                {
                    THEN("A RuntimeException is thrown")
                    {
                        CHECK_THROWS_AS(client.copy_list("empty_list","copied_empty_list"),
                                        RuntimeException);
                    }
                }

                AND_WHEN("The aggregation list is copied with valid names")
                {
                    std::string new_list_name =
                        "copied_unit_test_list";
                    client.copy_list(list_name, new_list_name);

                    THEN("The DataSet objects can be retrieved via  "\
                         "the aggregation list and match the original "\
                         "DataSet objects")
                    {
                        std::vector<DataSet> retrieved_datasets =
                            client.get_datasets_from_list(new_list_name);

                        CHECK(retrieved_datasets.size() == datasets.size());

                        // This assumes datasets are in the same order,
                        // and they should be in the current API
                        for (size_t i = 0; i < datasets.size(); i++) {
                            CHECK(is_same_dataset<double>(datasets[i],
                                retrieved_datasets[i]));
                        }
                    }
                }

                AND_WHEN("The aggregation list is copied to an existing "\
                         "list name")
                {
                    std::string existing_list_name("existing_list");

                    // Make sure that the list does not exist
                    client.delete_list(existing_list_name);

                    // Append one dataset to the list
                    client.append_to_list(existing_list_name, datasets[1]);

                    // Copy the list that has more than one dataset
                    // and confirm that the contents are correct
                    client.copy_list(list_name, existing_list_name);

                    THEN("The destination list has the correct size and "\
                         "contents.")
                    {
                        std::vector<DataSet> retrieved_datasets =
                            client.get_datasets_from_list(existing_list_name);

                        CHECK(retrieved_datasets.size() == datasets.size());

                        // This assumes datasets are in the same order,
                        // and they should be in the current API
                        for (size_t i = 0; i < datasets.size(); i++) {
                            CHECK(is_same_dataset<double>(datasets[i],
                                retrieved_datasets[i]));
                        }
                    }
                }

                AND_WHEN("The aggregation list is renamed")
                {
                    std::string new_list_name =
                        "renamed_unit_test_list";
                    client.rename_list(list_name, new_list_name);

                    THEN("The DataSet objects can be retrieved via  "\
                         "the aggregation list and match the original "\
                         "DataSet objects")
                    {
                        std::vector<DataSet> retrieved_datasets =
                            client.get_datasets_from_list(new_list_name);

                        CHECK(retrieved_datasets.size() == datasets.size());

                        // This assumes datasets are in the same order,
                        // and they should be in the current API
                        for (size_t i = 0; i < datasets.size(); i++) {
                            CHECK(is_same_dataset<double>(datasets[i],
                                retrieved_datasets[i]));
                        }
                    }

                    THEN("The original aggregation list was deleted")
                    {
                        std::vector<DataSet> retrieved_datasets =
                            client.get_datasets_from_list(list_name);

                        CHECK(retrieved_datasets.size() == 0);
                    }
                }

                AND_WHEN("The aggregation list is attempted to be renamed "\
                         "from an empty string")
                {
                    THEN("A SmartRedis::ParameterException is thrown")
                    {
                        CHECK_THROWS_AS(client.rename_list("", "error_list"),
                                        ParameterException);
                    }
                }

                AND_WHEN("The aggregation list is attempted to be renamed "\
                         "to an empty string")
                {
                    THEN("A SmartRedis::ParameterException is thrown")
                    {
                        CHECK_THROWS_AS(client.rename_list(list_name, ""),
                                        ParameterException);
                    }
                }

                AND_WHEN("The aggregation list is renamed to the same name")
                {
                    client.rename_list(list_name, list_name);

                    THEN("The DataSet objects can be retrieved via  "\
                         "the aggregation list and match the original "\
                         "DataSet objects")
                    {
                        std::vector<DataSet> retrieved_datasets =
                            client.get_datasets_from_list(list_name);

                        CHECK(retrieved_datasets.size() == datasets.size());

                        // This assumes datasets are in the same order,
                        // and they should be in the current API
                        for (size_t i = 0; i < datasets.size(); i++) {
                            CHECK(is_same_dataset<double>(datasets[i],
                                retrieved_datasets[i]));
                        }
                    }
                }
            }
        }
    }
}
