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

void rename_dataset(std::string keyout)
{
    std::vector<size_t> dims({10,10,2});

    DATASET_TEST_UTILS::DatasetTestClient client("client_test_ensemble_dataset");
    client.use_tensor_ensemble_prefix(true);
    client.use_dataset_ensemble_prefix(true);

    double*** t_send_1 =
        allocate_3D_array<double>(dims[0], dims[1], dims[2]);
    set_3D_array_floating_point_values<double>(t_send_1, dims[0], dims[1], dims[2]);

    double*** t_send_2 =
        allocate_3D_array<double>(dims[0], dims[1], dims[2]);
    set_3D_array_floating_point_values<double>(t_send_2, dims[0], dims[1], dims[2]);

    std::string name = "ensemble_dataset";
    SmartRedis::DataSet dataset(name);

    DATASET_TEST_UTILS::fill_dataset_with_metadata(dataset);

    //Add tensors to the DataSet
    std::string t_name_1 = "tensor_1";
    std::string t_name_2 = "tensor_2";

    dataset.add_tensor(t_name_1, t_send_1, dims,
                       SRTensorTypeDouble, SRMemLayoutNested);
    dataset.add_tensor(t_name_2, t_send_2, dims,
                       SRTensorTypeDouble, SRMemLayoutNested);

    // Put the DataSet into the database
    client.put_dataset(dataset);

    // Rename the DataSet
    std::string new_name = "ensemble_dataset_renamed";
    client.rename_dataset(name, new_name);

    // Build strings for testing that the DataSet has been renamed.
    // It is assumed the keys are of the form:
    // ensemble_member.{dataset_name}
    std::string new_dataset_key = keyout + "." + "{"  + new_name + "}";
    std::string old_dataset_key = keyout + "." + "{" + name + "}";
    std::string ack_field = client.ack_field();

    // Test that the acknowledgement hash field in the
    // ensemble_member.{dataset_name.meta} key is present
    // for the new DataSet
    if(!client.hash_field_exists(new_dataset_key + ".meta", ack_field))
        throw std::runtime_error("The dataset ack key for the new "\
                                 "DataSet does not exist in the "
                                 "database.");

    // Test that the acknowledgement hash field in the
    // ensemble_member.{dataset_name.meta} key has been deleted
    // for the old DataSet
    if(client.hash_field_exists(old_dataset_key + ".meta", ack_field))
        throw std::runtime_error("The dataset ack key for the old "\
                                 "DataSet was not deleted.");

    // Test that the metadata key ensemble_member.{dataset_name.meta}.meta
    // for the old DataSet has been deleted
    if(client.key_exists(old_dataset_key + ".meta"))
        throw std::runtime_error("The dataset meta key for the old "\
                                 "DataSet was not deleted.");

    // Test that tensor key ensemble_member.{dataset_name.meta}.tensor_name
    // for the old DataSet has been deleted
    if(client.key_exists(old_dataset_key + "." + t_name_1))
        throw std::runtime_error("The dataset tensor key for " +
                                 t_name_1 +
                                 " was not deleted.");

    // Test that tensor key ensemble_member.{dataset_name.meta}.tensor_name
    // for the old DataSet has been deleted
    if(client.key_exists(old_dataset_key + "." + t_name_2))
        throw std::runtime_error("The dataset tensor key for " +
                                 t_name_2 +
                                 " was not deleted.");

    //Retrieve the new dataset
    SmartRedis::DataSet retrieved_dataset =
        client.get_dataset(new_name);


    DATASET_TEST_UTILS::check_tensor_names(retrieved_dataset,
                                          {t_name_1, t_name_2});

    //Check that the tensors are the same
    DATASET_TEST_UTILS::check_nested_3D_tensor(retrieved_dataset,
                                               t_name_1,
                                               SRTensorTypeDouble,
                                               t_send_1, dims);
    DATASET_TEST_UTILS::check_nested_3D_tensor(retrieved_dataset,
                                               t_name_2,
                                               SRTensorTypeDouble,
                                               t_send_2, dims);

    //Check that the metadata values are correct for the metadata
    DATASET_TEST_UTILS::check_dataset_metadata(retrieved_dataset);
}

void add_to_aggregation_list(std::string keyout)
{
    std::vector<size_t> dims({10,10,2});

    DATASET_TEST_UTILS::DatasetTestClient client("client_test_ensemble_dataset");
    client.use_tensor_ensemble_prefix(true);
    client.use_dataset_ensemble_prefix(true);
    client.use_list_ensemble_prefix(true);

    double*** t_send_1 =
        allocate_3D_array<double>(dims[0], dims[1], dims[2]);
    set_3D_array_floating_point_values<double>(t_send_1, dims[0], dims[1], dims[2]);

    double*** t_send_2 =
        allocate_3D_array<double>(dims[0], dims[1], dims[2]);
    set_3D_array_floating_point_values<double>(t_send_2, dims[0], dims[1], dims[2]);

    std::string name = "ensemble_dataset";
    SmartRedis::DataSet dataset(name);

    DATASET_TEST_UTILS::fill_dataset_with_metadata(dataset);

    //Add tensors to the DataSet
    std::string t_name_1 = "tensor_1";
    std::string t_name_2 = "tensor_2";

    dataset.add_tensor(t_name_1, t_send_1, dims,
                       SRTensorTypeDouble, SRMemLayoutNested);
    dataset.add_tensor(t_name_2, t_send_2, dims,
                       SRTensorTypeDouble, SRMemLayoutNested);

    // Put the DataSet into the database
    client.put_dataset(dataset);

    // Aggregation list name
    std::string list_name("ensemble_dataset_list");

    // Delete the list if it already exists
    client.delete_list(list_name);

    // Add to an aggregation list
    client.append_to_list(list_name, dataset);

    std::string dataset_key = keyout + "." + "{" + name + "}";
    std::string ack_field = client.ack_field();

    // Test that the dataset was correctly placed in the database
    if(!client.hash_field_exists(dataset_key + ".meta", ack_field))
        throw std::runtime_error("The dataset ack key was not found.");

    // Test that the metadata key ensemble_member.{dataset_name.meta}.meta
    // is present
    if(!client.key_exists(dataset_key + ".meta"))
        throw std::runtime_error("The dataset meta key for the "\
                                 "DataSet was not found.");

    // Test that tensor key ensemble_member.{dataset_name.meta}.tensor_name
    // for the DataSet is present
    if(!client.key_exists(dataset_key + "." + t_name_1))
        throw std::runtime_error("The dataset tensor key for " +
                                 t_name_1 +
                                 " was not found.");

    // Test that tensor key ensemble_member.{dataset_name.meta}.tensor_name
    // for the old DataSet is present
    if(!client.key_exists(dataset_key + "." + t_name_2))
        throw std::runtime_error("The dataset tensor key for " +
                                 t_name_2 +
                                 " was not found.");

    // Check that the ensemble list is present ensemble_member.list_name
    std::string list_key = keyout + "." + list_name;

    if(!client.key_exists(list_key))
        throw std::runtime_error("The dataset aggregation list was not found.");

    //Retrieve the new dataset
    std::vector<SmartRedis::DataSet> retrieved_datasets =
        client.get_datasets_from_list(list_name);

    if (retrieved_datasets.size() != 1) {
        throw std::runtime_error("The aggregation list should return one value "\
                                 "but it returned " +
                                 std::to_string(retrieved_datasets.size()) + ".");
    }

    SmartRedis::DataSet& retrieved_dataset = retrieved_datasets[0];

    DATASET_TEST_UTILS::check_tensor_names(retrieved_dataset,
                                          {t_name_1, t_name_2});

    //Check that the tensors are the same
    DATASET_TEST_UTILS::check_nested_3D_tensor(retrieved_dataset,
                                               t_name_1,
                                               SRTensorTypeDouble,
                                               t_send_1, dims);
    DATASET_TEST_UTILS::check_nested_3D_tensor(retrieved_dataset,
                                               t_name_2,
                                               SRTensorTypeDouble,
                                               t_send_2, dims);

    //Check that the metadata values are correct for the metadata
    DATASET_TEST_UTILS::check_dataset_metadata(retrieved_dataset);
}

int main(int argc, char* argv[]) {


    const char* old_keyin = std::getenv("SSKEYIN");
    const char* old_keyout = std::getenv("SSKEYOUT");
    const char* keyin_env_put = "producer_0,producer_1";
    const char* keyout_env_put = "producer_0";
    setenv("SSKEYIN", keyin_env_put, (NULL != old_keyin));
    setenv("SSKEYOUT", keyout_env_put, (NULL != old_keyout));

    rename_dataset("producer_0");
    add_to_aggregation_list("producer_0");

    if (old_keyin != nullptr) {
        setenv("SSKEYIN", old_keyin, 1);
    }
    else {
        unsetenv("SSKEYIN");
    }

    if (old_keyout != nullptr) {
        setenv("SSKEYOUT", old_keyout, 1);
    }
    else {
        unsetenv("SSKEYOUT");
    }

    std::cout<<"Ensemble test complete"<<std::endl;
    return 0;
}
