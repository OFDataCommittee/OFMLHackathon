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
void put_get_dataset(
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
    DATASET_TEST_UTILS::DatasetTestClient client("client_test_dataset");
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

    // Make sure a nonexistant dataset doesn't exist
    std::string nonexistant("nonexistant");
    if (client.dataset_exists(nonexistant))
      throw std::runtime_error("client.dataset_exists() returns true for "\
                               "for a nonexistant dataset.");
    if (client.poll_dataset(nonexistant, 50, 5))
      throw std::runtime_error("client.poll_dataset() returns true for "\
                               "for a nonexistant dataset.");

    // Put the DataSet into the database
    client.put_dataset(sent_dataset);

    // Check that the ack key was placed for the copied dataset
    std::string ack_key = "{" + dataset_name + "}" + ".meta";
    std::string ack_field = client.ack_field();
    if(!client.hash_field_exists(ack_key, ack_field))
        throw RuntimeException("The DataSet confirmation key is not set.");

    // Make sure it exists
    if (!client.dataset_exists(dataset_name))
      throw std::runtime_error("Client.dataset_exists() returns false "\
                               "after dataset placed into database.");
    if (!client.poll_dataset(dataset_name, 50, 5))
      throw std::runtime_error("Client.poll_dataset() returns false "\
                               "after dataset placed into database.");

    // Test that the acknowledgement field exists after placement
    ack_key = "{" + dataset_name + "}" + ".meta";
    ack_field = client.ack_field();
    if(!client.hash_field_exists(ack_key, ack_field))
        throw std::runtime_error("The dataset ack field was not set.");

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

    return;
}

int main(int argc, char* argv[]) {

    //Declare the dimensions for the 3D arrays
    std::vector<size_t> dims{5,4,17};

    std::string dataset_name;

    dataset_name = "3D_dbl_dataset_put_get";
    put_get_dataset<double,double>(
                    &set_3D_array_floating_point_values<double>,
                    dims, SRTensorTypeDouble,
                    "_dbl", dataset_name);

    dataset_name = "3D_flt_dataset_put_get";
    put_get_dataset<float,float>(
                    &set_3D_array_floating_point_values<float>,
                    dims, SRTensorTypeFloat,
                    "_flt", dataset_name);

    dataset_name = "3D_i64_dataset_put_get";
    put_get_dataset<int64_t,int64_t>(
                        &set_3D_array_integral_values<int64_t>,
                        dims, SRTensorTypeInt64,
                        "_i64", dataset_name);

    dataset_name = "3D_i32_dataset_put_get";
    put_get_dataset<int32_t,int32_t>(
                        &set_3D_array_integral_values<int32_t>,
                        dims, SRTensorTypeInt32,
                        "_i32", dataset_name);

    dataset_name = "3D_i16_dataset_put_get";
    put_get_dataset<int16_t,int16_t>(
                        &set_3D_array_integral_values<int16_t>,
                        dims, SRTensorTypeInt16,
                        "_i16", dataset_name);

    dataset_name = "3D_i8_dataset_put_get";
    put_get_dataset<int8_t,int8_t>(
                        &set_3D_array_integral_values<int8_t>,
                        dims, SRTensorTypeInt8,
                        "_i8", dataset_name);

    dataset_name = "3D_ui16_dataset_put_get";
    put_get_dataset<uint16_t,uint16_t>(
                        &set_3D_array_integral_values<uint16_t>,
                        dims, SRTensorTypeUint16,
                        "_ui16", dataset_name);

    dataset_name = "3D_ui8_dataset_put_get";
    put_get_dataset<uint8_t,uint8_t>(
                        &set_3D_array_integral_values<uint8_t>,
                        dims, SRTensorTypeUint8,
                        "_ui8", dataset_name);

    return 0;
}
