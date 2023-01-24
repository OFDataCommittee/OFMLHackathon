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

template <typename T_send, typename T_recv>
void copy_assignment(
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
    DATASET_TEST_UTILS::DatasetTestClient client(use_cluster());
    SmartRedis::DataSet* dataset = new SmartRedis::DataSet(dataset_name);

    //Add tensors to the DataSet
    std::string t_name_1 = "tensor_1";
    std::string t_name_2 = "tensor_2";
    std::string t_name_3 = "tensor_3";

    dataset->add_tensor(t_name_1, t_send_1, dims, type, SRMemLayoutNested);
    dataset->add_tensor(t_name_2, t_send_2, dims, type, SRMemLayoutNested);
    dataset->add_tensor(t_name_3, t_send_3, dims, type, SRMemLayoutNested);

    //Add only a portion of the metadata values so that we can test
    //that a user can add metadata after the object has been copied

    dataset->add_meta_scalar("dbl_field_1",
                            &DATASET_TEST_UTILS::dbl_meta_1,
                            SRMetadataTypeDouble);
    dataset->add_meta_scalar("dbl_field_1",
                            &DATASET_TEST_UTILS::dbl_meta_2,
                            SRMetadataTypeDouble);
    dataset->add_meta_scalar("dbl_field_2",
                            &DATASET_TEST_UTILS::dbl_meta_3,
                            SRMetadataTypeDouble);

    dataset->add_meta_scalar("flt_field_1",
                            &DATASET_TEST_UTILS::flt_meta_1,
                            SRMetadataTypeFloat);
    dataset->add_meta_scalar("flt_field_1",
                            &DATASET_TEST_UTILS::flt_meta_2,
                            SRMetadataTypeFloat);
    dataset->add_meta_scalar("flt_field_2",
                            &DATASET_TEST_UTILS::flt_meta_3,
                            SRMetadataTypeFloat);

    dataset->add_meta_scalar("i64_field_1",
                            &DATASET_TEST_UTILS::i64_meta_1,
                            SRMetadataTypeInt64);
    dataset->add_meta_scalar("i64_field_1",
                            &DATASET_TEST_UTILS::i64_meta_2,
                            SRMetadataTypeInt64);
    dataset->add_meta_scalar("i64_field_2",
                            &DATASET_TEST_UTILS::i64_meta_3,
                            SRMetadataTypeInt64);

    dataset->add_meta_scalar("i32_field_1",
                            &DATASET_TEST_UTILS::i32_meta_1,
                            SRMetadataTypeInt32);
    dataset->add_meta_scalar("i32_field_1",
                            &DATASET_TEST_UTILS::i32_meta_2,
                            SRMetadataTypeInt32);
    dataset->add_meta_scalar("i32_field_2",
                            &DATASET_TEST_UTILS::i32_meta_3,
                            SRMetadataTypeInt32);

    dataset->add_meta_scalar("ui64_field_1",
                            &DATASET_TEST_UTILS::ui64_meta_1,
                            SRMetadataTypeUint64);
    dataset->add_meta_scalar("ui64_field_1",
                            &DATASET_TEST_UTILS::ui64_meta_2,
                            SRMetadataTypeUint64);
    dataset->add_meta_scalar("ui64_field_2",
                            &DATASET_TEST_UTILS::ui64_meta_3,
                            SRMetadataTypeUint64);

    dataset->add_meta_scalar("ui32_field_1",
                            &DATASET_TEST_UTILS::ui32_meta_1,
                            SRMetadataTypeUint32);

    SmartRedis::DataSet copied_dataset = *dataset;

    copied_dataset.add_meta_scalar("ui32_field_1",
                            &DATASET_TEST_UTILS::ui32_meta_2,
                            SRMetadataTypeUint32);
    copied_dataset.add_meta_scalar("ui32_field_2",
                            &DATASET_TEST_UTILS::ui32_meta_3,
                            SRMetadataTypeUint32);

    copied_dataset.add_meta_string("str_field_1",
                            DATASET_TEST_UTILS::str_meta_1);
    copied_dataset.add_meta_string("str_field_1",
                            DATASET_TEST_UTILS::str_meta_2);
    copied_dataset.add_meta_string("str_field_2",
                            DATASET_TEST_UTILS::str_meta_3);

    client.put_dataset(*dataset);

    std::string ack_key = "{" + dataset_name + "}" + ".meta";
    std::string ack_field = client.ack_field();
    if(!client.hash_field_exists(ack_key, ack_field))
        throw std::runtime_error("The first DataSet confirmation "\
                                 "key is not set.");

    //Check that the meta fields we added to CopiedDataSet do not show up
    //in MyDataSet if we put MyDataSet in the repo.
    //Check that the metadata values are correct for ui32
    //TODO we should add a method to DataSet to check if a field exists
    //and confirm that the other fields have not been added.  Currently
    //if we get a field that does not exist we will get a SRRuntimeException.
    //Currently the test that there is only one entry in ui32_field_1
    //is sufficient.
    SmartRedis::DataSet partial_dataset = client.get_dataset(dataset_name);
    DATASET_TEST_UTILS::check_meta_field<uint32_t>(
                                    partial_dataset,
                                    "ui32_field_1",
                                    SRMetadataTypeUint32,
                                    {DATASET_TEST_UTILS::ui32_meta_1});

    delete dataset;

    client.put_dataset(copied_dataset);
    SmartRedis::DataSet full_dataset = client.get_dataset(dataset_name);

    if(!client.hash_field_exists(ack_key, ack_field))
        throw std::runtime_error("The copy assigned DataSet "\
                                 "confirmation key is not set.");

    DATASET_TEST_UTILS::check_tensor_names(full_dataset,
                                    {t_name_1, t_name_2, t_name_3});

    //Check that the tensors are the same
    DATASET_TEST_UTILS::check_nested_3D_tensor(full_dataset,
                                               t_name_1,
                                               type, t_send_1, dims);
    DATASET_TEST_UTILS::check_nested_3D_tensor(full_dataset,
                                               t_name_2,
                                               type, t_send_2, dims);
    DATASET_TEST_UTILS::check_nested_3D_tensor(full_dataset,
                                               t_name_3,
                                               type, t_send_3, dims);

    //Check that the metadata values are correct for the metadata
    DATASET_TEST_UTILS::check_dataset_metadata(full_dataset);

    return;
}

int main(int argc, char* argv[]) {

    //Declare the dimensions for the 3D arrays
    std::vector<size_t> dims{5,4,17};

    std::string dataset_name;

    dataset_name = "3D_dbl_dataset_copy_assign";
    copy_assignment<double,double>(
                    &set_3D_array_floating_point_values<double>,
                    dims, SRTensorTypeDouble, "_dbl", dataset_name);

    dataset_name = "3D_flt_dataset_copy_assign";
    copy_assignment<float,float>(
                    &set_3D_array_floating_point_values<float>,
                    dims, SRTensorTypeFloat, "_flt", dataset_name);

    dataset_name = "3D_i64_dataset_copy_assign";
    copy_assignment<int64_t,int64_t>(
                        &set_3D_array_integral_values<int64_t>,
                        dims, SRTensorTypeInt64, "_i64", dataset_name);

    dataset_name = "3D_i32_dataset_copy_assign";
    copy_assignment<int32_t,int32_t>(
                        &set_3D_array_integral_values<int32_t>,
                        dims, SRTensorTypeInt32, "_i32", dataset_name);

    dataset_name = "3D_i16_dataset_copy_assign";
    copy_assignment<int16_t,int16_t>(
                        &set_3D_array_integral_values<int16_t>,
                        dims, SRTensorTypeInt16, "_i16", dataset_name);

    dataset_name = "3D_i8_dataset_copy_assign";
    copy_assignment<int8_t,int8_t>(
                        &set_3D_array_integral_values<int8_t>,
                        dims, SRTensorTypeInt8, "_i8", dataset_name);

    dataset_name = "3D_ui16_dataset_copy_assign";
    copy_assignment<uint16_t,uint16_t>(
                        &set_3D_array_integral_values<uint16_t>,
                        dims, SRTensorTypeUint16, "_ui16", dataset_name);

    dataset_name = "3D_ui8_dataset_copy_assign";
    copy_assignment<uint8_t,uint8_t>(
                        &set_3D_array_integral_values<uint8_t>,
                        dims, SRTensorTypeUint8, "_ui8", dataset_name);

    std::cout<<"Finished DataSet copy assignment test."<<std::endl;
    return 0;
}
