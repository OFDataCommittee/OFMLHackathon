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
#include <vector>
#include <string>

int main(int argc, char* argv[]) {

    // Initialize tensor dimensions
    size_t dim1 = 3;
    size_t dim2 = 2;
    size_t dim3 = 5;
    size_t n_values = dim1 * dim2 * dim3;
    std::vector<size_t> dims = {3, 2, 5};

    // Initialize two tensors to random values
    std::vector<double> tensor_1(n_values, 0);
    std::vector<int64_t> tensor_2(n_values, 0);

    for(size_t i=0; i<n_values; i++) {
        tensor_1[i] = 2.0*rand()/RAND_MAX - 1.0;
        tensor_2[i] = rand();
    }

    // Initialize three metadata values we will add
    // to the DataSet
    uint32_t meta_scalar_1 = 1;
    uint32_t meta_scalar_2 = 2;
    int64_t meta_scalar_3 = 3;

    // Initialize a SmartRedis client
    bool cluster_mode = true; // Set to false if not using a clustered database
    SmartRedis::Client client(cluster_mode, __FILE__);

    // Create a DataSet
    SmartRedis::DataSet dataset("example_dataset");

    // Add tensors to the DataSet
    dataset.add_tensor("tensor_1", tensor_1.data(), dims,
                       SRTensorTypeDouble, SRMemLayoutContiguous);

    dataset.add_tensor("tensor_2", tensor_2.data(), dims,
                       SRTensorTypeInt64, SRMemLayoutContiguous);

    // Add metadata scalar values to the DataSet
    dataset.add_meta_scalar("meta_field_1", &meta_scalar_1, SRMetadataTypeUint32);
    dataset.add_meta_scalar("meta_field_1", &meta_scalar_2, SRMetadataTypeUint32);
    dataset.add_meta_scalar("meta_field_2", &meta_scalar_3, SRMetadataTypeInt64);


    // Put the DataSet in the database
    client.put_dataset(dataset);

    // Retrieve the DataSet from the database
    SmartRedis::DataSet retrieved_dataset =
        client.get_dataset("example_dataset");

    // Retrieve one of the tensors
    std::vector<int64_t> unpack_dataset_tensor(n_values, 0);
    retrieved_dataset.unpack_tensor("tensor_2",
                                    unpack_dataset_tensor.data(),
                                    {n_values},
                                    SRTensorTypeInt64,
                                    SRMemLayoutContiguous);

    // Print out the retrieved values
    std::cout<<"Comparing sent and received "\
               "values for tensor_2: "<<std::endl;

    for(size_t i=0; i<n_values; i++)
        std::cout<<"Sent: "<<tensor_2[i]<<" "
                 <<"Received: "
                 <<unpack_dataset_tensor[i]<<std::endl;

    //Retrieve a metadata field
    size_t get_n_meta_values;
    void* get_meta_values;
    SRMetaDataType get_type;
    dataset.get_meta_scalars("meta_field_1",
                             get_meta_values,
                             get_n_meta_values,
                             get_type);

    // Print out the metadata field values
    for(size_t i=0; i<get_n_meta_values; i++)
        std::cout<<"meta_field_1 value "<<i<<" = "
                 <<((uint32_t*)get_meta_values)[i]<<std::endl;

    return 0;
}
