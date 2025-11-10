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

void put_dataset_no_tensors(std::string dataset_name)
{
    //Create Client and DataSet
    DATASET_TEST_UTILS::DatasetTestClient client("client_test_dataset_no_tensors");
    SmartRedis::DataSet sent_dataset(dataset_name);

    //Add metadata to the DataSet
    DATASET_TEST_UTILS::fill_dataset_with_metadata(sent_dataset);

    //Put the DataSet into the database
    client.put_dataset(sent_dataset);

    // Test that the acknowledgement field exists after placement
    std::string ack_key = "{" + dataset_name + "}" + ".meta";
    std::string ack_field = client.ack_field();
    if(!client.hash_field_exists(ack_key, ack_field))
        throw std::runtime_error("The dataset ack field was not set.");

    SmartRedis::DataSet retrieved_dataset = client.get_dataset(dataset_name);

    //Check that the metadata values are correct for the metadata
    DATASET_TEST_UTILS::check_dataset_metadata(retrieved_dataset);

    return;
}

int main(int argc, char* argv[]) {

    put_dataset_no_tensors("dataset_no_tensors");
    return 0;
}
