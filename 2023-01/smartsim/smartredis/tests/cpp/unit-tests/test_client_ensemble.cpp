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
#include "client.h"
#include "dataset.h"
#include "logger.h"
#include "../client_test_utils.h"

unsigned long get_time_offset();

using namespace SmartRedis;

// helper function for resetting environment
// variables to their original state
void reset_env_vars(const char* old_keyin, const char* old_keyout)
{
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
}

// helper function for loading mnist
void load_mnist_image_to_array(float**** img)
{
    std::string image_file = "../mnist_data/one.raw";
    std::ifstream fin(image_file, std::ios::binary);
    std::ostringstream ostream;
    ostream << fin.rdbuf();
    fin.close();

    const std::string tmp = ostream.str();
    const char *image_buf = tmp.data();
    int image_buf_length = tmp.length();

    int position = 0;
    for(int i=0; i<28; i++) {
        for(int j=0; j<28; j++) {
            img[0][0][i][j] = ((float*)image_buf)[position++];
        }
    }
}


SCENARIO("Testing Client ensemble using a producer/consumer paradigm")
{
    std::cout << std::to_string(get_time_offset()) << ": Testing Client ensemble using a producer/consumer paradigm" << std::endl;
    std::string context("test_client_ensemble");
    log_data(context, LLDebug, "***Beginning Client Ensemble testing***");

    GIVEN("Variables that will be used by the producer and consumer")
    {
        const char* old_keyin = std::getenv("SSKEYIN");
        const char* old_keyout = std::getenv("SSKEYOUT");
        char keyin_env_put[] = "producer_0,producer_1";
        char keyout_env_put[] = "producer_0";
        char keyin_env_get[] = "producer_1,producer_0";
        char keyout_env_get[] = "producer_1";
        size_t dim1 = 10;
        std::vector<size_t> dims = {dim1};
        std::string producer_keyout = "producer_0";
        std::string producer_keyin = "producer_0";
        std::string consumer_keyout = "producer_1";
        std::string consumer_keyin = "producer_0";
        // for tensor
        std::string tensor_key = "ensemble_test";
        // for model
        std::string model_key = "mnist_model";
        std::string model_file = "./../mnist_data/mnist_cnn.pt";
        // for script
        std::string script_key = "mnist_script";
        std::string script_file =
            "./../mnist_data/data_processing_script.txt";
        // for setup mnist
        std::string in_key = "mnist_input";
        std::string out_key = "mnist_output";
        std::string script_out_key = "mnist_processed_input";
        std::string model_name = "mnist_model";
        std::string script_name = "mnist_script";
        // for setup mnist with dataset
        std::string in_key_ds = "mnist_input_ds";
        std::string script_out_key_ds = "mnist_processed_input_ds";
        std::string out_key_ds = "mnist_output_ds";
        std::string dataset_name = "mnist_input_dataset_ds";
        std::string dataset_in_key = "{" + dataset_name + "}." + in_key_ds;
        // for consumer tensor
        SRTensorType g_type;
        std::vector<size_t> g_dims;
        void* g_result;

        THEN("The Client ensemble can be tested with "
             "a producer/consumer relationship")
        {
            ////////////////////////////////////////////////////////////
            // do producer stuff
            log_data(context, LLDebug, "***Beginning producer operations***");
            setenv("SSKEYIN", keyin_env_put, (old_keyin != NULL));
            setenv("SSKEYOUT", keyout_env_put, (old_keyout != NULL));

            Client producer_client("test_client_ensemble::producer");
            producer_client.use_model_ensemble_prefix(true);
            producer_client.set_model_chunk_size(1024 * 1024);

            // Tensors
            float* array = (float*)malloc(dims[0]*sizeof(float));
            for(int i=0; i<dims[0]; i++)
                array[i] = (float)(rand()/((float)RAND_MAX/10.0));
            producer_client.put_tensor(tensor_key, (void*)array,
                              dims, SRTensorTypeFloat,
                              SRMemLayoutNested);
            CHECK(producer_client.tensor_exists(tensor_key) == true);
            CHECK(producer_client.key_exists(producer_keyout+"."+tensor_key) ==
                  true);

            // Models
            producer_client.set_model_from_file(model_key, model_file,
                                               "TORCH", "CPU");
            CHECK(producer_client.model_exists(model_key) == true);
            CHECK(producer_client.poll_model(model_key, 300, 100) == true);

            // Scripts
            producer_client.set_script_from_file(script_key, "CPU",
                                                 script_file);
            CHECK(producer_client.model_exists(script_key) == true);

            // Setup mnist
            float**** mnist_array = allocate_4D_array<float>(1,1,28,28);
            load_mnist_image_to_array(mnist_array);
            producer_client.put_tensor(in_key, mnist_array, {1,1,28,28},
                              SRTensorTypeFloat, SRMemLayoutNested);
            producer_client.run_script(script_name, "pre_process",
                             {in_key}, {script_out_key});
            producer_client.run_model(model_name, {script_out_key}, {out_key});

            // Setup mnist with dataset
            DataSet dataset = DataSet(dataset_name);
            dataset.add_tensor(in_key_ds, mnist_array, {1,1,28,28},
                               SRTensorTypeFloat, SRMemLayoutNested);
            producer_client.put_dataset(dataset);
            CHECK(producer_client.dataset_exists(dataset_name) == true);
            producer_client.run_script(script_name, "pre_process",
                             {dataset_in_key}, {script_out_key_ds});
            producer_client.run_model(model_name, {script_out_key_ds},
                                     {out_key_ds});
            free_4D_array(mnist_array, 1, 1, 28);
            log_data(context, LLDebug, "***End producer operations***");

            ////////////////////////////////////////////////////////////
            // do consumer stuff
            log_data(context, LLDebug, "***Beginning consumer operations***");
            setenv("SSKEYIN", keyin_env_get, 1);
            setenv("SSKEYOUT", keyout_env_get, 1);

            Client consumer_client("test_client_ensemble::consumer");
            consumer_client.use_model_ensemble_prefix(true);

            // Tensors
            float* u_result = (float*)malloc(dims[0]*sizeof(float));
            CHECK(consumer_client.tensor_exists(tensor_key) == false);
            CHECK(consumer_client.key_exists(consumer_keyout+"."+tensor_key) ==
                  false);

            consumer_client.set_data_source("producer_0");
            CHECK(consumer_client.tensor_exists(tensor_key) == true);
            CHECK(consumer_client.key_exists(consumer_keyin+"."+tensor_key) ==
                  true);

            consumer_client.unpack_tensor(tensor_key, u_result,
                                          dims, SRTensorTypeFloat,
                                          SRMemLayoutNested);
            for(int i=0; i<dims[0]; i++)
                CHECK(array[i] == u_result[i]);

            consumer_client.get_tensor(tensor_key, g_result, g_dims,
                                       g_type, SRMemLayoutNested);
            float* g_type_result = (float*)g_result;
            for(int i=0; i<dims[0]; i++)
                CHECK(array[i] == g_type_result[i]);
            CHECK(SRTensorTypeFloat == g_type);
            CHECK(g_dims == dims);

            free_1D_array(array);
            free_1D_array(u_result);

            // Models
            consumer_client.set_data_source("producer_1");
            CHECK(consumer_client.model_exists(model_key) == false);
            consumer_client.set_data_source("producer_0");
            std::string_view model = consumer_client.get_model(model_key);

            // Scripts
            consumer_client.set_data_source("producer_1");
            CHECK(consumer_client.model_exists(script_key) == false);
            consumer_client.set_data_source("producer_0");
            std::string_view script = consumer_client.get_script(script_key);

            // Get mnist result
            float** mnist_result = allocate_2D_array<float>(1, 10);
            consumer_client.unpack_tensor(out_key, mnist_result,
                                          {1,10}, SRTensorTypeFloat,
                                          SRMemLayoutNested);
            consumer_client.unpack_tensor(out_key_ds, mnist_result,
                                          {1,10}, SRTensorTypeFloat,
                                          SRMemLayoutNested);

            // Cleanup
            consumer_client.delete_model(model_key);
            if(consumer_client.model_exists(model_key))
                throw std::runtime_error("The model still exists in the database after being deleted.");
            consumer_client.delete_script(script_key);
            if(consumer_client.model_exists(script_key))
                throw std::runtime_error("The script still exists in the database after being deleted.");
            free_2D_array(mnist_result, 1);

            // reset environment variables to their original state
            reset_env_vars(old_keyin, old_keyout);
            log_data(context, LLDebug, "***End consumer operations***");
        }
    }
    log_data(context, LLDebug, "***End Client Ensemble testing***");
}