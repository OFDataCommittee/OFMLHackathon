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
#include "client_test_utils.h"
#include <vector>
#include <string>
#include <cstdlib>
#include "stdlib.h"

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

void produce(
            std::vector<size_t> dims,
        std::string keyout="",
        std::string keyin="")
{
  SmartRedis::Client client(use_cluster());
  client.use_model_ensemble_prefix(true);

  // Tensors
  float* array = (float*)malloc(dims[0]*sizeof(float));

  std::string key = "ensemble_test";

  client.put_tensor(key, (void*)array, dims, SRTensorTypeFloat, SRMemLayoutNested);

  if(!client.tensor_exists(key))
    throw std::runtime_error("The tensor key does not exist in the database.");
  if(!client.key_exists(keyout + "." + key))
    throw std::runtime_error("The tensor key does not exist in the database.");

  free_1D_array(array);

  // Models
  std::string model_key = "mnist_model";
  std::string model_file = "./../mnist_data/mnist_cnn.pt";
  client.set_model_from_file(model_key, model_file, "TORCH", "CPU");


  if(!client.model_exists(model_key))
    throw std::runtime_error("The model key does not exist in the database.");

  // Scripts
  std::string script_key = "mnist_script";
  std::string script_file = "./../mnist_data/data_processing_script.txt";
  client.set_script_from_file(script_key, "CPU", script_file);

  if(!client.model_exists(script_key))
    throw std::runtime_error("The script key does not exist in the database.");

  // Setup mnist
  float**** mnist_array = allocate_4D_array<float>(1,1,28,28);
  load_mnist_image_to_array(mnist_array);
  std::string in_key = "mnist_input";
  std::string out_key = "mnist_output";
  std::string script_out_key = "mnist_processed_input";
  std::string model_name = "mnist_model";
  std::string script_name = "mnist_script";
  client.put_tensor(in_key, mnist_array, {1,1,28,28}, SRTensorTypeFloat, SRMemLayoutNested);
  client.run_script(script_name, "pre_process", {in_key}, {script_out_key});
  client.run_model(model_name, {script_out_key}, {out_key});

  // Setup mnist with dataset
  std::string in_key_ds = "mnist_input_ds";
  std::string script_out_key_ds = "mnist_processed_input_ds";
  std::string out_key_ds = "mnist_output_ds";

  std::string dataset_name = "mnist_input_dataset_ds";
  SmartRedis::DataSet dataset = SmartRedis::DataSet(dataset_name);
  dataset.add_tensor(in_key_ds, mnist_array, {1,1,28,28}, SRTensorTypeFloat, SRMemLayoutNested);
  client.put_dataset(dataset);

  if(!client.dataset_exists(dataset_name))
    throw std::runtime_error("The dataset key does not exist in the database.");

  std::string dataset_in_key = "{" + dataset_name + "}." + in_key_ds;
  client.run_script(script_name, "pre_process", {dataset_in_key}, {script_out_key_ds});
  client.run_model(model_name, {script_out_key_ds}, {out_key_ds});

  free_4D_array(mnist_array, 1, 1, 28);
}

void consume(std::vector<size_t> dims,
             std::string keyout="",
             std::string keyin="")
{
  SmartRedis::Client client(use_cluster());
  client.use_model_ensemble_prefix(true);

  // Tensors
  std::string tensor_key = "ensemble_test";
  float* u_result = (float*)malloc(dims[0]*sizeof(float));

  // Check for false positives
  if(client.tensor_exists(tensor_key))
    throw std::runtime_error("The key should not exist in the database.");
  if(client.key_exists(keyout + "." + tensor_key))
    throw std::runtime_error("The key should not exist in the database.");

  client.set_data_source("producer_0");

  // Check for false positives
  if(!client.tensor_exists(tensor_key))
    throw std::runtime_error("The key does not exist in the database.");
  if(!client.key_exists(keyin + "." + tensor_key))
    throw std::runtime_error("The key does not exist in the database.");

  client.unpack_tensor(tensor_key, u_result, dims,
                       SRTensorTypeFloat, SRMemLayoutNested);

  SRTensorType g_type;
  std::vector<size_t> g_dims;
  void* g_result;
  client.get_tensor(tensor_key, g_result, g_dims,
                    g_type, SRMemLayoutNested);
  float* g_type_result = (float*)g_result;

  if(SRTensorTypeFloat!=g_type)
    throw std::runtime_error("The tensor type "\
                             "retrieved with client.get_tensor() "\
                             "does not match the known type.");

  if(g_dims!=dims)
    throw std::runtime_error("The tensor dimensions retrieved "\
                             "client.get_tensor() do not match "\
                             "the known tensor dimensions.");

  free_1D_array(u_result);

  // Models
  std::string model_key = "mnist_model";
  client.set_data_source("producer_1");

  // Check for false positives
  if(client.model_exists(model_key))
    throw std::runtime_error("The model key should not exist in the database.");

  client.set_data_source("producer_0");
  std::string_view model = client.get_model(model_key);

  // Scripts
  std::string script_key = "mnist_script";
  client.set_data_source("producer_1");

  // Check for false positives
  if(client.model_exists(script_key))
    throw std::runtime_error("The script key should not exist in the database.");

  client.set_data_source("producer_0");
  std::string_view script = client.get_script(script_key);

  // Get mnist result
  float** mnist_result = allocate_2D_array<float>(1, 10);

  std::string out_key = "mnist_output";
  client.unpack_tensor(out_key, mnist_result, {1,10}, SRTensorTypeFloat, SRMemLayoutNested);

  std::string out_key_ds = "mnist_output_ds";
  client.unpack_tensor(out_key_ds, mnist_result, {1,10}, SRTensorTypeFloat, SRMemLayoutNested);
  free_2D_array(mnist_result, 1);
}

int main(int argc, char* argv[]) {

  const char* old_keyin = std::getenv("SSKEYIN");
  const char* old_keyout = std::getenv("SSKEYOUT");
  char keyin_env_put[] = "SSKEYIN=producer_0,producer_1";
  char keyout_env_put[] = "SSKEYOUT=producer_0";
  putenv( keyin_env_put );
  putenv( keyout_env_put );
  size_t dim1 = 10;
  std::vector<size_t> dims = {dim1};

  produce(dims,
          std::string("producer_0"),
          std::string("producer_0"));

  char keyin_env_get[] = "SSKEYIN=producer_1,producer_0";
  char keyout_env_get[] = "SSKEYOUT=producer_1";
  putenv(keyin_env_get);
  putenv(keyout_env_get);
  consume(dims,
          std::string("producer_1"),
          std::string("producer_0"));

  if (old_keyin != nullptr) {
    std::string reset_keyin = std::string("SSKEYIN=") + std::string(old_keyin);
    char* reset_keyin_c = new char[reset_keyin.size() + 1];
    std::copy(reset_keyin.begin(), reset_keyin.end(), reset_keyin_c);
    reset_keyin_c[reset_keyin.size()] = '\0';
    putenv( reset_keyin_c);
    delete[] reset_keyin_c;
  }
  else {
    unsetenv("SSKEYIN");
  }
  if (old_keyout != nullptr) {
    std::string reset_keyout = std::string("SSKEYOUT=") + std::string(old_keyout);
    char* reset_keyout_c = new char[reset_keyout.size() + 1];
    std::copy(reset_keyout.begin(), reset_keyout.end(), reset_keyout_c);
    reset_keyout_c[reset_keyout.size()] = '\0';
    putenv( reset_keyout_c);
    delete[] reset_keyout_c;
  }
  else {
    unsetenv("SSKEYOUT");
  }


  std::cout<<"Ensemble test complete"<<std::endl;

  return 0;
}
