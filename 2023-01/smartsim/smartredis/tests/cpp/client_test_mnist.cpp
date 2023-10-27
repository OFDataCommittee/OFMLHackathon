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
#include "client_test_utils.h"

void load_mnist_image_to_array(float**** img)
{
  std::string image_file = "mnist_data/one.raw";
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

void run_mnist(const std::string& model_name,
               const std::string& script_name)
{
  SmartRedis::Client client("client_test_mnist");

  float**** array = allocate_4D_array<float>(1,1,28,28);
  float** result = allocate_2D_array<float>(1, 10);

  load_mnist_image_to_array(array);

  std::string in_key = "mnist_input";
  std::string script_out_key = "mnist_processed_input";
  std::string out_key = "mnist_output";


  client.put_tensor(in_key, array, {1,1,28,28}, SRTensorTypeFloat, SRMemLayoutNested);
  client.run_script(script_name, "pre_process", {in_key}, {script_out_key});
  client.run_model(model_name, {script_out_key}, {out_key});
  client.unpack_tensor(out_key, result, {1,10}, SRTensorTypeFloat, SRMemLayoutNested);

  for(int i=0; i<10; i++)
    std::cout<<"result "<<result[0][i]<<std::endl;

  free_4D_array(array, 1, 1, 28);
  free_2D_array(result, 1);
}

int main(int argc, char* argv[]) {

  SmartRedis::Client client("client_test_mnist");
  std::string model_key = "mnist_model";
  std::string model_file = "mnist_data/mnist_cnn.pt";
  client.set_model_chunk_size(1024 * 1024);
  client.set_model_from_file(model_key, model_file, "TORCH", "CPU");

  std::string script_key = "mnist_script";
  std::string script_file = "mnist_data/data_processing_script.txt";
  client.set_script_from_file(script_key, "CPU", script_file);

  std::string_view model = client.get_model(model_key);

  std::string_view script = client.get_script(script_key);


  run_mnist("mnist_model", "mnist_script");

  std::cout<<"Finished MNIST test."<<std::endl;

  return 0;
}
