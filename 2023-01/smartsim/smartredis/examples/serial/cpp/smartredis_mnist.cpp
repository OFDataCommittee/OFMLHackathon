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

void run_mnist(const std::string& model_name,
               const std::string& script_name,
               SmartRedis::Client& client)
{
    // Initialize a vector that will hold input image tensor
    size_t n_values = 1*1*28*28;
    std::vector<float> img(n_values, 0);

    // Load the MNIST image from a file
    std::string image_file = "../../common/mnist_data/one.raw";
    std::ifstream fin(image_file, std::ios::binary);
    std::ostringstream ostream;
    ostream << fin.rdbuf();
    fin.close();

    const std::string tmp = ostream.str();
    std::memcpy(img.data(), tmp.data(), img.size()*sizeof(float));

    // Declare keys that we will use in forthcoming client commands
    std::string in_key = "mnist_input";
    std::string script_out_key = "mnist_processed_input";
    std::string out_key = "mnist_output";

    // Put the image tensor on the database
    client.put_tensor(in_key, img.data(), {1,1,28,28},
                      SRTensorTypeFloat, SRMemLayoutContiguous);

    // Run the preprocessing script
    client.run_script(script_name, "pre_process",
                      {in_key}, {script_out_key});

    // Run the model
    client.run_model(model_name, {script_out_key}, {out_key});

    // Get the result of the model
    std::vector<float> result(1*10);
    client.unpack_tensor(out_key, result.data(), {10},
                      SRTensorTypeFloat, SRMemLayoutContiguous);

    // Print out the results of the model
    for(size_t i=0; i<result.size(); i++)
        std::cout<<"Rank 0: Result["<<i<<"] = "<<result[i]<<std::endl;
}

int main(int argc, char* argv[]) {

    // Initialize a client for setting the model and script
    SmartRedis::Client client(__FILE__);

    // Build model key, file name, and then set model
    // from file using client API
    std::string model_key = "mnist_model";
    std::string model_file = "../../common/mnist_data/mnist_cnn.pt";
    client.set_model_from_file(
        model_key, model_file, "TORCH", "CPU", 20);

    // Build script key, file name, and then set script
    // from file using client API
    std::string script_key = "mnist_script";
    std::string script_file = "../../common/mnist_data/data_processing_script.txt";
    client.set_script_from_file(script_key, "CPU", script_file);

    // Get model and script to illustrate client API
    // functionality, but this is not necessary for this example.
    std::string_view model = client.get_model(model_key);
    std::string_view script = client.get_script(script_key);

    run_mnist("mnist_model", "mnist_script", client);

    std::cout<<"Finished SmartRedis MNIST example."<<std::endl;

    return 0;
}
