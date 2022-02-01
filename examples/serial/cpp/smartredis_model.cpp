#include "client.h"
#include <vector>
#include <fstream>

int main(int argc, char* argv[]) {

    // Initialize a vector that will hold input image tensor
    size_t n_values = 1*1*28*28;
    std::vector<float> img(n_values, 0);

    // Load the mnist image from a file
    std::string image_file = "../../../common/mnist_data/one.raw";
    std::ifstream fin(image_file, std::ios::binary);
    std::ostringstream ostream;
    ostream << fin.rdbuf();
    fin.close();

    const std::string tmp = ostream.str();
    std::memcpy(img.data(), tmp.data(), img.size()*sizeof(float));

    // Initialize a SmartRedis client to connect to the Redis database
    bool cluster_mode = true; // Set to false if not using a clustered database
    SmartRedis::Client client(cluster_mode);

    // Use the client to set a model in the database from a file
    std::string model_key = "mnist_model";
    std::string model_file = "../../../common/mnist_data/mnist_cnn.pt";
    client.set_model_from_file(model_key, model_file, "TORCH", "CPU", 20);

    // Use the client to set a script from the database form a file
    std::string script_key = "mnist_script";
    std::string script_file = "../../../common/mnist_data/data_processing_script.txt";
    client.set_script_from_file(script_key, "CPU", script_file);

    // Declare keys that we will use in forthcoming client commands
    std::string in_key = "mnist_input";
    std::string script_out_key = "mnist_processed_input";
    std::string out_key = "mnist_output";

    // Put the tensor into the database that was loaded from file
    client.put_tensor(in_key, img.data(), {1,1,28,28},
                      SRTensorTypeFloat, SRMemLayoutContiguous);


    // Run the preprocessing script on the input tensor
    client.run_script("mnist_script", "pre_process", {in_key}, {script_out_key});

    // Run the model using the output of the preprocessing script
    client.run_model("mnist_model", {script_out_key}, {out_key});

    // Retrieve the output of the model
    std::vector<float> result(10, 0);
    client.unpack_tensor(out_key, result.data(), {10},
                         SRTensorTypeFloat, SRMemLayoutContiguous);

    // Print out the results of the model evaluation
    for(size_t i=0; i<result.size(); i++) {
        std::cout<<"Result["<<i<<"] = "<<result[i]<<std::endl;
    }

    return 0;
}
