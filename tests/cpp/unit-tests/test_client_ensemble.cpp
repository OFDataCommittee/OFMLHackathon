#include "../../../third-party/catch/catch.hpp"
#include "client.h"
#include "dataset.h"
#include "../client_test_utils.h"

using namespace SmartRedis;

// helper function for resetting environment
// variables to their original state
void reset_env_vars(const char* old_keyin, const char* old_keyout)
{
    if (old_keyin != nullptr) {
        std::string reset_keyin =
            std::string("SSKEYIN=") + std::string(old_keyin);
        char* reset_keyin_c = new char[reset_keyin.size() + 1];
        std::copy(reset_keyin.begin(), reset_keyin.end(), reset_keyin_c);
        reset_keyin_c[reset_keyin.size()] = '\0';
        putenv( reset_keyin_c);
        delete [] reset_keyin_c;
    }
    else {
        unsetenv("SSKEYIN");
    }
    if (old_keyout != nullptr) {
        std::string reset_keyout =
            std::string("SSKEYOUT=") + std::string(old_keyout);
        char* reset_keyout_c = new char[reset_keyout.size() + 1];
        std::copy(reset_keyout.begin(), reset_keyout.end(), reset_keyout_c);
        reset_keyout_c[reset_keyout.size()] = '\0';
        putenv( reset_keyout_c);
        delete [] reset_keyout_c;
    }
    else {
        unsetenv("SSKEYOUT");
    }
}

// helper function for loading mnist
void load_mnist_image_to_array(float**** img)
{
    std::string image_file = "../../mnist_data/one.raw";
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

    GIVEN("Variables that will be used by the producer and consumer")
    {
        const char* old_keyin = std::getenv("SSKEYIN");
        const char* old_keyout = std::getenv("SSKEYOUT");
        char keyin_env_put[] = "SSKEYIN=producer_0,producer_1";
        char keyout_env_put[] = "SSKEYOUT=producer_0";
        char keyin_env_get[] = "SSKEYIN=producer_1,producer_0";
        char keyout_env_get[] = "SSKEYOUT=producer_1";
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
        std::string model_file = "./../../mnist_data/mnist_cnn.pt";
        // for script
        std::string script_key = "mnist_script";
        std::string script_file =
            "./../../mnist_data/data_processing_script.txt";
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
        TensorType g_type;
        std::vector<size_t> g_dims;
        void* g_result;

        THEN("The Client ensemble can be tested with "
             "a producer/consumer relationship")
        {
            // do producer stuff
            putenv(keyin_env_put);
            putenv(keyout_env_put);

            Client producer_client(use_cluster());
            producer_client.use_model_ensemble_prefix(true);

            // Tensors
            float* array = (float*)malloc(dims[0]*sizeof(float));
            producer_client.put_tensor(tensor_key, (void*)array,
                              dims, TensorType::flt,
                              MemoryLayout::nested);
            CHECK(producer_client.tensor_exists(tensor_key) == true);
            CHECK(producer_client.key_exists(producer_keyout+"."+tensor_key) ==
                  true);
            free_1D_array(array);

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
                              TensorType::flt, MemoryLayout::nested);
            producer_client.run_script(script_name, "pre_process",
                             {in_key}, {script_out_key});
            producer_client.run_model(model_name, {script_out_key}, {out_key});

            // Setup mnist with dataset
            DataSet dataset = DataSet(dataset_name);
            dataset.add_tensor(in_key_ds, mnist_array, {1,1,28,28},
                               TensorType::flt, MemoryLayout::nested);
            producer_client.put_dataset(dataset);
            CHECK(producer_client.tensor_exists(dataset_name) == true);
            producer_client.run_script(script_name, "pre_process",
                             {dataset_in_key}, {script_out_key_ds});
            producer_client.run_model(model_name, {script_out_key_ds},
                                     {out_key_ds});
            free_4D_array(mnist_array, 1, 1, 28);


            // do consumer stuff
            putenv(keyin_env_get);
            putenv(keyout_env_get);

            Client consumer_client(use_cluster());
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
                                          dims, TensorType::flt,
                                          MemoryLayout::nested);
            consumer_client.get_tensor(tensor_key, g_result, g_dims,
                                       g_type, MemoryLayout::nested);
            float* g_type_result = (float*)g_result;
            CHECK(TensorType::flt == g_type);
            CHECK(g_dims == dims);
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
                                          {1,10}, TensorType::flt,
                                          MemoryLayout::nested);
            consumer_client.unpack_tensor(out_key_ds, mnist_result,
                                          {1,10}, TensorType::flt,
                                          MemoryLayout::nested);
            free_2D_array(mnist_result, 1);

            // reset environment variables to their original state
            reset_env_vars(old_keyin, old_keyout);
        }
    }
}