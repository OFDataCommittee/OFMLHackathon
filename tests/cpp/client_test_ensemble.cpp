#include "client.h"
#include "client_test_utils.h"
#include <vector>
#include <string>
#include "stdlib.h"

void produce(
		    std::vector<size_t> dims,
        std::string keyout="",
        std::string keyin="")
{
  SILC::Client client(use_cluster());
  client.use_model_ensemble_prefix(true);

  // Tensors
  float* array = (float*)malloc(dims[0]*sizeof(float));

  std::string key = "ensemble_test";

  client.put_tensor(key, (void*)array,
                    dims, SILC::TensorType::flt,
                    SILC::MemoryLayout::nested);

  if(!client.tensor_exists(key))
    std::runtime_error("The tensor key does not exist in the database.");
  if(!client.key_exists(keyout + "." + key))
    std::runtime_error("The tensor key does not exist in the database.");
 
  // Models
  std::string model_key = "mnist_model";
  std::string model_file = "./../mnist_data/mnist_cnn.pt";
  client.set_model_from_file(model_key, model_file, "TORCH", "CPU");

  std::cout << model_key << std::endl;
  std::cout << keyout+'.'+model_key << std::endl;

  if(!client.model_exists(model_key))
    std::runtime_error("The model key does not exist in the database.");
  if(!client.key_exists(keyout + "." + model_key))
    std::runtime_error("The model key does not exist in the database.");

  // Scripts
  std::string script_key = "mnist_script";
  std::string script_file = "./../mnist_data/data_processing_script.txt";
  client.set_script_from_file(script_key, "CPU", script_file);

  if(!client.model_exists(script_key))
    std::runtime_error("The script key does not exist in the database.");
  if(!client.key_exists(keyout + "." + script_key))
    std::runtime_error("The script key does not exist in the database.");


  free_1D_array(array);

  return;
}

void consume(
        std::vector<size_t> dims,
        std::string keyout="",
        std::string keyin="")
{
  SILC::Client client(use_cluster());
  client.use_model_ensemble_prefix(true);

  std::string key = "ensemble_test";
  float* u_result = (float*)malloc(dims[0]*sizeof(float));

  // Check for false positives
  if(client.tensor_exists(key))
    std::runtime_error("The key should not exist in the database.");
  if(client.key_exists(keyout + "." + key))
    std::runtime_error("The key should not exist in the database.");

  client.set_data_source(keyin);

  // Check for false positives
  if(!client.tensor_exists(key))
    std::runtime_error("The key does not exist in the database.");
  if(!client.key_exists(keyin + "." + key))
    std::runtime_error("The key does not exist in the database.");

  client.unpack_tensor(key, u_result, dims,
                       SILC::TensorType::flt,
                       SILC::MemoryLayout::nested);

  SILC::TensorType g_type;
  std::vector<size_t> g_dims;
  void* g_result;
  client.get_tensor(key, g_result, g_dims,
                    g_type, SILC::MemoryLayout::nested);
  float* g_type_result = (float*)g_result;

  if(SILC::TensorType::flt!=g_type)
    throw std::runtime_error("The tensor type "\
                             "retrieved with client.get_tensor() "\
                             "does not match the known type.");

  if(g_dims!=dims)
    throw std::runtime_error("The tensor dimensions retrieved "\
                             "client.get_tensor() do not match "\
                             "the known tensor dimensions.");

  free_1D_array(u_result);

  return;
}

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

void run_mnist(const std::string& model_name,
               const std::string& script_name)
{
  SILC::Client client(use_cluster());

  float**** array = allocate_4D_array<float>(1,1,28,28);
  float** result = allocate_2D_array<float>(1, 10);

  load_mnist_image_to_array(array);

  std::string in_key = "mnist_input";
  std::string script_out_key = "mnist_processed_input";
  std::string out_key = "mnist_output";


  client.put_tensor(in_key, array, {1,1,28,28}, SILC::TensorType::flt,
                    SILC::MemoryLayout::nested);
  client.run_script(script_name, "pre_process", {in_key}, {script_out_key});
  client.run_model(model_name, {script_out_key}, {out_key});
  client.unpack_tensor(out_key, result, {1,10}, SILC::TensorType::flt,
                       SILC::MemoryLayout::nested);

  for(int i=0; i<10; i++)
    std::cout<<"result "<<result[0][i]<<std::endl;

  free_4D_array(array, 1, 1, 28);
  free_2D_array(result, 1);
  return;
}

int main(int argc, char* argv[]) {

  char keyin_env_put[] = "SSKEYIN=producer_0,producer_1";
  char keyout_env_put[] = "SSKEYOUT=producer_0";
  putenv( keyin_env_put ); 
  putenv( keyout_env_put ); 
  size_t dim1 = 10;
  std::vector<size_t> dims = {dim1};

  produce(dims,
				  std::string("producer_0"),
          std::string("producer_0"));

  /*
  std::string model_key = "mnist_model";
  std::string model_file = "./../mnist_data/mnist_cnn.pt";
  client.set_model_from_file(model_key, model_file, "TORCH", "CPU");

  std::string script_key = "mnist_script";
  std::string script_file = "./../mnist_data/data_processing_script.txt";
  client.set_script_from_file(script_key, "CPU", script_file);

  std::string_view model = client.get_model(model_key);

  std::string_view script = client.get_script(script_key);


  run_mnist("mnist_model", "mnist_script");
  */

  char keyin_env_get[] = "SSKEYIN=producer_0,producer_1";
  char keyout_env_get[] = "SSKEYOUT=producer_1";
  putenv( keyin_env_get ); 
  putenv( keyout_env_get ); 
  consume(dims,
          std::string("producer_1"),
          std::string("producer_0"));

  std::cout<<"Ensemble test complete"<<std::endl;

  return 0;
}
