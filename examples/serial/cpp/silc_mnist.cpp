#include "client.h"

void load_mnist_image_to_array(float**** img)
{
  std::string image_file = "../../../common/mnist_data/one.raw";
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

int main(int argc, char* argv[]) {

  //Allocate a continugous memory
  float* p = (float*)malloc(28*28*sizeof(float));
  float**** array = (float****)malloc(1*sizeof(float***));
  array[0] = (float***)malloc(1*sizeof(float**));
  array[0][0] = (float**)malloc(28*sizeof(float*));
  int pos = 0;
  for(int i=0; i<28; i++) {
    array[0][0][i] = &p[pos];
    pos+=28;
  }

  float **result = (float **)malloc(1*sizeof(float *));
  for (int i=0; i<1; i++)
    result[i] = (float *)malloc(10*sizeof(float));

  load_mnist_image_to_array(array);
  
  SILC::Client client(false);

  std::string model_key = "mnist_model";
  std::string model_file = "../../../common/mnist_data/mnist_cnn.pt";
  client.set_model_from_file(model_key, model_file, "TORCH", "CPU", 20);

  std::string script_key = "mnist_script";
  std::string script_file = "../../../common/mnist_data/data_processing_script.txt";
  client.set_script_from_file(script_key, "CPU", script_file);


  std::string in_key = "mnist_input";
  std::string script_out_key = "mnist_processed_input";
  std::string out_key = "mnist_output";
  client.put_tensor(in_key, array, {1,1,28,28},
                    SILC::TensorType::flt,
                    SILC::MemoryLayout::nested);

  client.run_script("mnist_script", "pre_process", {in_key}, {script_out_key});
  client.run_model("mnist_model", {script_out_key}, {out_key});
  client.unpack_tensor(out_key, result, {1,10},
                       SILC::TensorType::flt,
                       SILC::MemoryLayout::nested);

  // Clean up
  free(p);
  for(int i=0; i<1; i++)
    free(result[i]);
  free(result);

  return 0;
}
