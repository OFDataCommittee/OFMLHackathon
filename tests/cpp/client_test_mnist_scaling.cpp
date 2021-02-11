#include "client.h"
#include "client_test_utils.h"
#include <chrono>

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
               const std::string& script_name,
               std::ofstream& timing_file)
{
  int rank = 0;

  if(!rank)
    std::cout<<"Connecting clients"<<std::endl<<std::flush;

  auto constructor_start = std::chrono::high_resolution_clock::now();
  SILC::Client client(true);
  auto constructor_end = std::chrono::high_resolution_clock::now();
  auto delta_t = constructor_end-constructor_start;

  timing_file << rank << "," << "client()" << ","
              << std::chrono::duration<double, std::milli>(delta_t).count()
              << std::endl << std::flush;

  //Allocate a continugous memory to make bcast easier
  float* p = (float*)malloc(28*28*sizeof(float));

  float**** array = (float****)malloc(1*sizeof(float***));
  array[0] = (float***)malloc(1*sizeof(float**));
  array[0][0] = (float**)malloc(28*sizeof(float*));
  int pos = 0;
  for(int i=0; i<28; i++) {
    array[0][0][i] = &p[pos];
    pos+=28;
  }

  float** result = allocate_2D_array<float>(1, 10);

  if(rank == 0)
    load_mnist_image_to_array(array);

  if(!rank)
    std::cout<<"All ranks have MNIST image"<<std::endl;

  std::string in_key = "mnist_input_rank_" + std::to_string(rank);
  std::string script_out_key = "mnist_processed_input_rank_" + std::to_string(rank);
  std::string out_key = "mnist_output_rank_" + std::to_string(rank);

  auto put_tensor_start = std::chrono::high_resolution_clock::now();
  client.put_tensor(in_key, array, {1,1,28,28},
                    SILC::TensorType::flt,
                    SILC::MemoryLayout::nested);
  auto put_tensor_end = std::chrono::high_resolution_clock::now();
  delta_t = put_tensor_end - put_tensor_start;
  timing_file << rank << "," << "put_tensor" << ","
              << std::chrono::duration<double, std::milli>(delta_t).count() << std::endl << std::flush;

  auto run_script_start = std::chrono::high_resolution_clock::now();
  client.run_script(script_name, "pre_process", {in_key}, {script_out_key});
  auto run_script_end = std::chrono::high_resolution_clock::now();
  delta_t = run_script_end - run_script_start;
  timing_file << rank << "," << "run_script" << ","
              << std::chrono::duration<double, std::milli>(delta_t).count() << std::endl << std::flush;

  auto run_model_start = std::chrono::high_resolution_clock::now();
  client.run_model(model_name, {script_out_key}, {out_key});
  auto run_model_end = std::chrono::high_resolution_clock::now();
  delta_t = run_model_end - run_model_start;
  timing_file << rank << "," << "run_model" << ","
              << std::chrono::duration<double, std::milli>(delta_t).count() << std::endl << std::flush;

  auto unpack_tensor_start = std::chrono::high_resolution_clock::now();
  client.unpack_tensor(out_key, result, {1,10},
                       SILC::TensorType::flt,
                       SILC::MemoryLayout::nested);
  auto unpack_tensor_end = std::chrono::high_resolution_clock::now();
  delta_t = unpack_tensor_end - unpack_tensor_start;
  timing_file << rank << "," << "unpack_tensor" << ","
              << std::chrono::duration<double, std::milli>(delta_t).count() << std::endl << std::flush;

  free(p);
  free_2D_array(result, 1);
  return;
}

int main(int argc, char* argv[]) {
  auto main_start = std::chrono::high_resolution_clock::now();

  int rank;

  //Open Timing file
  std::ofstream timing_file;
  timing_file.open("rank_"+std::to_string(rank)+"_timing.csv");

  if(rank==0) {

    auto constructor_start = std::chrono::high_resolution_clock::now();
    SILC::Client client(true);
    auto constructor_end = std::chrono::high_resolution_clock::now();
    auto delta_t = constructor_end - constructor_start;
    timing_file << rank << "," << "client()" << ","
                << std::chrono::duration<double, std::milli>(delta_t).count()
                << std::endl << std::flush;


    std::string model_key = "mnist_model";
    std::string model_file = "./../mnist_data/mnist_cnn.pt";
    auto model_set_start = std::chrono::high_resolution_clock::now();
    client.set_model_from_file(model_key, model_file, "TORCH", "CPU", 20);
    auto model_set_end = std::chrono::high_resolution_clock::now();
    delta_t = model_set_end - model_set_start;
    timing_file << rank << "," << "model_set" << ","
                << std::chrono::duration<double, std::milli>(delta_t).count()
                << std::endl << std::flush;

    std::string script_key = "mnist_script";
    std::string script_file = "./../mnist_data/data_processing_script.txt";

    auto script_set_start = std::chrono::high_resolution_clock::now();
    client.set_script_from_file(script_key, "CPU", script_file);
    auto script_set_end = std::chrono::high_resolution_clock::now();
    delta_t = script_set_end - script_set_start;
    timing_file << rank << "," << "script_set" << ","
                << std::chrono::duration<double, std::milli>(delta_t).count()
                << std::endl << std::flush;

    auto model_get_start = std::chrono::high_resolution_clock::now();
    std::string_view model = client.get_model(model_key);
    auto model_get_end = std::chrono::high_resolution_clock::now();
    delta_t = model_get_end - model_get_start;
    timing_file << rank << "," << "model_get" << ","
                << std::chrono::duration<double, std::milli>(delta_t).count() 
                << std::endl << std::flush;

    auto script_get_start = std::chrono::high_resolution_clock::now();
    std::string_view script = client.get_script(script_key);
    auto script_get_end = std::chrono::high_resolution_clock::now();
    delta_t = script_get_end - script_get_start;
    timing_file << rank << "," << "script_get" << ","
                << std::chrono::duration<double, std::milli>(delta_t).count()
                << std::endl << std::flush;
  }


  run_mnist("mnist_model", "mnist_script", timing_file);

  if(rank==0)
    std::cout<<"Finished MNIST test."<<std::endl;

  auto main_end = std::chrono::high_resolution_clock::now();
  auto delta_t = main_end - main_start;
  timing_file << rank << "," << "main()" << ","
                << std::chrono::duration<double, std::milli>(delta_t).count()
                << std::endl << std::flush;


  return 0;
}
