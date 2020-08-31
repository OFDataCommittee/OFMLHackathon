#include "client.h"
#include "client_test_utils.h"
#include <mpi.h>

template <typename T_send, typename T_recv>
void put_get_2D_array(
		      void (*fill_array)(T_send***, int, int, int),
		    int dim1, int dim2,
        std::string dtype,
        std::string key_suffix="")
{
  SmartSimClient client(true);

  T_send**** array = (T_send****)malloc(1*sizeof(T_send***));
  array[0] = (T_send***)allocate_3D_array<T_send>(1, dim1, dim2);
  fill_array((T_send***)(array[0]), 1, dim1, dim2);

  int dims[4] = {1, 1, dim1, dim2};
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);


  std::string key = "mnist_{rank_" + std::to_string(rank) + "}";
  std::string_view key_view(key.c_str(), key.length());
  std::string_view type_view(dtype.c_str(), dtype.length());

  std::cout<<"Rank "<<rank<<" starting put."<<std::endl<<std::flush;
  client._put_redisai_tensor<T_send>(key_view, type_view, (int*)dims, 4, (void*)array);
  std::cout<<"Rank "<<rank<<" finished put."<<std::endl<<std::flush;

  MPI_Barrier(MPI_COMM_WORLD);

  //Set the input tensor outside of this program for ease
  std::string out_key = "mnist_out_{rank_" + std::to_string(rank) + "}";
  std::string_view out_key_view(out_key.c_str(), out_key.length());


  // set model key
  std::string model_key = "mnist_model_{rank_" + std::to_string(rank) + "}";
  std::string_view model_key_view(model_key.c_str(), model_key.length());

  std::vector<std::string_view> model_inputs;
  model_inputs.push_back(key_view);
  std::vector<std::string_view> model_outputs;
  model_outputs.push_back(out_key_view);
  client.run_model(model_key_view, model_inputs, model_outputs);
  MPI_Barrier(MPI_COMM_WORLD);

  //  client.set_script("test_script",
  //		    "GPU","/lus/snx11260/spartee/Poseidon-Ideation/PSD-Inference-Clients/code/data_processing_script.txt");


  float** result = allocate_2D_array<float>(1, 10);
  client._get_redisai_tensor<T_recv>(out_key_view, result);

  free_2D_array(array, dim1);
  free_2D_array(result, dim1);

  return;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int dim_1 = 28;
  int dim_2 = 28;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  SmartSimClient set_client(true);

  std::string model_key = "mnist_model_{rank_" + std::to_string(rank) + "}";
  std::string_view model_key_view(model_key.c_str(), model_key.length());

  set_client.set_model(model_key_view, "TORCH", "GPU",
		       "/lus/snx11260/spartee/Poseidon-Ideation/PSD-Inference-Clients/code/mnist_cnn.pt");

  MPI_Barrier(MPI_COMM_WORLD);

  
  put_get_2D_array<uint8_t, uint8_t>(
				  &set_3D_array_integral_values<uint8_t>,
				  dim_1, dim_2, "UINT8", "_uint8");
  
  MPI_Finalize();

  if(rank==0)
    std::cout<<"Finished 2D put and get tests."<<std::endl;

  return 0;
}
