#include "client.h"
#include "client_test_utils.h"
#include <mpi.h>

template <typename T_send, typename T_recv>
void put_get_2D_array(
		    void (*fill_array)(T_send**, int, int),
		    int dim1, int dim2,
        std::string dtype,
        std::string key_suffix="")
{
  SmartSimClient client(true);

  T_send** array = allocate_2D_array<T_send>(dim1, dim2);
  T_recv** result = allocate_2D_array<T_recv>(dim1, dim2);

  fill_array(array, dim1, dim2);

  int dims[2] = {dim1, dim2};

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string key = "2d_test_rank_" + std::to_string(rank) + key_suffix;

  for(int i = 0; i < dim1; i++) {
    for(int j = 0; j < dim2; j++) {
      std::cout.precision(17);
      std::cout<<"Sending value "<<i<<","<<j<<": "<<std::fixed<<array[i][j]<<std::endl;
    }
  }

  std::string_view key_view(key.c_str(), key.length());
  std::string_view type_view(dtype.c_str(), dtype.length());

  std::cout<<"Rank "<<rank<<" starting put."<<std::endl<<std::flush;
  client._put_redisai_tensor<T_send>(key_view, type_view, (int*)dims, 2, (void*)array);
  std::cout<<"Rank "<<rank<<" finished put."<<std::endl<<std::flush;

  MPI_Barrier(MPI_COMM_WORLD);

  if(!client.key_exists(key.c_str()))
    throw std::runtime_error("Key existence could not be "\
			     "verified with key_exists()");

  client._get_redisai_tensor<T_recv>(key_view, result);

  for(int i = 0; i < dim1; i++) {
    for(int j = 0; j < dim2; j++) {
      //std::cout treats uint8_t as a char, so we have to
      //add a unary operator to make sure we can see the number.
        std::cout<<"Value "<<i<<","<<j<<" Sent: "<<array[i][j]
                 <<" Received: "<<result[i][j]<<std::endl;
    }
  }
  if (!is_equal_2D_array<T_send, T_recv>(array, result, dim1, dim2))
	throw std::runtime_error("The results do not match for "\
				 "the 2d put and get test!");

  
  //client.set_model("test_model", "TORCH", "CPU", "/Users/mellis/Poseidon-Ideation/PSD-Inference-Clients/code/mnist_cnn.pt");
  // Set the input tensor outside of this program for ease
  //std::vector<std::string_view> model_inputs;
  //model_inputs.push_back("test_input");
  //std::vector<std::string_view> model_outputs;
  //model_outputs.push_back("test_output");
  //client.run_model("test_model", model_inputs, model_outputs);
  //MPI_Barrier(MPI_COMM_WORLD);

  //client.set_script("test_script", "CPU", "/Users/mellis/Poseidon-Ideation/PSD-Inference-Clients/code/data_processing_script.txt");

  free_2D_array(array, dim1);
  free_2D_array(result, dim1);

  return;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int dim_1 = atoi(argv[1]);
  int dim_2 = dim_1;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  put_get_2D_array<double,double>(
				  &set_2D_array_floating_point_values<double>,
				  dim_1, dim_2, "DOUBLE", "_dbl");

  put_get_2D_array<float,float>(
				&set_2D_array_floating_point_values<float>,
				dim_1, dim_2, "FLOAT", "_flt");

  /*
  sendFunction = &SmartSimClient::put_array_int64;
  recvFunction = &SmartSimClient::get_array_int64;
  put_get_2D_array<int64_t,int64_t>(sendFunction, recvFunction,
				    &set_2D_array_integral_values<int64_t>,
				    dim_1, dim_2, "_i64");

  sendFunction = &SmartSimClient::put_array_int32;
  recvFunction = &SmartSimClient::get_array_int32;
  put_get_2D_array<int32_t,int32_t>(sendFunction, recvFunction,
				    &set_2D_array_integral_values<int32_t>,
				    dim_1, dim_2, "_i32");

  sendFunction = &SmartSimClient::put_array_uint64;
  recvFunction = &SmartSimClient::get_array_uint64;
  put_get_2D_array<uint64_t,uint64_t>(sendFunction, recvFunction,
				      &set_2D_array_integral_values<uint64_t>,
				      dim_1, dim_2, "_ui64");

  sendFunction = &SmartSimClient::put_array_uint32;
  recvFunction = &SmartSimClient::get_array_uint32;
  put_get_2D_array<uint32_t,uint32_t>(sendFunction, recvFunction,
				      &set_2D_array_integral_values<uint32_t>,
				      dim_1, dim_2, "_ui32");
  */
  MPI_Finalize();

  if(rank==0)
    std::cout<<"Finished 2D put and get tests."<<std::endl;

  return 0;
}
