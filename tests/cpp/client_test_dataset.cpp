#include "client.h"
#include "dataset.h"
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
