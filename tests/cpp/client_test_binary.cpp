#include "client.h"
#include <mpi.h>
#include "client_test_utils.h"

template <typename T_send, typename T_recv>
void put_get_1D_array(void (*fill_array)(T_send*, int),
                      int dim1,
                      std::string dtype,
                      std::string key_suffix="")
{

  SmartSimClient client(false);

  T_send* array = new T_send[dim1];
  T_recv* result = new T_recv[dim1];

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  fill_array(array, dim1);
  int dims[1] = {dim1};

  std::string key = "1d_test_rank_"+std::to_string(rank) + key_suffix;
  std::string_view key_view(key.c_str(), key.length());
  std::string_view type_view(dtype.c_str(), dtype.length());
  std::string dims_string = std::to_string(dim1);

  for(int i = 0; i < dim1; i++) {
    std::cout.precision(17);
    std::cout<<"Sending value "<<i<<": "<<std::fixed<<array[i]<<std::endl;
  }

  std::cout<<"Rank "<<rank<<" starting put."<<std::endl<<std::flush;

  client.put_tensor<T_send>(key_view, type_view, (int*)dims, 1, (void*)array);

  std::cout<<"Rank "<<rank<<" finished put."<<std::endl<<std::flush;

  if(!client.key_exists(key.c_str()))
    throw std::runtime_error("Key existence could not be "\
			     "verified with key_exists()");

  client.get_tensor<T_recv>(key_view, result);

  for(int i = 0; i < dim1; i++) {
    //std::cout treats uint8_t as a char, so we have to
    //add a unary operator to make sure we can see the number.
    if(sizeof(T_recv) == 1)
      std::cout<<"Value "<<i<<" Sent: "<<+array[i]
               <<" Received: "<<+result[i]<<std::endl;
    else
      std::cout<<"Value "<<i<<" Sent: "<<array[i]
               <<" Received: "<<result[i]<<std::endl;
  }

  if (!is_equal_1D_array<T_send, T_recv>(array, result, dim1))
    throw std::runtime_error("The arrays don't match");

  free_1D_array(array);
  free_1D_array(result);

  MPI_Barrier(MPI_COMM_WORLD);

  return;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int dim_1 = atoi(argv[1]);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  put_get_1D_array<double,double>(
          &set_1D_array_floating_point_values<double>,
				  dim_1, "DOUBLE", "_dbl");

  put_get_1D_array<float,float>(
  				&set_1D_array_floating_point_values<float>,
	  			dim_1, "FLOAT", "_flt");

  put_get_1D_array<int64_t,int64_t>(
				  &set_1D_array_integral_values<int64_t>,
				  dim_1, "INT64", "_i64");

  put_get_1D_array<int32_t,int32_t>(
				  &set_1D_array_integral_values<int32_t>,
				  dim_1, "INT32", "_i32");

  put_get_1D_array<uint8_t,uint8_t>(
				    &set_1D_array_integral_values<uint8_t>,
				    dim_1, "UINT8", "_ui8");

  put_get_1D_array<uint16_t,uint16_t>(
				    &set_1D_array_integral_values<uint16_t>,
				    dim_1, "UINT16", "_ui16");

  MPI_Finalize();

  if(rank==0)
    std::cout<<"Finished 1D put and get tests."<<std::endl;

  return 0;
}
