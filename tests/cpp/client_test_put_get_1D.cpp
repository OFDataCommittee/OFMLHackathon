#include "client.h"
#include "client_test_utils.h"
#include <mpi.h>
#include <vector>
#include <string>

template <typename T_send, typename T_recv>
void put_get_1D_array(
		    void (*fill_array)(T_send*, int),
		    std::vector<int> dims,
        std::string type,
        std::string key_suffix="")
{
  SmartSimClient client(false);

  //Allocate and fill arrays
  T_send* array = (T_send*)malloc(dims[0]*sizeof(T_send));
  T_recv* result = (T_recv*)malloc(dims[0]*sizeof(T_recv));
  fill_array(array, dims[0]);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string key = "1D_tensor_test_rank_" +
                    std::to_string(rank) + key_suffix;

  for(int i = 0; i < dims[0]; i++) {
      std::cout.precision(17);
      std::cout<<"Sending value "<<i<<": "
               <<std::fixed<<array[i]<<std::endl;
  }

  client.put_tensor(key, type, (void*)array, dims);
  client.get_tensor(key, type, result, dims);

  for(int i = 0; i < dims[0]; i++) {
      std::cout<< "Value " << i
               << " Sent: " << array[i] <<" Received: "
               << result[i] << std::endl;
  }
  if (!is_equal_1D_array<T_send, T_recv>(array, result,
                                         dims[0]))
	  throw std::runtime_error("The results do not match for "\
				                     "the 1D put and get test!");

  free_1D_array(array);
  free_1D_array(result);

  return;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int dim1 = 10;
  std::vector<int> dims = {dim1};

  put_get_1D_array<double,double>(
				  &set_1D_array_floating_point_values<double>,
				  dims, "DOUBLE", "_dbl");

  put_get_1D_array<float,float>(
				&set_1D_array_floating_point_values<float>,
				dims, "FLOAT", "_flt");
  std::cout<<"starting int64"<<std::endl;
  put_get_1D_array<int64_t,int64_t>(
				    &set_1D_array_integral_values<int64_t>,
				    dims, "INT64", "_i64");

  put_get_1D_array<int32_t,int32_t>(
				    &set_1D_array_integral_values<int32_t>,
				    dims, "INT32", "_i32");

  put_get_1D_array<int16_t,int16_t>(
				      &set_1D_array_integral_values<int16_t>,
				      dims, "INT16", "_i16");

  put_get_1D_array<int8_t,int8_t>(
				      &set_1D_array_integral_values<int8_t>,
				      dims, "INT8", "_i8");

  put_get_1D_array<uint16_t,uint16_t>(
				      &set_1D_array_integral_values<uint16_t>,
				      dims, "UINT16", "_ui16");

  put_get_1D_array<uint8_t,uint8_t>(
				      &set_1D_array_integral_values<uint8_t>,
				      dims, "UINT8", "_ui8");

  MPI_Finalize();

  return 0;
}
