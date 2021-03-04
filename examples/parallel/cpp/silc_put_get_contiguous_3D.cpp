#include "client.h"
#include "example_utils.h"
#include <mpi.h>
#include <vector>
#include <string>

template <typename T_send, typename T_recv>
void put_get_3D_array(
		    void (*fill_array)(T_send*, int),
		    std::vector<size_t> dims,
        SILC::TensorType type,
        std::string key_suffix="")
{
  SILC::Client client(use_client());

  //Allocate and fill arrays
  T_send* array =
    (T_send*)malloc(dims[0]*dims[1]*dims[2]*sizeof(T_send));

  T_recv* u_contig_result =
    (T_recv*)malloc(dims[0]*dims[1]*dims[2]*sizeof(T_recv));

  T_recv*** u_nested_result =
    allocate_3D_array<T_recv>(dims[0], dims[1], dims[2]);

  fill_array(array, dims[0]*dims[1]*dims[2]);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string key = "3d_tensor_contig_test_rank_" +
                    std::to_string(rank) + key_suffix;

  /*
  int c=0;
  for(int i = 0; i < dims[0]; i++) {
    for(int j = 0; j < dims[1]; j++) {
      for(int k = 0; k < dims[2]; k++) {
        std::cout.precision(17);
        std::cout<<"Sending value "<<c<<": "
                 <<std::fixed<<array[c]<<std::endl;
        c++;
      }
    }
  }
  */

  client.put_tensor(key, (void*)array, dims, type,
                    SILC::MemoryLayout::contiguous);

  client.unpack_tensor(key, u_contig_result,
                       {dims[0]*dims[1]*dims[2]}, type,
                       SILC::MemoryLayout::contiguous);

  client.unpack_tensor(key, u_nested_result, dims, type,
                       SILC::MemoryLayout::nested);

  /*
  int d = 0;
  for(int i = 0; i < dims[0]; i++) {
    for(int j = 0; j < dims[1]; j++) {
      for(int k = 0; k < dims[2]; k++) {
        std::cout<< "Value " << d
                 << " Sent: " << array[d]
                 <<" Received: " << u_contig_result[d]
                 << std::endl;
        d++;
      }
    }
  }
  */

  if (!is_equal_1D_array<T_send, T_recv>(array, u_contig_result,
                                         dims[0]*dims[1]*dims[2]))
	  throw std::runtime_error("The results do not match for "\
				                     "when unpacking the result into "\
                             "a contiguous array!");

  int e = 0;
  for(int i = 0; i < dims[0]; i++) {
    for(int j = 0; j < dims[1]; j++) {
      for(int k = 0; k < dims[2]; k++) {
        if(u_nested_result[i][j][k]!=array[e]) {
          throw std::runtime_error("The results do not match for "\
				                           "when unpacking the result into "\
                                   "a nested array!");
        }
        e++;
      }
    }
  }

  SILC::TensorType g_type_nested;
  std::vector<size_t> g_dims_nested;
  void* g_nested_result;
  client.get_tensor(key, g_nested_result,
                    g_dims_nested, g_type_nested,
                    SILC::MemoryLayout::nested);
  T_recv*** g_type_nested_result = (T_recv***)g_nested_result;

  if(type!=g_type_nested)
    throw std::runtime_error("The tensor type "\
                             "retrieved with client.get_tensor() "\
                             "does not match the known type.");
  if(g_dims_nested!=dims)
    throw std::runtime_error("The tensor dimensions retrieved "\
                             "client.get_tensor() do not match "\
                             "the known tensor dimensions.");

  /*
  for(int i = 0; i < dims[0]; i++) {
    for(int j = 0; j < dims[1]; j++) {
      for(int k = 0; k < dims[2]; k++) {
        std::cout<< "Value " << i << "," << j << "," << k
                 << " Sent: " << array[i][j][k] <<" Received: "
                 << g_result[i][j][k] << std::endl;
      }
    }
  }
  */

  int g = 0;
  for(int i = 0; i < dims[0]; i++) {
    for(int j = 0; j < dims[1]; j++) {
      for(int k = 0; k < dims[2]; k++) {
        if(g_type_nested_result[i][j][k]!=array[g]) {
          throw std::runtime_error("The results do not match "\
				                           "when using get_tensor() with"\
                                   "a nested destination array!");
        }
        g++;
      }
    }
  }

  SILC::TensorType g_type_contig;
  std::vector<size_t> g_dims_contig;
  void* g_contig_result;
  client.get_tensor(key, g_contig_result,
                    g_dims_contig, g_type_contig,
                    SILC::MemoryLayout::contiguous);

  if(g_dims_contig!=dims)
    throw std::runtime_error("The tensor dimensions retrieved "\
                             "client.get_tensor() do not match "\
                             "the known tensor dimensions.");


  if(type!=g_type_contig)
    throw std::runtime_error("The tensor type "\
                             "retrieved with client.get_tensor() "\
                             "does not match the known type.");

  if (!is_equal_1D_array<T_send, T_recv>(array,
                                         (T_recv*)g_contig_result,
                                         dims[0]*dims[1]*dims[2])) {
	  throw std::runtime_error("The tensor data retrieved with "\
                             "get_tensor() does not match "\
                             "when unpacked into nested array!");
  }

  free(array);
  free(u_contig_result);
  free_3D_array(u_nested_result, dims[0], dims[1]);

  return;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  size_t dim1 = 3;
  size_t dim2 = 2;
  size_t dim3 = 5;

  std::vector<size_t> dims = {dim1, dim2, dim3};

  put_get_3D_array<double,double>(
				  &set_1D_array_floating_point_values<double>,
				  dims, SILC::TensorType::dbl, "_dbl");

  put_get_3D_array<float,float>(
				&set_1D_array_floating_point_values<float>,
				dims, SILC::TensorType::flt, "_flt");

  put_get_3D_array<int64_t,int64_t>(
				    &set_1D_array_integral_values<int64_t>,
				    dims, SILC::TensorType::int64, "_i64");

  put_get_3D_array<int32_t,int32_t>(
				    &set_1D_array_integral_values<int32_t>,
				    dims, SILC::TensorType::int32, "_i32");

  put_get_3D_array<int16_t,int16_t>(
				      &set_1D_array_integral_values<int16_t>,
				      dims, SILC::TensorType::int16, "_i16");

  put_get_3D_array<int8_t,int8_t>(
				      &set_1D_array_integral_values<int8_t>,
				      dims, SILC::TensorType::int8, "_i8");

  put_get_3D_array<uint16_t,uint16_t>(
				      &set_1D_array_integral_values<uint16_t>,
				      dims, SILC::TensorType::uint16, "_ui16");

  put_get_3D_array<uint8_t,uint8_t>(
				      &set_1D_array_integral_values<uint8_t>,
				      dims, SILC::TensorType::uint8, "_ui8");

  std::cout<<"SILC 3D put and get with contiguous "\
             "data example complete."<<std::endl;
  MPI_Finalize();

  return 0;
}
