#include "client.h"
#include "client_test_utils.h"
#include <vector>
#include <string>

template <typename T_send, typename T_recv>
void put_get_1D_array(
		    void (*fill_array)(T_send*, int),
		    std::vector<size_t> dims,
        SILC::TensorType type,
        std::string key_suffix="")
{
  SILC::Client client(use_cluster());

  //Allocate and fill arrays
  T_send* array = (T_send*)malloc(dims[0]*sizeof(T_send));
  T_recv* u_result = (T_recv*)malloc(dims[0]*sizeof(T_recv));

  fill_array(array, dims[0]);

  std::string key = "1D_tensor_test" +
                     key_suffix;

  /*
  for(int i = 0; i < dims[0]; i++) {
      std::cout.precision(17);
      std::cout<<"Sending value "<<i<<": "
               <<std::fixed<<array[i]<<std::endl;
  }
  */
  client.put_tensor(key, (void*)array, dims,
                    type, SILC::MemoryLayout::nested);

  if(!client.key_exists(key))
    throw std::runtime_error("The key does not exist in the database.");

  client.unpack_tensor(key, u_result, dims,
                       type, SILC::MemoryLayout::nested);
  /*
  for(int i = 0; i < dims[0]; i++) {
      std::cout<< "Value " << i
               << " Sent: " << array[i] <<" Received: "
               << u_result[i] << std::endl;
  }
  */

  if (!is_equal_1D_array<T_send, T_recv>(array, u_result,
                                         dims[0]))
	  throw std::runtime_error("The results do not match for "\
				                     "the 1D put and get test!");

  SILC::TensorType g_type;
  std::vector<size_t> g_dims;
  void* g_result;
  client.get_tensor(key, g_result, g_dims,
                    g_type, SILC::MemoryLayout::nested);
  T_recv* g_type_result = (T_recv*)g_result;

  /*
  for(int i = 0; i < dims[0]; i++) {
      std::cout<< "Value " << i
               << " Sent: " << array[i] <<" Received: "
               << g_type_result[i] << std::endl;
  }
  */
  if(type!=g_type)
    throw std::runtime_error("The tensor type "\
                             "retrieved with client.get_tensor() "\
                             "does not match the known type.");

  if(g_dims!=dims)
    throw std::runtime_error("The tensor dimensions retrieved "\
                             "client.get_tensor() do not match "\
                             "the known tensor dimensions.");

  if (!is_equal_1D_array<T_send, T_recv>(array, g_type_result,
                                         dims[0]))
	  throw std::runtime_error("The results do not match for "\
				                     "the 1D put and get test!");

  free_1D_array(array);
  free_1D_array(u_result);

  return;
}

int main(int argc, char* argv[]) {

  size_t dim1 = 10;
  std::vector<size_t> dims = {dim1};

  put_get_1D_array<double,double>(
				  &set_1D_array_floating_point_values<double>,
				  dims, SILC::TensorType::dbl, "_dbl");

  put_get_1D_array<float,float>(
				&set_1D_array_floating_point_values<float>,
				dims, SILC::TensorType::flt, "_flt");

  put_get_1D_array<int64_t,int64_t>(
				    &set_1D_array_integral_values<int64_t>,
				    dims, SILC::TensorType::int64, "_i64");

  put_get_1D_array<int32_t,int32_t>(
				    &set_1D_array_integral_values<int32_t>,
				    dims, SILC::TensorType::int32, "_i32");

  put_get_1D_array<int16_t,int16_t>(
				      &set_1D_array_integral_values<int16_t>,
				      dims, SILC::TensorType::int16, "_i16");

  put_get_1D_array<int8_t,int8_t>(
				      &set_1D_array_integral_values<int8_t>,
				      dims, SILC::TensorType::int8, "_i8");

  put_get_1D_array<uint16_t,uint16_t>(
				      &set_1D_array_integral_values<uint16_t>,
				      dims, SILC::TensorType::uint16, "_ui16");

  put_get_1D_array<uint8_t,uint8_t>(
				      &set_1D_array_integral_values<uint8_t>,
				      dims, SILC::TensorType::uint8, "_ui8");

  std::cout<<"1D put and test complete"<<std::endl;

  return 0;
}
