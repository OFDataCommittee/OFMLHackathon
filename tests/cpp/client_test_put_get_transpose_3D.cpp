#include "client.h"
#include "client_test_utils.h"
#include <vector>
#include <string>

inline size_t _c_index(const std::vector<size_t>& dims,
                       const std::vector<size_t> dim_positions)
{
    /* This function will return the row major
    index in a contiguous memory array corresponding
    to the dimensions and dimension positions.
    */
    size_t position = 0;
    size_t sum_product;
    for(size_t k = 0; k < dims.size(); k++) {
        sum_product = dim_positions[k];
        for(size_t m = k+1; m < dims.size(); m++) {
            sum_product *= dims[m];
        }
        position += sum_product;
    }
    return position;
}


template <typename T_send, typename T_recv>
void put_get_3D_array(
		    void (*fill_array)(T_send*, int),
		    std::vector<size_t> dims,
        SILC::TensorType type,
        std::string key_suffix="",
        SILC::MemoryLayout send_direction=SILC::MemoryLayout::contiguous,
        SILC::MemoryLayout recv_direction=SILC::MemoryLayout::contiguous)
{
  SILC::Client client(true);

  //Allocate and fill arrays
  T_send* array =
    (T_send*)malloc(dims[0]*dims[1]*dims[2]*sizeof(T_send));

  T_recv* u_array =
    (T_recv*)malloc(dims[0]*dims[1]*dims[2]*sizeof(T_recv));

  fill_array(array, dims[0]*dims[1]*dims[2]);

  int rank = 0;

  std::string key = "3d_tensor_transpose_test_rank_" +
                    std::to_string(rank) + key_suffix;

  /*
  size_t c=0;
  for(size_t i = 0; i < dims[0]; i++) {
    for(size_t j = 0; j < dims[1]; j++) {
      for(size_t k = 0; k < dims[2]; k++) {
        std::cout.precision(17);
        std::cout<<"Sending value "<<c<<": "
                 <<std::fixed<<array[c]<<std::endl;
        c++;
      }
    }
  }
  */

  client.put_tensor(key, (void*)array, dims, type,
                    send_direction);

  client.unpack_tensor(key, u_array,
                       {dims[0]*dims[1]*dims[2]}, type,
                       recv_direction);

  /*
  size_t d = 0;
  for(size_t i = 0; i < dims[0]; i++) {
    for(size_t j = 0; j < dims[1]; j++) {
      for(size_t k = 0; k < dims[2]; k++) {
        std::cout<< "Value " << d
                 << " Sent: " << array[d]
                 <<" Received: " << u_array[d]
                 << std::endl;
        d++;
      }
    }
  }
  */

  size_t u_index;
  size_t index;
  std::vector<size_t> r_dims(dims.rbegin(), dims.rend());
  for(size_t i = 0; i < dims[0]; i++) {
    for(size_t j = 0; j < dims[1]; j++) {
      for(size_t k = 0; k < dims[2]; k++) {
        if(send_direction == SILC::MemoryLayout::fortran_contiguous &&
           recv_direction == SILC::MemoryLayout::contiguous) {
          u_index = _c_index(dims, {i,j,k});
          index = _c_index(r_dims, {k,j,i});
        }
        else if(send_direction == SILC::MemoryLayout::contiguous &&
                recv_direction == SILC::MemoryLayout::fortran_contiguous) {
          index = _c_index(dims, {i,j,k});
          u_index = _c_index(r_dims, {k,j,i});
        }
        else {
          throw std::runtime_error("Invalid test configuration.");
        }
        if(u_array[u_index]!=array[index]) {
          throw std::runtime_error("The returned matrix is not a "\
                                     "transpose of the original matrix.");
        }
      }
    }
  }

  SILC::TensorType g_type_transpose;
  std::vector<size_t> g_dims_transpose;
  void* g_array;
  client.get_tensor(key, g_array,
                    g_dims_transpose, g_type_transpose,
                    recv_direction);

  if(g_dims_transpose!=dims)
    throw std::runtime_error("The tensor dimensions retrieved "\
                             "client.get_tensor() do not match "\
                             "the known tensor dimensions.");


  if(type!=g_type_transpose)
    throw std::runtime_error("The tensor type "\
                             "retrieved with client.get_tensor() "\
                             "does not match the known type.");

  size_t g_index;
  for(size_t i = 0; i < dims[0]; i++) {
    for(size_t j = 0; j < dims[1]; j++) {
      for(size_t k = 0; k < dims[2]; k++) {
        if(send_direction == SILC::MemoryLayout::fortran_contiguous &&
           recv_direction == SILC::MemoryLayout::contiguous) {
          g_index = _c_index(dims, {i,j,k});
          index = _c_index(r_dims, {k,j,i});
        }
        else if(send_direction == SILC::MemoryLayout::contiguous &&
          recv_direction == SILC::MemoryLayout::fortran_contiguous) {
          index = _c_index(dims, {i,j,k});
          g_index = _c_index(r_dims, {k,j,i});
        }
        else {
          throw std::runtime_error("Invalid test configuration.");
        }
        if(((T_recv*)g_array)[g_index]!=array[index]) {
          throw std::runtime_error("The returned matrix is not a "\
                                     "transpose of the original matrix.");
        }
      }
    }
  }

  free(array);
  free(u_array);
  return;
}

int main(int argc, char* argv[]) {

  /* This tests whether the conversion from
  column major to row major is implemented
  correctly in the client.  To do this,
  we put a 3D array assuming it is fortran
  (column major) and retrieve assuming
  it is c-style (row major).  If the
  two tensors are the transpose of each other
  it has been implemented correctly.
  */

  size_t dim1 = 4;
  size_t dim2 = 3;
  size_t dim3 = 2;

  std::vector<size_t> dims = {dim1, dim2, dim3};

  /* Test conversion on the put side
  */
  put_get_3D_array<double,double>(
				  &set_1D_array_floating_point_values<double>,
				  dims, SILC::TensorType::dbl, "_dbl",
          SILC::MemoryLayout::fortran_contiguous,
          SILC::MemoryLayout::contiguous);

  put_get_3D_array<float,float>(
				&set_1D_array_floating_point_values<float>,
				dims, SILC::TensorType::flt, "_flt",
        SILC::MemoryLayout::fortran_contiguous,
        SILC::MemoryLayout::contiguous);

  put_get_3D_array<int64_t,int64_t>(
				    &set_1D_array_integral_values<int64_t>,
				    dims, SILC::TensorType::int64, "_i64",
            SILC::MemoryLayout::fortran_contiguous,
            SILC::MemoryLayout::contiguous);

  put_get_3D_array<int32_t,int32_t>(
				    &set_1D_array_integral_values<int32_t>,
				    dims, SILC::TensorType::int32, "_i32",
            SILC::MemoryLayout::fortran_contiguous,
            SILC::MemoryLayout::contiguous);

  put_get_3D_array<int16_t,int16_t>(
				      &set_1D_array_integral_values<int16_t>,
				      dims, SILC::TensorType::int16, "_i16",
              SILC::MemoryLayout::fortran_contiguous,
              SILC::MemoryLayout::contiguous);

  put_get_3D_array<int8_t,int8_t>(
				      &set_1D_array_integral_values<int8_t>,
				      dims, SILC::TensorType::int8, "_i8",
              SILC::MemoryLayout::fortran_contiguous,
              SILC::MemoryLayout::contiguous);

  put_get_3D_array<uint16_t,uint16_t>(
				      &set_1D_array_integral_values<uint16_t>,
				      dims, SILC::TensorType::uint16, "_ui16",
              SILC::MemoryLayout::fortran_contiguous,
              SILC::MemoryLayout::contiguous);

  put_get_3D_array<uint8_t,uint8_t>(
				      &set_1D_array_integral_values<uint8_t>,
				      dims, SILC::TensorType::uint8, "_ui8",
              SILC::MemoryLayout::fortran_contiguous,
              SILC::MemoryLayout::contiguous);

  /* Test conversion on the get side
  */
  put_get_3D_array<double,double>(
				  &set_1D_array_floating_point_values<double>,
				  dims, SILC::TensorType::dbl, "_dbl",
          SILC::MemoryLayout::contiguous,
          SILC::MemoryLayout::fortran_contiguous);

  put_get_3D_array<float,float>(
				&set_1D_array_floating_point_values<float>,
				dims, SILC::TensorType::flt, "_flt",
        SILC::MemoryLayout::contiguous,
        SILC::MemoryLayout::fortran_contiguous);

  put_get_3D_array<int64_t,int64_t>(
				    &set_1D_array_integral_values<int64_t>,
				    dims, SILC::TensorType::int64, "_i64",
            SILC::MemoryLayout::contiguous,
            SILC::MemoryLayout::fortran_contiguous);

  put_get_3D_array<int32_t,int32_t>(
				    &set_1D_array_integral_values<int32_t>,
				    dims, SILC::TensorType::int32, "_i32",
            SILC::MemoryLayout::contiguous,
            SILC::MemoryLayout::fortran_contiguous);

  put_get_3D_array<int16_t,int16_t>(
				      &set_1D_array_integral_values<int16_t>,
				      dims, SILC::TensorType::int16, "_i16",
              SILC::MemoryLayout::contiguous,
              SILC::MemoryLayout::fortran_contiguous);

  put_get_3D_array<int8_t,int8_t>(
				      &set_1D_array_integral_values<int8_t>,
				      dims, SILC::TensorType::int8, "_i8",
              SILC::MemoryLayout::contiguous,
              SILC::MemoryLayout::fortran_contiguous);

  put_get_3D_array<uint16_t,uint16_t>(
				      &set_1D_array_integral_values<uint16_t>,
				      dims, SILC::TensorType::uint16, "_ui16",
              SILC::MemoryLayout::contiguous,
              SILC::MemoryLayout::fortran_contiguous);

  put_get_3D_array<uint8_t,uint8_t>(
				      &set_1D_array_integral_values<uint8_t>,
				      dims, SILC::TensorType::uint8, "_ui8",
              SILC::MemoryLayout::contiguous,
              SILC::MemoryLayout::fortran_contiguous);

  std::cout<<"3D put and get to test matrix "\
             "transpose complete."<<std::endl;

  return 0;
}
