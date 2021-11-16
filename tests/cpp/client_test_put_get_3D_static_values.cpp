/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Hewlett Packard Enterprise
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "client.h"
#include "client_test_utils.h"
#include <vector>
#include <string>

template <typename T_send, typename T_recv>
void put_get_3D_array(
		    std::vector<size_t> dims,
        SRTensorType type,
        std::string key_suffix="")
{
  SmartRedis::Client client(use_cluster());

  //Allocate and fill arrays
  T_send*** array = allocate_3D_array<T_send>(dims[0], dims[1], dims[2]);
  T_recv*** u_result = allocate_3D_array<T_recv>(dims[0], dims[1], dims[2]);

  int a = 0;
  for(int i = 0; i < dims[0]; i++) {
    for(int j = 0; j < dims[1]; j++) {
      for(int k = 0; k < dims[2]; k++) {
        array[i][j][k] = (T_send)a+2;
        a++;
      }
    }
  }


  std::string key = "3d_tensor_test" +
                    key_suffix;

  /*
  for(int i = 0; i < dims[0]; i++) {
    for(int j = 0; j < dims[1]; j++) {
      for(int k = 0; k < dims[2]; k++) {
        std::cout.precision(17);
        std::cout<<"Sending value "<<i<<","<<j<<","<<k<<": "
                 <<std::fixed<<array[i][j][k]<<std::endl;
      }
    }
  }
  */

  client.put_tensor(key, (void*)array, dims, type, sr_layout_nested);
  client.unpack_tensor(key, u_result, dims, type, sr_layout_nested);

  /*
  for(int i = 0; i < dims[0]; i++) {
    for(int j = 0; j < dims[1]; j++) {
      for(int k = 0; k < dims[2]; k++) {
        std::cout<< "Value " << i << "," << j << "," << k
                 << " Sent: " << array[i][j][k] <<" Received: "
                 << u_result[i][j][k] << std::endl;
      }
    }
  }
  */

  int b = 0;
  for(int i = 0; i < dims[0]; i++) {
    for(int j = 0; j < dims[1]; j++) {
      for(int k = 0; k < dims[2]; k++) {
        if( (array[i][j][k]!=(T_send)(b+2)) ||
            (u_result[i][j][k]!=(T_recv)(b+2)) ) {
              throw std::runtime_error("The results do not match for "\
				                               "the 3d put and get test!");
            }
        b++;
      }
    }
  }

  SRTensorType g_type = sr_tensor_invalid;
  std::vector<size_t> g_dims;
  void* g_result;
  client.get_tensor(key, g_result, g_dims, g_type, sr_layout_nested);
  T_recv*** g_type_result = (T_recv***)g_result;

  if(type!=g_type)
    throw std::runtime_error("The tensor type "\
                             "retrieved with client.get_tensor() "\
                             "does not match the known type.");


  if(g_dims!=dims)
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

  int c = 0;
  for(int i = 0; i < dims[0]; i++) {
    for(int j = 0; j < dims[1]; j++) {
      for(int k = 0; k < dims[2]; k++) {
        if( (array[i][j][k]!=(T_send)(c+2)) ||
            (g_type_result[i][j][k]!=(T_recv)(c+2)) ) {
              throw std::runtime_error("The results do not match for "\
				                               "the 3d put and get test!");
            }
        c++;
      }
    }
  }

  free_3D_array(array, dims[0], dims[1]);
  free_3D_array(u_result, dims[0], dims[1]);
}

int main(int argc, char* argv[]) {

  size_t dim1 = 2;
  size_t dim2 = 6;
  size_t dim3 = 3;

  std::vector<size_t> dims = {dim1, dim2, dim3};

  put_get_3D_array<double,double>(dims, sr_tensor_dbl, "_dbl");
  put_get_3D_array<float,float>(dims, sr_tensor_flt, "_flt");
  put_get_3D_array<int64_t,int64_t>(dims, sr_tensor_int64, "_i64");
  put_get_3D_array<int32_t,int32_t>(dims, sr_tensor_int32, "_i32");
  put_get_3D_array<int16_t,int16_t>(dims, sr_tensor_int16, "_i16");
  put_get_3D_array<int8_t,int8_t>(dims, sr_tensor_int8, "_i8");
  put_get_3D_array<uint16_t,uint16_t>(dims, sr_tensor_uint16, "_ui16");
  put_get_3D_array<uint8_t,uint8_t>(dims, sr_tensor_uint8, "_ui8");


  std::cout<<"3D put and get test using "\
             "static values is complete."<<std::endl;

  return 0;
}
