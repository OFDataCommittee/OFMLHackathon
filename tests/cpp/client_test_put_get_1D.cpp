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
void put_get_1D_array(
            void (*fill_array)(T_send*, int),
        std::vector<size_t> dims,
        SRTensorType type,
        std::string key_suffix="")
{
  SmartRedis::Client client(use_cluster());

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
  client.put_tensor(key, (void*)array, dims, type, SRMemLayoutNested);

  if(!client.key_exists(get_prefix() + key))
    throw std::runtime_error("The key does not exist in the database.");

  client.unpack_tensor(key, u_result, dims, type, SRMemLayoutNested);
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

  SRTensorType g_type;
  std::vector<size_t> g_dims;
  void* g_result;
  client.get_tensor(key, g_result, g_dims, g_type, SRMemLayoutNested);
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

  if (!is_equal_1D_array<T_send, T_recv>(array, g_type_result, dims[0]))
      throw std::runtime_error("The results do not match for "\
                                     "the 1D put and get test!");

  free_1D_array(array);
  free_1D_array(u_result);
}

int main(int argc, char* argv[]) {

  size_t dim1 = 10;
  std::vector<size_t> dims = {dim1};

  put_get_1D_array<double,double>(
                  &set_1D_array_floating_point_values<double>,
                  dims, SRTensorTypeDouble, "_dbl");

  put_get_1D_array<float,float>(
                &set_1D_array_floating_point_values<float>,
                dims, SRTensorTypeFloat, "_flt");

  put_get_1D_array<int64_t,int64_t>(
                    &set_1D_array_integral_values<int64_t>,
                    dims, SRTensorTypeInt64, "_i64");

  put_get_1D_array<int32_t,int32_t>(
                    &set_1D_array_integral_values<int32_t>,
                    dims, SRTensorTypeInt32, "_i32");

  put_get_1D_array<int16_t,int16_t>(
                      &set_1D_array_integral_values<int16_t>,
                      dims, SRTensorTypeInt16, "_i16");

  put_get_1D_array<int8_t,int8_t>(
                      &set_1D_array_integral_values<int8_t>,
                      dims, SRTensorTypeInt8, "_i8");

  put_get_1D_array<uint16_t,uint16_t>(
                      &set_1D_array_integral_values<uint16_t>,
                      dims, SRTensorTypeUint16, "_ui16");

  put_get_1D_array<uint8_t,uint8_t>(
                      &set_1D_array_integral_values<uint8_t>,
                      dims, SRTensorTypeUint8, "_ui8");

  std::cout<<"1D put and test complete"<<std::endl;

  return 0;
}
