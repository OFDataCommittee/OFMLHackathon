/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

#include "c_client.h"
#include "c_client_test_utils.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "stdint.h"
#include "srexception.h"

bool cluster = true;

/* This function is a data type agnostic put and
get for 3D tensors.  The result vector
is filled with the fetched tensor.  Accuracy
checking is done outside of this function because
the type is not known.
*/
int put_get_2D_tensor(void* client,
                      void* tensor,
                      size_t* dims,
                      size_t n_dims,
                      void** result,
                      char* key_suffix,
                      size_t key_suffix_length,
                      SRTensorType type)
{
  char* prefix_str = "2D_tensor_test";

  size_t prefix_str_length = strlen(prefix_str);

  int key_length = prefix_str_length +
                   key_suffix_length;
  char* key = (char*)malloc((key_length+1)*sizeof(char));

  int pos;
  pos = 0;
  memcpy(&key[pos], prefix_str, prefix_str_length);
  pos += prefix_str_length;
  memcpy(&key[pos], key_suffix, key_suffix_length);
  pos += key_suffix_length;
  key[pos] = 0;

  SRTensorType g_type;
  size_t* g_dims;
  size_t g_n_dims;

  SRMemoryLayout layout = SRMemLayoutNested;
  if (SRNoError != put_tensor(client, key, key_length,
                          (void*)tensor, dims, n_dims, type, layout)) {
    return -1;
  }

  if (SRNoError != get_tensor(client, key, key_length,
                          result, &g_dims, &g_n_dims,
                          &g_type, layout)) {
    return -1;
  }

  int r_value = 0;
  if(g_n_dims!=n_dims) {
    printf("%s", "The number of fetched dimensions with "\
                 "client.get_tensor() does not match "\
                 "known length.\n");
    r_value = -1;
  }

  for(size_t i=0; i<n_dims; i++) {
    if(g_dims[i]!=dims[i]) {
      printf("%s", "The fetched dimensions with "\
                   "client.get_tensor() do not match "\
                   "the known tensor dimensions.\n");
      r_value = -1;
    }
  }

  if(g_type!=type) {
    printf("%s", "The fetched type with "\
                 "client.get_tensor() does not match "\
                  "the known tensor type.\n");
    r_value = -1;
  }

  free(key);
  return r_value;
}

/* This function puts and gets a 2D tensor of double
values.  If the sent and received tensors do not match,
a non-zero value is returned.
*/
int put_get_2D_tensor_double(size_t* dims, size_t n_dims,
                             char* key_suffix, int key_suffix_length)
{
  void* client = NULL;
  const char* logger_name = "put_get_2D_tensor_double";
  size_t cid_len = strlen(logger_name);
  if (SRNoError != SimpleCreateClient(logger_name, cid_len, &client))
    return -1;

  double** tensor = (double**)malloc(dims[0]*sizeof(double*));
  double** result = 0;

  for(int i=0; i<dims[0]; i++) {
    tensor[i] = (double*)malloc(dims[1]*sizeof(double));
  }

  for(int i=0; i<dims[0]; i++)
    for(int j=0; j<dims[1]; j++)
      tensor[i][j] = ((double)rand())/(double)RAND_MAX;

  int r_value = 0;
  r_value = put_get_2D_tensor(client, (void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              SRTensorTypeDouble);

  for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++) {
      if(tensor[i][j]!=result[i][j]) {
        printf("%s", "The tensors do not match!");
        r_value = -1;
      }
    }
  }

  for(int i=0; i<dims[0]; i++) {
    free(tensor[i]);
  }
  free(tensor);
  if (SRNoError != DeleteCClient(&client))
    return -1;
  return r_value;
}

/* This function puts and gets a 2D tensor of float
values.  If the sent and received tensors do not match,
a non-zero value is returned.
*/
int put_get_2D_tensor_float(size_t* dims, size_t n_dims,
                            char* key_suffix,
                            int key_suffix_length)
{
  void* client = NULL;
  const char* logger_name = "put_get_2D_tensor_float";
  size_t cid_len = strlen(logger_name);
  if (SRNoError != SimpleCreateClient(logger_name, cid_len, &client))
    return -1;

  float** tensor = (float**)malloc(dims[0]*sizeof(float*));
  float** result = 0;

  for(int i=0; i<dims[0]; i++) {
    tensor[i] = (float*)malloc(dims[1]*sizeof(float));
  }

  for(int i=0; i<dims[0]; i++)
    for(int j=0; j<dims[1]; j++)
      tensor[i][j] = ((float)rand())/(float)RAND_MAX;

  int r_value = 0;
  r_value = put_get_2D_tensor(client, (void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              SRTensorTypeFloat);

  for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++) {
      if(tensor[i][j]!=result[i][j]) {
        printf("%s", "The tensors do not match!");
        r_value = -1;
      }
    }
  }

  for(int i=0; i<dims[0]; i++) {
    free(tensor[i]);
  }
  free(tensor);
  if (SRNoError != DeleteCClient(&client))
    return -1;
  return r_value;
}

/* This function puts and gets a 2D tensor of int8_t
values.  If the sent and received tensors do not match,
a non-zero value is returned.
*/
int put_get_2D_tensor_i8(size_t* dims, size_t n_dims,
                         char* key_suffix,
                         int key_suffix_length)
{
  void* client = NULL;
  const char* logger_name = "put_get_2D_tensor_i8";
  size_t cid_len = strlen(logger_name);
  if (SRNoError != SimpleCreateClient(logger_name, cid_len, &client))
    return -1;

  int8_t** tensor = (int8_t**)malloc(dims[0]*sizeof(int8_t*));
  int8_t** result = 0;

  for(int i=0; i<dims[0]; i++) {
    tensor[i] = (int8_t*)malloc(dims[1]*sizeof(int8_t));
  }

  for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++) {
      tensor[i][j] = rand()%INT8_MAX;
      if(rand()%2)
        tensor[i][j] *= -1;
    }
  }

  int r_value = 0;
  r_value = put_get_2D_tensor(client, (void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              SRTensorTypeInt8);

for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++){
      if(tensor[i][j]!=result[i][j]) {
        printf("%s", "The tensors do not match!");
        r_value = -1;
      }
    }
  }

  for(int i=0; i<dims[0]; i++) {
    free(tensor[i]);
  }
  free(tensor);
  if (SRNoError != DeleteCClient(&client))
    return -1;
  return r_value;
}

/* This function puts and gets a 2D tensor of int16_t
values.  If the sent and received tensors do not match,
a non-zero value is returned.
*/
int put_get_2D_tensor_i16(size_t* dims, size_t n_dims,
                         char* key_suffix,
                         int key_suffix_length)
{
  void* client = NULL;
  const char* logger_name = "put_get_2D_tensor_i16";
  size_t cid_len = strlen(logger_name);
  if (SRNoError != SimpleCreateClient(logger_name, cid_len, &client))
    return -1;

  int16_t** tensor = (int16_t**)malloc(dims[0]*sizeof(int16_t*));
  int16_t** result = 0;

  for(int i=0; i<dims[0]; i++) {
    tensor[i] = (int16_t*)malloc(dims[1]*sizeof(int16_t));
  }

  for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++) {
      tensor[i][j] = rand()%INT16_MAX;
      if(rand()%2)
        tensor[i][j] *= -1;
    }
  }

  int r_value = 0;
  r_value = put_get_2D_tensor(client, (void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              SRTensorTypeInt16);

  for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++){
      if(tensor[i][j]!=result[i][j]) {
        printf("%s", "The tensors do not match!");
        r_value = -1;
      }
    }
  }

  for(int i=0; i<dims[0]; i++) {
    free(tensor[i]);
  }
  free(tensor);
  if (SRNoError != DeleteCClient(&client))
    return -1;
  return r_value;
}

/* This function puts and gets a 2D tensor of int32_t
values.  If the sent and received tensors do not match,
a non-zero value is returned.
*/
int put_get_2D_tensor_i32(size_t* dims, size_t n_dims,
                         char* key_suffix,
                         int key_suffix_length)
{
  void* client = NULL;
  const char* logger_name = "put_get_2D_tensor_i32";
  size_t cid_len = strlen(logger_name);
  if (SRNoError != SimpleCreateClient(logger_name, cid_len, &client))
    return -1;

  int32_t** tensor = (int32_t**)malloc(dims[0]*sizeof(int32_t*));
  int32_t** result = 0;

  for(int i=0; i<dims[0]; i++) {
    tensor[i] = (int32_t*)malloc(dims[1]*sizeof(int32_t));
  }

  for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++) {
      tensor[i][j] = rand()%INT32_MAX;
      if(rand()%2)
        tensor[i][j] *= -1;
    }
  }

  int r_value = 0;
  r_value = put_get_2D_tensor(client, (void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              SRTensorTypeInt32);

  for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++){
      if(tensor[i][j]!=result[i][j]) {
        printf("%s", "The tensors do not match!");
        r_value = -1;
      }
    }
  }

  for(int i=0; i<dims[0]; i++) {
    free(tensor[i]);
  }
  free(tensor);
  if (SRNoError != DeleteCClient(&client))
    return -1;
  return r_value;
}

/* This function puts and gets a 2D tensor of int64_t
values.  If the sent and received tensors do not match,
a non-zero value is returned.
*/
int put_get_2D_tensor_i64(size_t* dims, size_t n_dims,
                         char* key_suffix,
                         int key_suffix_length)
{
  void* client = NULL;
  const char* logger_name = "put_get_2D_tensor_i64";
  size_t cid_len = strlen(logger_name);
  if (SRNoError != SimpleCreateClient(logger_name, cid_len, &client))
    return -1;

  int64_t** tensor = (int64_t**)malloc(dims[0]*sizeof(int64_t*));
  int64_t** result = 0;

  for(int i=0; i<dims[0]; i++) {
    tensor[i] = (int64_t*)malloc(dims[1]*sizeof(int64_t));
  }

  for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++) {
      tensor[i][j] = rand()%INT64_MAX;
      if(rand()%2)
        tensor[i][j] *= -1;
    }
  }

  int r_value = 0;
  r_value = put_get_2D_tensor(client, (void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              SRTensorTypeInt64);

  for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++){
      if(tensor[i][j]!=result[i][j]) {
        printf("%s", "The tensors do not match!");
        r_value = -1;
      }
    }
  }

  for(int i=0; i<dims[0]; i++) {
    free(tensor[i]);
  }
  free(tensor);
  if (SRNoError != DeleteCClient(&client))
    return -1;
  return r_value;
}

/* This function puts and gets a 2D tensor of uint8_t
values.  If the sent and received tensors do not match,
a non-zero value is returned.
*/
int put_get_2D_tensor_ui8(size_t* dims, size_t n_dims,
                          char* key_suffix,
                          int key_suffix_length)
{
  void* client = NULL;
  const char* logger_name = "put_get_2D_tensor_ui8";
  size_t cid_len = strlen(logger_name);
  if (SRNoError != SimpleCreateClient(logger_name, cid_len, &client))
    return -1;

  uint8_t** tensor = (uint8_t**)malloc(dims[0]*sizeof(uint8_t*));
  uint8_t** result = 0;

  for(int i=0; i<dims[0]; i++) {
    tensor[i] = (uint8_t*)malloc(dims[1]*sizeof(uint8_t));
  }

  for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++) {
      tensor[i][j] = rand()%UINT8_MAX;
    }
  }

  int r_value = 0;
  r_value = put_get_2D_tensor(client, (void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              SRTensorTypeUint8);

  for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++){
      if(tensor[i][j]!=result[i][j]) {
        printf("%s", "The tensors do not match!");
        r_value = -1;
      }
    }
  }

  for(int i=0; i<dims[0]; i++) {
    free(tensor[i]);
  }
  free(tensor);
  if (SRNoError != DeleteCClient(&client))
    return -1;
  return r_value;
}

/* This function puts and gets a 2D tensor of uint16_t
values.  If the sent and received tensors do not match,
a non-zero value is returned.
*/
int put_get_2D_tensor_ui16(size_t* dims, size_t n_dims,
                          char* key_suffix,
                          int key_suffix_length)
{
  void* client = NULL;
  const char* logger_name = "put_get_2D_tensor_ui16";
  size_t cid_len = strlen(logger_name);
  if (SRNoError != SimpleCreateClient(logger_name, cid_len, &client))
    return -1;

  uint16_t** tensor = (uint16_t**)malloc(dims[0]*sizeof(uint16_t*));
  uint16_t** result = 0;

  for(int i=0; i<dims[0]; i++) {
    tensor[i] = (uint16_t*)malloc(dims[1]*sizeof(uint16_t));
  }

  for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++) {
      tensor[i][j] = rand()%UINT16_MAX;
    }
  }

  int r_value = 0;
  r_value = put_get_2D_tensor(client, (void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              SRTensorTypeUint16);

  for(int i=0; i<dims[0]; i++) {
    for(int j=0; j<dims[1]; j++){
      if(tensor[i][j]!=result[i][j]) {
        printf("%s", "The tensors do not match!");
        r_value = -1;
      }
    }
  }

  for(int i=0; i<dims[0]; i++) {
    free(tensor[i]);
  }
  free(tensor);
  if (SRNoError != DeleteCClient(&client))
    return -1;
  return r_value;
}

int main(int argc, char* argv[]) {
  size_t* dims = malloc(2*sizeof(size_t));
  dims[0] = 10;
  dims[1] = 26;
  size_t n_dims = 2;

  int result = 0;
  //2D double tensor
  char* dbl_suffix = "_dbl_c";
  result += put_get_2D_tensor_double(dims, n_dims,
                      dbl_suffix, strlen(dbl_suffix));

  //2D float tensor
  char* flt_suffix = "_flt_c";
  result += put_get_2D_tensor_float(dims, n_dims,
                      flt_suffix, strlen(flt_suffix));

  //2D int8 tensor
  char* i8_suffix = "_i8_c";
  result += put_get_2D_tensor_i8(dims, n_dims,
                      i8_suffix, strlen(i8_suffix));

  //2D int16 tensor
  char* i16_suffix = "_i16_c";
  result += put_get_2D_tensor_i16(dims, n_dims,
                      i16_suffix, strlen(i16_suffix));

  //2D int32 tensor
  char* i32_suffix = "_i32_c";
  result += put_get_2D_tensor_i32(dims, n_dims,
                      i32_suffix, strlen(i32_suffix));

  //2D int64 tensor
  char* i64_suffix = "_i64_c";
  result += put_get_2D_tensor_i64(dims, n_dims,
                      i64_suffix, strlen(i64_suffix));

  //2D uint8 tensor
  char* ui8_suffix = "_ui8_c";
  result += put_get_2D_tensor_ui8(dims, n_dims,
                      ui8_suffix, strlen(ui8_suffix));

  //2D uint16 tensor
  char* ui16_suffix = "_ui16_c";
  result += put_get_2D_tensor_ui16(dims, n_dims,
                      ui16_suffix, strlen(ui16_suffix));

  free(dims);
  printf("Test passed: %s\n", result == 0 ? "YES" : "NO");
  return result;
}
