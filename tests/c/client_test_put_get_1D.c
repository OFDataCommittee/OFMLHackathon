#include "c_client.h"
#include "c_client_test_utils.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "stdint.h"

bool cluster = true;

int put_get_1D_tensor(void* client,
                      void* tensor,
                      size_t* dims,
                      size_t n_dims,
                      void** result,
                      char* key_suffix,
                      size_t key_suffix_length,
                      CTensorType type)
{
  /* This function is a data type agnostic put and
  get for 1D tensors.  The result vector
  is filled with the fetched tensor.  Accuracy
  checking is done outside of this function because
  the type is not known.
  */

  char* prefix_str = "1D_tensor_test";

  size_t prefix_str_length = strlen(prefix_str);

  size_t key_length = prefix_str_length +
                      key_suffix_length;
  char* key = (char*)malloc((key_length+1)*sizeof(char));

  int pos;
  pos = 0;
  memcpy(&key[pos], prefix_str, prefix_str_length);
  pos += prefix_str_length;
  memcpy(&key[pos], key_suffix, key_suffix_length);
  pos += key_suffix_length;
  key[pos] = 0;

  CTensorType g_type;
  size_t* g_dims;
  size_t g_n_dims;

  CMemoryLayout layout = c_nested;
  put_tensor(client, key, key_length,
             (void*)tensor, dims, n_dims, type, layout);
  get_tensor(client, key, key_length,
             result, &g_dims, &g_n_dims,
             &g_type, layout);

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

int put_get_1D_tensor_double(size_t* dims, size_t n_dims,
                  char* key_suffix, size_t key_suffix_length)
{
  /* This function puts and gets a 1D tensor of double
  values.  If the sent and received tensors do not match,
  a non-zero value is returned.
  */

  void* client = SmartSimCClient(use_cluster());

  double* tensor = (double*)malloc(dims[0]*sizeof(double));
  double* result = 0;

  for(size_t i=0; i<dims[0]; i++)
    tensor[i] = ((double)rand())/RAND_MAX;

  int r_value = 0;
  r_value = put_get_1D_tensor(client,(void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              c_dbl);

  if(!is_equal_1D_tensor_dbl(tensor, result, dims[0])) {
      printf("%s", "The double tensors do not match!\n");
      r_value = -1;
  }

  free(tensor);
  DeleteCClient(client);
  return r_value;
}

int put_get_1D_tensor_float(size_t* dims, size_t n_dims,
                           char* key_suffix,
                           size_t key_suffix_length)
{
  /* This function puts and gets a 1D tensor of float
  values.  If the sent and received tensors do not match,
  a non-zero value is returned.
  */

  void* client = SmartSimCClient(use_cluster());

  float* tensor = (float*)malloc(dims[0]*sizeof(float));
  float* result;

  for(size_t i=0; i<dims[0]; i++)
    tensor[i] = ((float)rand())/RAND_MAX;

  int r_value = 0;
  r_value = put_get_1D_tensor(client,(void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              c_flt);

  free(tensor);
  DeleteCClient(client);
  return r_value;
}

int put_get_1D_tensor_i8(size_t* dims, size_t n_dims,
                         char* key_suffix,
                         size_t key_suffix_length)
{
  /* This function puts and gets a 1D tensor of int8_t
  values.  If the sent and received tensors do not match,
  a non-zero value is returned.
  */

  void* client = SmartSimCClient(use_cluster());

  int8_t* tensor = (int8_t*)malloc(dims[0]*sizeof(int8_t));
  int8_t* result = (int8_t*)malloc(dims[0]*sizeof(int8_t));

  for(size_t i=0; i<dims[0]; i++) {
    tensor[i] = rand()%INT8_MAX;
    if(rand()%2)
      tensor[i] *= -1;
  }

  int r_value = 0;
  r_value = put_get_1D_tensor(client,(void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              c_int8);

  if(!is_equal_1D_tensor_i8(tensor, result, dims[0])) {
      printf("%s", "The i8 tensors do not match!\n");
      r_value = -1;
  }

  free(tensor);
  DeleteCClient(client);
  return r_value;
}

int put_get_1D_tensor_i16(size_t* dims, size_t n_dims,
                         char* key_suffix,
                         size_t key_suffix_length)
{
  /* This function puts and gets a 1D tensor of int16_t
  values.  If the sent and received tensors do not match,
  a non-zero value is returned.
  */

  void* client = SmartSimCClient(use_cluster());

  int16_t* tensor = (int16_t*)malloc(dims[0]*sizeof(int16_t));
  int16_t* result;

  for(size_t i=0; i<dims[0]; i++) {
    tensor[i] = rand()%INT16_MAX;
    if(rand()%2)
      tensor[i] *= -1;
  }

  int r_value = 0;
  r_value = put_get_1D_tensor(client,(void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              c_int16);

  if(!is_equal_1D_tensor_i16(tensor, result, dims[0])) {
      printf("%s", "The i16 tensors do not match!\n");
      r_value = -1;
  }

  free(tensor);
  DeleteCClient(client);
  return r_value;
}

int put_get_1D_tensor_i32(size_t* dims, size_t n_dims,
                         char* key_suffix,
                         size_t key_suffix_length)
{
  /* This function puts and gets a 1D tensor of int32_t
  values.  If the sent and received tensors do not match,
  a non-zero value is returned.
  */

  void* client = SmartSimCClient(use_cluster());

  int32_t* tensor = (int32_t*)malloc(dims[0]*sizeof(int32_t));
  int32_t* result;

  for(size_t i=0; i<dims[0]; i++) {
    tensor[i] = rand()%INT32_MAX;
    if(rand()%2)
      tensor[i] *= -1;
  }

  int r_value = 0;
  r_value = put_get_1D_tensor(client,(void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              c_int32);

  if(!is_equal_1D_tensor_i32(tensor, result, dims[0])) {
      printf("%s", "The i32 tensors do not match!\n");
      r_value = -1;
  }

  free(tensor);
  DeleteCClient(client);
  return r_value;
}

int put_get_1D_tensor_i64(size_t* dims, size_t n_dims,
                         char* key_suffix,
                         size_t key_suffix_length)
{
  /* This function puts and gets a 1D tensor of int64_t
  values.  If the sent and received tensors do not match,
  a non-zero value is returned.
  */

  void* client = SmartSimCClient(use_cluster());

  int64_t* tensor = (int64_t*)malloc(dims[0]*sizeof(int64_t));
  int64_t* result;

  for(size_t i=0; i<dims[0]; i++) {
    tensor[i] = rand()%INT64_MAX;
    if(rand()%2)
      tensor[i] *= -1;
  }

  int r_value = 0;
  r_value = put_get_1D_tensor(client,(void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              c_int64);

  if(!is_equal_1D_tensor_i64(tensor, result, dims[0])) {
      printf("%s", "The i64 tensors do not match!\n");
      r_value = -1;
  }

  free(tensor);
  DeleteCClient(client);
  return r_value;
}

int put_get_1D_tensor_ui8(size_t* dims, size_t n_dims,
                          char* key_suffix,
                          size_t key_suffix_length)
{
  /* This function puts and gets a 1D tensor of uint8_t
  values.  If the sent and received tensors do not match,
  a non-zero value is returned.
  */

  void* client = SmartSimCClient(use_cluster());

  uint8_t* tensor = (uint8_t*)malloc(dims[0]*sizeof(uint8_t));
  uint8_t* result;

  for(size_t i=0; i<dims[0]; i++)
    tensor[i] = rand()%UINT8_MAX;

  int r_value = 0;
  r_value = put_get_1D_tensor(client,(void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              c_uint8);

  if(!is_equal_1D_tensor_ui8(tensor, result, dims[0])) {
      printf("%s", "The ui8 tensors do not match!\n");
      r_value = -1;
  }

  free(tensor);
  DeleteCClient(client);
  return r_value;
}

int put_get_1D_tensor_ui16(size_t* dims, size_t n_dims,
                          char* key_suffix,
                          size_t key_suffix_length)
{
  /* This function puts and gets a 1D tensor of uint16_t
  values.  If the sent and received tensors do not match,
  a non-zero value is returned.
  */

  void* client = SmartSimCClient(use_cluster());

  uint16_t* tensor = (uint16_t*)malloc(dims[0]*sizeof(uint16_t));
  uint16_t* result;

  for(int i=0; i<dims[0]; i++)
    tensor[i] = rand()%UINT16_MAX;

  int r_value = 0;
  r_value = put_get_1D_tensor(client,(void*)tensor,
                              dims, n_dims, (void**)(&result),
                              key_suffix, key_suffix_length,
                              c_uint16);

  if(!is_equal_1D_tensor_ui16(tensor, result, dims[0])) {
      printf("%s", "The ui16 tensors do not match!\n");
      r_value = -1;
  }

  free(tensor);
  DeleteCClient(client);
  return r_value;
}

int main(int argc, char* argv[]) {

  size_t* dims = malloc(sizeof(size_t));
  dims[0] = 10;
  size_t n_dims = 1;

  int result = 0;
  //1D double tensor
  char* dbl_suffix = "_dbl_c";
  result += put_get_1D_tensor_double(dims, n_dims,
                      dbl_suffix, strlen(dbl_suffix));

  //1D float tensor
  char* flt_suffix = "_flt_c";
  result += put_get_1D_tensor_float(dims, n_dims,
                      flt_suffix, strlen(flt_suffix));

  //1D int8 tensor
  char* i8_suffix = "_i8_c";
  result += put_get_1D_tensor_i8(dims, n_dims,
                      i8_suffix, strlen(i8_suffix));

  //1D int16 tensor
  char* i16_suffix = "_i16_c";
  result += put_get_1D_tensor_i16(dims, n_dims,
                      i16_suffix, strlen(i16_suffix));

  //1D int32 tensor
  char* i32_suffix = "_i32_c";
  result += put_get_1D_tensor_i32(dims, n_dims,
                      i32_suffix, strlen(i32_suffix));

  //1D int64 tensor
  char* i64_suffix = "_i64_c";
  result += put_get_1D_tensor_i64(dims, n_dims,
                      i64_suffix, strlen(i64_suffix));

  //1D uint8 tensor
  char* ui8_suffix = "_ui8_c";
  result += put_get_1D_tensor_ui8(dims, n_dims,
                      ui8_suffix, strlen(ui8_suffix));

  //1D uint16 tensor
  char* ui16_suffix = "_ui16_c";
  result += put_get_1D_tensor_ui16(dims, n_dims,
                      ui16_suffix, strlen(ui16_suffix));

  free(dims);
  printf("%s","Test passed: ");
  if(result==0)
    printf("%s", "YES\n");
  else
    printf("%s", "NO\n");

  return result;
}
