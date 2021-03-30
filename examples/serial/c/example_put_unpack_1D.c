#include "c_client.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "stdint.h"

int main(int argc, char* argv[]) {

  size_t* dims = malloc(sizeof(size_t));
  dims[0] = 10;
  size_t n_dims = 1;

  //1D float tensor
  float* tensor = (float*)malloc(dims[0]*sizeof(float));

  for(size_t i=0; i<dims[0]; i++)
    tensor[i] = ((float)rand())/RAND_MAX;

  void* client = SmartRedisCClient(false);

  char key[] = "1D_tensor_test";

  size_t key_length = strlen(key);

  put_tensor(client, key, key_length,
             (void*)tensor, dims, n_dims, c_flt, c_nested);

  
  float* result = (float*)malloc(dims[0]*sizeof(float));
  unpack_tensor(client, key, key_length,
                result, dims, n_dims,
                c_flt, c_nested);

  DeleteCClient(client);
  free(tensor);
  free(result);
  free(dims);

  return 0;
}
