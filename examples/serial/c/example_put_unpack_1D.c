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

  void* client = NULL;
  bool cluster_mode = true; // Set to false if not using a clustered database
  if (SRNoError != SmartRedisCClient(cluster_mode, &client)) {
    printf("Client initialization failed!\n");
    exit(-1);
  }

  char key[] = "1D_tensor_test";

  size_t key_length = strlen(key);

  if (SRNoError != put_tensor(client, key, key_length,
                              (void*)tensor, dims, n_dims,
                              SRTensorTypeFloat, SRMemLayoutNested)) {
    printf("Call to put_tensor failed!\n");
    exit(-1);
  }

  float* result = (float*)malloc(dims[0]*sizeof(float));
  if (SRNoError != unpack_tensor(client, key, key_length,
                                 result, dims, n_dims,
                                 SRTensorTypeFloat, SRMemLayoutNested)) {
    printf("Call to unpack_tensor failed!\n");
    exit(-1);
  }

  DeleteCClient(&client);
  free(tensor);
  free(result);
  free(dims);

  return 0;
}
