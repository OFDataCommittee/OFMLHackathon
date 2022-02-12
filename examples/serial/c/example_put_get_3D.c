#include "c_client.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "stdint.h"
#include "sr_enums.h"

int main(int argc, char* argv[]) {
  /* This function puts and gets a 3D tensor of double
  values.
  */

  size_t n_dims = 3;
  size_t* dims = malloc(n_dims*sizeof(size_t));
  dims[0] = 10;
  dims[1] = 26;
  dims[2] = 3;

  void* client = NULL;
  bool cluster_mode = true; // Set to false if not using a clustered database
  if (SRNoError != SmartRedisCClient(cluster_mode, &client)) {
    printf("Client initialization failed!\n");
    exit(-1);
  }

  // Allocate tensors
  double*** tensor = (double***)malloc(dims[0]*sizeof(double**));

  for(size_t i=0; i<dims[0]; i++) {
    tensor[i] = (double**)malloc(dims[1]*sizeof(double*));
    for(size_t j=0; j<dims[1]; j++) {
      tensor[i][j] = (double*)malloc(dims[2]*sizeof(double));
    }
  }

  for(size_t i=0; i<dims[0]; i++)
    for(size_t j=0; j<dims[1]; j++)
      for(size_t k=0; k<dims[2]; k++)
        tensor[i][j][k] = ((double)rand())/RAND_MAX;

  char key[] = "3D_tensor_example";
  size_t key_length = strlen(key);

  if (SRNoError != put_tensor(client, key, key_length,
                              (void*)tensor, dims, n_dims,
                              SRTensorTypeDouble, SRMemLayoutNested)) {
    printf("Call to put_tensor failed!\n");
    exit(-1);
  }

  // Allocate return values
  SRTensorType g_type;
  size_t* g_dims;
  size_t g_n_dims;
  double*** result = 0;

  if (SRNoError != get_tensor(client, key, key_length,
                              (void**) &result, &g_dims, &g_n_dims,
                              &g_type, SRMemLayoutNested)) {
    printf("Call to get_tensor failed!\n");
    exit(-1);
  }

  // Clean up
  DeleteCClient(&client);
  for(size_t i=0; i<dims[0]; i++) {
    for(size_t j=0; j<dims[1]; j++) {
      free(tensor[i][j]);
    }
    free(tensor[i]);
  }
  free(tensor);
  return 0;
}
