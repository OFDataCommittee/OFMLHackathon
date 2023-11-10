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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "stdint.h"
#include "sr_enums.h"

int main(int argc, char* argv[]) {
  /* This function puts and gets a 3D tensor of double
  values.
  */

  const char* logger_name = "put_get_3d";
  size_t cid_len = strlen(logger_name);
  size_t n_dims = 3;
  size_t* dims = malloc(n_dims*sizeof(size_t));
  dims[0] = 10;
  dims[1] = 26;
  dims[2] = 3;

  void* client = NULL;
  if (SRNoError != SimpleCreateClient(logger_name, cid_len, &client)) {
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
        tensor[i][j][k] = ((double)rand())/(double)RAND_MAX;

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
