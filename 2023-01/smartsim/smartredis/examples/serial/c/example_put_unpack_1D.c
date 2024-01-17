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

int main(int argc, char* argv[]) {

  const char* logger_name = "put_unpack_1d";
  size_t cid_len = strlen(logger_name);
  size_t* dims = malloc(sizeof(size_t));
  dims[0] = 10;
  size_t n_dims = 1;

  //1D float tensor
  float* tensor = (float*)malloc(dims[0]*sizeof(float));

  for(size_t i=0; i<dims[0]; i++)
    tensor[i] = ((float)rand())/(float)RAND_MAX;

  void* client = NULL;
  if (SRNoError != SimpleCreateClient(logger_name, cid_len, &client)) {
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
