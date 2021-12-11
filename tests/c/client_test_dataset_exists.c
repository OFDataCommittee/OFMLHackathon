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

#include "c_client.h"
#include "c_dataset.h"
#include "c_client_test_utils.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "stdint.h"
#include "SRException.h"

bool cluster = true;


int missing_dataset(char *dataset_name, size_t dataset_name_len)
{
  void *client = NULL;
  if (SRNoError != SmartRedisCClient(use_cluster(), &client))
    return -1;

  bool exists = false;
  if (SRNoError != dataset_exists(client, dataset_name, dataset_name_len, &exists))
    return -1;
  return exists ? 1 : 0;
}

int present_dataset(char *dataset_name, size_t dataset_name_len)
{
  void *client = NULL;
  void *dataset = NULL;
  char *t1 = "tensor_1";
  char *t2 = "tensor_2";
  char *t3 = "tensor_3";
  const size_t n_dims = 3;
  size_t dims[n_dims];
  uint16_t ***tensor = NULL;
  int i, j, k;
  bool exists = false;

  // Initialize client and dataset
  if (SRNoError != SmartRedisCClient(use_cluster(), &client) || NULL == client)
    return -1;
  if (SRNoError != CDataSet(dataset_name, dataset_name_len, &dataset) || NULL == dataset)
    return -1;

  // Allocate memory for tensors
  dims[0] = 5;
  dims[1] = 4;
  dims[2] = 17;
  tensor = (uint16_t ***)malloc(dims[0] * sizeof(uint16_t **));

  for (i = 0; i < dims[0]; i++) {
    tensor[i] = (uint16_t **)malloc(dims[1] * sizeof(uint16_t *));
    for (j = 0; j < dims[1]; j++) {
      tensor[i][j] = (uint16_t *)malloc(dims[2] * sizeof(uint16_t));
    }
  }

  // Add tensors to the DataSet
  for (i = 0; i < dims[0]; i++) {
    for (j = 0; j < dims[1]; j++) {
      for (k = 0; k < dims[2]; k++) {
        tensor[i][j][k] = rand()%UINT16_MAX;
      }
    }
  }
  if (SRNoError != add_tensor(dataset, t1, strlen(t1), tensor, dims, n_dims, SRTensorTypeInt16, SRMemLayoutNested))
    return -1;

  for (i = 0; i < dims[0]; i++) {
    for (j = 0; j < dims[1]; j++) {
      for (k = 0; k < dims[2]; k++) {
        tensor[i][j][k] = rand()%UINT16_MAX;
      }
    }
  }
  if (SRNoError != add_tensor(dataset, t2, strlen(t2), tensor, dims, n_dims, SRTensorTypeInt16, SRMemLayoutNested))
    return -1;

  for (i = 0; i < dims[0]; i++) {
    for (j = 0; j < dims[1]; j++) {
      for (k = 0; k < dims[2]; k++) {
        tensor[i][j][k] = rand()%UINT16_MAX;
      }
    }
  }
  if (SRNoError != add_tensor(dataset, t3, strlen(t3), tensor, dims, n_dims, SRTensorTypeInt16, SRMemLayoutNested))
    return -1;

  // Put the DataSet into the database
  if (SRNoError != put_dataset(client, dataset))
    return -1;

  // Make sure it exists
  if (SRNoError != dataset_exists(client, dataset_name, dataset_name_len, &exists))
    return -1;
  return exists ? 0 : -1;
}

int main(int argc, char* argv[])
{
  int result = 0;
  // test with absence of dataset
  char* dbl_suffix = "_dbl_c";
  char *dataset_name = "missing_dataset";

  // test with absent dataset
  result += missing_dataset(dataset_name, strlen(dataset_name));

  // test with presence of dataset
  dataset_name = "present_dataset";
  result += present_dataset(dataset_name, strlen(dataset_name));

  printf("Test passed: %s\n", result == 0 ? "YES" : "NO");
  return result;
}
