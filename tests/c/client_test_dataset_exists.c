
#include "c_client.h"
#include "c_dataset.h"
#include "c_client_test_utils.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "stdint.h"

bool cluster = true;


int missing_dataset(char *dataset_name, size_t dataset_name_len)
{
  void *client = SmartRedisCClient(use_cluster());
  
  return dataset_exists(client, dataset_name, dataset_name_len) ? 1 : 0;
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
  
  // Initialize client and dataset
  client = SmartRedisCClient(use_cluster());
  if (NULL == client)
    return 1;
  dataset = CDataSet(dataset_name, dataset_name_len);
  if (NULL == dataset)
    return 1;

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
  add_tensor(dataset, t1, strlen(t1), tensor, dims, n_dims, c_int16, c_nested);

  for (i = 0; i < dims[0]; i++) {
    for (j = 0; j < dims[1]; j++) {
      for (k = 0; k < dims[2]; k++) {
        tensor[i][j][k] = rand()%UINT16_MAX;
      }
    }
  }
  add_tensor(dataset, t2, strlen(t2), tensor, dims, n_dims, c_int16, c_nested);

  for (i = 0; i < dims[0]; i++) {
    for (j = 0; j < dims[1]; j++) {
      for (k = 0; k < dims[2]; k++) {
        tensor[i][j][k] = rand()%UINT16_MAX;
      }
    }
  }
  add_tensor(dataset, t3, strlen(t3), tensor, dims, n_dims, c_int16, c_nested);
  
  // Put the DataSet into the database
  put_dataset(client, dataset);
  
  // Make sure it exists  
  return dataset_exists(client, dataset_name, dataset_name_len) ? 0 : 1;
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

  printf("%s","Test passed: ");
  if(result==0)
    printf("%s", "YES\n");
  else
    printf("%s", "NO\n");

  return result;
}
