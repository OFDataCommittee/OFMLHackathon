#include "c_client.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

int put_get_1D_array_double(int* dims, int n_dims,
                  char* key_suffix, int key_suffix_length)
{
  void* client = SmartSimCClient(true);

  //Allocate and fill arrays
  double* array = (double*)malloc(dims[0]*sizeof(double));
  double* result = (double*)malloc(dims[0]*sizeof(double));

  for(int i=0; i<dims[0]; i++)
    array[i] = ((double)rand())/RAND_MAX;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank > 9)
    printf("%s", "C test does not support MPI ranks "\
                 "greater than 9.");

  char* prefix_str = "1D_tensor_test_rank_";
  char* rank_str = malloc(2*sizeof(char));
  rank_str[0] = rank + '0';
  rank_str[1] = 0;

  size_t prefix_str_length = strlen(prefix_str);
  size_t rank_str_length = strlen(rank_str);

  int key_length = prefix_str_length + rank_str_length +
                   key_suffix_length;
  char* key = (char*)malloc((key_length+1)*sizeof(char));

  int pos;
  pos = 0;
  memcpy(&key[pos], prefix_str, prefix_str_length);
  pos += prefix_str_length + 1;
  memcpy(&key[pos], rank_str, rank_str_length);
  pos += rank_str_length;
  memcpy(&key[pos], key_suffix, key_suffix_length);
  pos += key_suffix_length;
  key[pos] = 0;

  char* type = "DOUBLE";
  size_t type_length = strlen(type);

  put_tensor(client, key, key_length, type, type_length,
             (void*)array, dims, n_dims);
  get_tensor(client, key, key_length, type, type_length,
             result, dims, n_dims);

  for(int i=0; i<dims[0]; i++) {
    if(array[i]!=result[i]) {
      printf("%s", "The arrays do not match!");
      free(array);
      free(result);
      return -1;
    }
  }

  free(rank_str);
  free(key);
  free(array);
  free(result);
  return 0;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int* dims = malloc(sizeof(int));
  dims[0] = 10;
  int n_dims = 1;

  char* suffix = "_dbl_c";
  size_t suffix_length = strlen(suffix);

  int result =  put_get_1D_array_double(dims, n_dims,
                                        suffix, suffix_length);
  free(dims);
  MPI_Finalize();
  printf("%s","Test passed: ");
  if(result==0)
    printf("%s", "YES\n");
  else
    printf("%s", "NO\n");

  return result;
}
