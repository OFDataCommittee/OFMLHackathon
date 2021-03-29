#include "client.h"
#include <vector>
#include <string>

void free_2D_array_double(double** array, int dim_1)
{
  /*  This function frees memory of dynamically
      allocated 2D array.
  */
  for(int i=0; i<dim_1; i++)
       free(array[i]);
  free(array);
}

void free_3D_array_double(double*** array, int dim_1, int dim_2)
{
  /* This function frees memory of dynamically
     allocated 3D array.
  */
  for(int i=0; i<dim_1; i++)
    free_2D_array_double(array[i], dim_2);
  free(array);
}

double*** allocate_3D_array_double(int dim_1, int dim_2, int dim_3)
{
  /* This function allocates a 3D array and returns
     a pointer to that 3D array.
  */
  double*** array = (double***)malloc(dim_1*sizeof(double**));
  for (int i=0; i<dim_1; i++) {
    array[i] = (double**)malloc(dim_2*sizeof(double*));
    for(int j=0; j<dim_2; j++){
      array[i][j] = (double*)malloc(dim_3 * sizeof(double));
    }
  }
  return array;
}

void set_1D_array_double_values(double* a, int dim_1)
{
  /* This function fills a 1D array with random
     floating point values.
  */
  std::default_random_engine generator(rand());
  std::uniform_real_distribution<double> distribution;
  for(int i=0; i<dim_1; i++)
    //a[i] = distribution(generator);
    a[i] = 2.0*rand()/RAND_MAX - 1.0;
}

void put_get_3D_array_double(std::vector<size_t> dims)
{

  //Allocate and fill arrays
  double* array =
    (double*)malloc(dims[0]*dims[1]*dims[2]*sizeof(double));

  double* u_contig_result =
    (double*)malloc(dims[0]*dims[1]*dims[2]*sizeof(double));

  double*** u_nested_result =
    allocate_3D_array_double(dims[0], dims[1], dims[2]);

  set_1D_array_double_values(array, dims[0]*dims[1]*dims[2]);


  SILC::Client client(false);

  std::string key = "3d_tensor_contig";
  client.put_tensor(key, (void*)array, dims, SILC::TensorType::dbl,
                    SILC::MemoryLayout::contiguous);

  client.unpack_tensor(key, u_contig_result,
                       {dims[0]*dims[1]*dims[2]}, SILC::TensorType::dbl,
                       SILC::MemoryLayout::contiguous);

  client.unpack_tensor(key, u_nested_result, dims, SILC::TensorType::dbl,
                       SILC::MemoryLayout::nested);


  SILC::TensorType g_type_nested;
  std::vector<size_t> g_dims_nested;
  void* g_nested_result;
  client.get_tensor(key, g_nested_result,
                    g_dims_nested, g_type_nested,
                    SILC::MemoryLayout::nested);
  double*** g_type_nested_result = (double***)g_nested_result;

  SILC::TensorType g_type_contig;
  std::vector<size_t> g_dims_contig;
  void* g_contig_result;
  client.get_tensor(key, g_contig_result,
                    g_dims_contig, g_type_contig,
                    SILC::MemoryLayout::contiguous);

  // Clean up
  free(array);
  free(u_contig_result);
  free_3D_array_double(u_nested_result, dims[0], dims[1]);

  return;
}


int main(int argc, char* argv[]) {

  size_t dim1 = 3;
  size_t dim2 = 2;
  size_t dim3 = 5;

  std::vector<size_t> dims = {dim1, dim2, dim3};

  put_get_3D_array_double(dims);

  return 0;
}
