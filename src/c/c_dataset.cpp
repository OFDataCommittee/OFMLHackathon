#include "c_dataset.h"

extern "C"
void* CDataSet(const char* name, const size_t name_length)
{
  /* Return a pointer to a new SmartSimClient.
  The user is responsible for deleting the client.
  */
  std::string name_str = std::string(name, name_length);
  DataSet* dataset = new DataSet(name_str);
  return (void*)dataset;
}

extern "C"
void add_tensor(void* dataset,
                const char* name, const size_t name_length,
                const char* type, const size_t type_length,
                void* data, const int* dims, const int n_dims)
{
  /* This function adds a tensor to the dataset.
  */
  DataSet* d = (DataSet*)dataset;
  std::string name_str = std::string(name, name_length);
  std::string type_str = std::string(type, type_length);

  std::vector<int> dims_vec;
  for(int i=0; i<n_dims; i++)
    dims_vec.push_back(dims[i]);

  d->add_tensor(name_str, type_str, data, dims_vec);
  return;
}

extern "C"
void add_meta(void* dataset,
              const char* name, size_t name_length,
              const char* type, size_t type_length,
              const void* data
              )
{
  /* Add a meta data value to the named meta data field
  */
  DataSet* d = (DataSet*)dataset;
  std::string name_str = std::string(name, name_length);
  std::string type_str = std::string(type, type_length);
  d->add_meta(name_str, type_str, data);
  return;
}

extern "C"
void get_dataset_tensor(void* dataset,
                const char* name, const size_t name_length,
                const char* type, const size_t type_length,
                void*& data, int*& dims, int& n_dims)
{
  /* Get a tensor of a specified type from the database.
  This function may allocate new memory for the tensor.
  This memory will be deleted when the user deletes the
  DataSet object.
  */
  DataSet* d = (DataSet*)dataset;
  std::string name_str = std::string(name, name_length);
  std::string type_str = std::string(type, type_length);

  d->get_tensor(name_str, type_str, data, dims, n_dims);
  return;
}

extern "C"
void unpack_dataset_tensor(void* dataset,
                const char* name, const size_t name_length,
                const char* type, const size_t type_length,
                void*& data, const int* dims, const int n_dims)
{
  /* This function will take the tensor data buffer and copy
  it into the provided memory space (data).
  */
  DataSet* d = (DataSet*)dataset;
  std::string name_str = std::string(name, name_length);
  std::string type_str = std::string(type, type_length);

  std::vector<int> dims_vec;
  for(int i=0; i<n_dims; i++)
    dims_vec.push_back(dims[i]);

  d->unpack_tensor(name_str, type_str, data, dims_vec);
  return;
}

extern "C"
void get_meta(void* dataset,
              const char* name, const size_t name_length,
              const char* type, const size_t type_length,
              void*& data, int* length)
{
  /* Get a meta data field.. This method may allocated
  memory that is cleared when the user deletes the
  DataSet object
  */
  DataSet* d = (DataSet*)dataset;
  std::string key_str = std::string(name, name_length);
  std::string type_str = std::string(type, type_length);
  d->get_meta(key_str, type_str, data, *length);
  return;
}
