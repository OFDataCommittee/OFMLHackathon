#include "c_dataset.h"

using namespace SmartRedis;

extern "C"
void* CDataSet(const char* name, const size_t name_length)
{
  /* Return a pointer to a new DataSet.
  The user is responsible for deleting the client.
  */
  std::string name_str = std::string(name, name_length);
  DataSet* dataset = new DataSet(name_str);
  return (void*)dataset;
}

extern "C"
void add_tensor(void* dataset,
                const char* name,
                const size_t name_length,
                void* data,
                const size_t* dims,
                const size_t n_dims,
                CTensorType type,
                CMemoryLayout mem_layout)
{
  /* This function adds a tensor to the dataset.
  */
  DataSet* d = (DataSet*)dataset;
  std::string name_str = std::string(name, name_length);

  std::vector<size_t> dims_vec;
  for(size_t i=0; i<n_dims; i++)
    dims_vec.push_back(dims[i]);

  d->add_tensor(name_str, data, dims_vec,
                convert_tensor_type(type),
                convert_layout(mem_layout));
  return;
}

extern "C"
void add_meta_scalar(void* dataset,
                     const char* name,
                     size_t name_length,
                     const void* data,
                     CMetaDataType type)
{
  /* Add a meta data value to the named meta data field
  */
  DataSet* d = (DataSet*)dataset;
  std::string name_str = std::string(name, name_length);
  d->add_meta_scalar(name_str, data,
                     convert_metadata_type(type));
  return;
}

extern "C"
void add_meta_string(void* dataset,
                     const char* name,
                     size_t name_length,
                     const char* data,
                     const size_t data_length)
{
  /* Add a meta data value to the named meta data field
  */
  DataSet* d = (DataSet*)dataset;
  std::string name_str = std::string(name, name_length);
  std::string data_str = std::string(data, data_length);
  d->add_meta_string(name_str, data_str);

  return;
}

extern "C"
void get_dataset_tensor(void* dataset,
                        const char* name,
                        const size_t name_length,
                        void** data,
                        size_t** dims,
                        size_t* n_dims,
                        CTensorType* type,
                        CMemoryLayout mem_layout)
{
  /* Get a tensor of a specified type from the database.
  This function may allocate new memory for the tensor.
  This memory will be deleted when the user deletes the
  DataSet object.
  */
  DataSet* d = (DataSet*)dataset;
  std::string name_str = std::string(name, name_length);

  TensorType t_type;
  d->get_tensor(name_str, *data, *dims, *n_dims,
                t_type, convert_layout(mem_layout));
  *type = convert_tensor_type(t_type);
  return;
}

extern "C"
void unpack_dataset_tensor(void* dataset,
                           const char* name,
                           const size_t name_length,
                           void* data,
                           const size_t* dims,
                           const size_t n_dims,
                           const CTensorType type,
                           const CMemoryLayout mem_layout)
{
  /* This function will take the tensor data buffer and copy
  it into the provided memory space (data).
  */
  DataSet* d = (DataSet*)dataset;
  std::string name_str = std::string(name, name_length);

  std::vector<size_t> dims_vec;
  for(size_t i=0; i<n_dims; i++)
    dims_vec.push_back(dims[i]);

  d->unpack_tensor(name_str, data, dims_vec,
                   convert_tensor_type(type),
                   convert_layout(mem_layout));

  return;
}

extern "C"
void* get_meta_scalars(void* dataset,
                      const char* name,
                      const size_t name_length,
                      size_t* length,
                      CMetaDataType* type)
{
  /* Get a meta data field. This method may allocate
  memory that is cleared when the user deletes the
  DataSet object
  */

  void* data;
  DataSet* d = (DataSet*)dataset;
  std::string key_str = std::string(name, name_length);

  MetaDataType m_type;
  d->get_meta_scalars(key_str, data, *length, m_type);
  *type = convert_metadata_type(m_type);
  return data;
}

extern "C"
void get_meta_strings(void* dataset,
                      const char* name,
                      const size_t name_length,
                      char*** data,
                      size_t* n_strings,
                      size_t** lengths)
{
  /* Get a meta data string field. This method may
  allocate memory that is cleared when the user
  deletes the DataSet object.
  */
  DataSet* d = (DataSet*)dataset;
  std::string key_str = std::string(name, name_length);
  d->get_meta_strings(key_str, *data, *n_strings, *lengths);
  return;
}
