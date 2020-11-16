#include "dataset.h"

DataSet::DataSet(const std::string& name)
{
    this->name = name;
}

DataSet::DataSet(DataSet&& dataset)
{
    /* This is the move constructor for DataSet
    */
    this->name = std::move(dataset.name);
    this->_metadata = std::move(dataset._metadata);
    this->_tensorpack = std::move(dataset._tensorpack);
}

DataSet& DataSet::operator=(DataSet&& dataset)
{
    /* Move assignment operator for DataSet
    */
   if(this!=&dataset) {
        this->name = std::move(dataset.name);
        this->_metadata = std::move(dataset._metadata);
        this->_tensorpack = std::move(dataset._tensorpack);
   }
   return *this;
}

DataSet::DataSet(const std::string& name, char* buf,
                 size_t buf_size)
{
    this->name = name;
    this->_metadata.fill_from_buffer(buf, buf_size);
}

void DataSet::add_tensor(const std::string& name,
                         const std::string& type,
                         void* data, std::vector<size_t> dims)
{
    /* Creates a tensor with the provided information and adds
    the tensor name to the .tensors metadata
    */
    this->_tensorpack.add_tensor(name, type, data, dims);
    this->_metadata.add_value(".tensors", "STRING", name.c_str());
    return;
}

void DataSet::add_tensor_buf_only(const std::string& name,
                                  const std::string& type,
                                  std::vector<size_t> dims,
                                  std::string_view buf)
{
    /* This adds tensors to the dataset, but does not add them
    to the .tensors metadata message because it is assumed that
    those tensors are being read from a buffer inventory and
    .tensors already exists
    */
    this->_tensorpack.add_tensor(name, type, dims, buf);
    return;
}

void DataSet::get_tensor(const std::string&  name,
                         const std::string&  type,
                         void*& data, std::vector<size_t>& dims)
{
    /* This function will retrieve tensor data pointer to the
    user.  If the pointer does not exist (e.g. it is a tensor
    with buffer data only), memory will be allocated and
    the buffer will be copied into the new memory.  This memory
    will be freed when the dataset is destroyed.  If the
    data pointer in the tensor already points to a memory
    space, that c_ptr will be returned.
    */
    if(!(this->_tensorpack.tensor_exists(name)))
        throw std::runtime_error("The tensor " + std::string(name)
                                               + " does not exist in "
                                               + this->name + " dataset.");

    //TODO should we do checks on the type here to
    //make sure they match up with tensor?


    data = this->_tensorpack.get_tensor_data(name);
    dims = this->_tensorpack.get_tensor(name)->get_tensor_dims();
    return;
}

void DataSet::get_tensor(const std::string&  name,
                         const std::string&  type,
                         void*& data, size_t*& dims,
                         size_t& n_dims)
{
    /* This function will retrieve tensor data
    pointer to the user.  If the pointer does not
    exist (e.g. it is a tensor with buffer data only),
    memory will be allocated and the buffer will be
    copied into the new memory.  This memory will be
    freed when the dataset is destroyed.  If the data
    pointer in the tensor already points to a memory
    space, that c_ptr will be returned.
    */
    std::vector<size_t> dims_vec;
    this->get_tensor(name, type, data, dims_vec);

    size_t n_bytes = sizeof(int)*dims_vec.size();
    dims = _dim_queries.allocate_bytes(n_bytes);

    std::vector<size_t>::const_iterator it = dims_vec.cbegin();
    std::vector<size_t>::const_iterator it_end = dims_vec.cend();
    size_t i = 0;
    while(it!=it_end) {
        dims[i] = *it;
        i++;
        it++;
    }
    return;
}

void DataSet::unpack_tensor(const std::string&  name,
                            const std::string&  type,
                            void* data, std::vector<size_t> dims)
{
    /* This function will take the tensor data buffer and put it into
    the provided memory space (data).
    */

   if(!(this->_tensorpack.tensor_exists(name)))
        throw std::runtime_error("The tensor " + std::string(name)
                                               + " does not exist in "
                                               + this->name + " dataset.");

    this->_tensorpack.get_tensor(name)->fill_data_from_buf(data, dims, type);
    return;
}

void DataSet::add_meta(const std::string& name,
                       const std::string& type,
                       const void* data)
{
    this->_metadata.add_value(name, type, data);
    return;
}

void DataSet::get_meta(const std::string& name,
                       const std::string& type,
                       void*& data, size_t& length)
{
    /* This function points the data pointer to a dynamically allocated
    array of the metadata and sets the length pointer value to the number
    of elements in the array.
    */
   //TODO this assumes the user knows what type the metadata is before
   //they request it.  We can throw an error if it doesn't match what
   //is in the database, or just return nothing with a warning.  We need
   //to figure that out.  I'm not sure how we can design something that
   //Maybe what we do is there is a separate function that returns
   //the metadata type, and the user can use that to typecast the
   //return value.  That seems like it could work for containerized
   //return types like a numpy.ndaray, but could get tricky
   //for c++, c, and fortran pointer types.  Maybe we need
   //should return a containerized metadata object for the data
   //that would allow us to define all the operators to iterator through
   //that would remove this from the user.
   this->_metadata.get_values(name, type, data, length);
   return;
}

std::string DataSet::get_tensor_type(const std::string& name)
{
    /* Returns the tensor data type
    */
    return this->_tensorpack.get_tensor(name)->get_tensor_name();
}

DataSet::tensor_iterator DataSet::tensor_begin()
{
    /* Returns a iterator pointing to the first
    tensor
    */
    return this->_tensorpack.tensor_begin();
}

DataSet::const_tensor_iterator DataSet::tensor_cbegin()
{
    /* Returns a const_iterator pointing to the first
    tensor
    */
    return this->_tensorpack.tensor_cbegin();
}

DataSet::tensor_iterator DataSet::tensor_end()
{
    /* Returns a iterator pointing to the past-the-end
    tensor
    */
    return this->_tensorpack.tensor_end();
}

DataSet::const_tensor_iterator DataSet::tensor_cend()
{
    /* Returns a const_iterator pointing to the past-the-end
    tensor
    */
    return this->_tensorpack.tensor_cend();
}

std::string_view DataSet::get_metadata_buf()
{
    /* Returns a std::string_view of the metadata serialized
    */
   return this->_metadata.get_metadata_buf();
}