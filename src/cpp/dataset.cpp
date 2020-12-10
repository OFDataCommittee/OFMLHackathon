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
    this->_dim_queries = std::move(dataset._dim_queries);
}

DataSet& DataSet::operator=(DataSet&& dataset)
{
    /* Move assignment operator for DataSet
    */
    if(this!=&dataset) {
        this->name = std::move(dataset.name);
        this->_metadata = std::move(dataset._metadata);
        this->_tensorpack = std::move(dataset._tensorpack);
        this->_dim_queries = std::move(dataset._dim_queries);
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
                         void* data,
                         const std::vector<size_t>& dims,
                         const TensorType type,
                         MemoryLayout mem_layout)
{
    /* Creates a tensor with the provided information and adds
    the tensor name to the .tensors metadata
    */
    this->_add_to_tensorpack(name, data, dims,
                             type, mem_layout);
    this->_metadata.add_value(".tensors", name.c_str(),
                              MetaDataType::string);
    return;
}

void DataSet::add_meta(const std::string& name,
                       const void* data,
                       const MetaDataType type)
{
    this->_metadata.add_value(name, data, type);
    return;
}

void DataSet::get_tensor(const std::string& name,
                         void*& data,
                         std::vector<size_t>& dims,
                         TensorType& type,
                         MemoryLayout mem_layout)
{
    /* This function gets a tensor from the database,
    allocates memory in the specified format for the
    user, sets the dimensions of the dims vector
    for the user, and points the data pointer to
    the allocated memory space.
    */
    if(!(this->_tensorpack.tensor_exists(name)))
        throw std::runtime_error("The tensor " +
                                 std::string(name) +
                                 " does not exist in " +
                                 this->name + " dataset.");

    type = this->_tensorpack.get_tensor(name)->type();
    data = this->_tensorpack.get_tensor(name)->data_view(mem_layout);
    dims = this->_tensorpack.get_tensor(name)->dims();
    return;
}

void DataSet::get_tensor(const std::string&  name,
                         void*& data,
                         size_t*& dims,
                         size_t& n_dims,
                         TensorType& type,
                         MemoryLayout mem_layout)
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
    this->get_tensor(name, data, dims_vec,
                     type, mem_layout);

    size_t n_bytes = sizeof(int)*dims_vec.size();
    dims = this->_dim_queries.allocate_bytes(n_bytes);
    n_dims = dims_vec.size();

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

void DataSet::unpack_tensor(const std::string& name,
                            void* data,
                            const std::vector<size_t>& dims,
                            const TensorType type,
                            MemoryLayout mem_layout)
{
    /* This function will take the tensor data buffer and put it into
    the provided memory space (data).
    */

   if(!(this->_tensorpack.tensor_exists(name)))
        throw std::runtime_error("The tensor " + std::string(name)
                                               + " does not exist in "
                                               + this->name + " dataset.");

    this->_tensorpack.get_tensor(name)->fill_mem_space(data, dims, mem_layout);
    return;
}

void DataSet::get_meta(const std::string& name,
                       void*& data,
                       size_t& length,
                       MetaDataType& type)
{
    /* This function points the data pointer to a
    dynamically allocated array of the metadata
    and sets the length pointer value to the number
    of elements in the array.  The parameter type
    is set to the return type so that the user
    knows how to use the values if they are
    unsure of the type.
    */
    this->_metadata.get_values(name, data, length, type);
    return;
}

std::string DataSet::get_tensor_type(const std::string& name)
{
    /* Returns the tensor data type
    */
    return this->_tensorpack.get_tensor(name)->name();
}

inline void DataSet::_add_to_tensorpack(const std::string& name,
                                        void* data,
                                        const std::vector<size_t>& dims,
                                        const TensorType type,
                                        const MemoryLayout mem_layout)
{
    /* This function adds the tensor to the
    internal TensorPack object.
    */
    this->_tensorpack.add_tensor(name, data, dims,
                                 type, mem_layout);
    return;
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