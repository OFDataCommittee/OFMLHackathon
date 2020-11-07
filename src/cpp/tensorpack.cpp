#include "tensorpack.h"

TensorPack::TensorPack()
{}

TensorPack::TensorPack(const TensorPack& tensorpack)
{
    /* This is the copy constructor for the TensorPack
    class.  A copy of each tensor is made using
    the tensor copy constructor.
    */

    TensorPack::_copy_tensor_list<double>(tensorpack._tensors_double,
                                          this->_tensors_double);
    TensorPack::_copy_tensor_list<float>(tensorpack._tensors_float,
                                         this->_tensors_float);
    TensorPack::_copy_tensor_list<int64_t>(tensorpack._tensors_int64,
                                           this->_tensors_int64);
    TensorPack::_copy_tensor_list<int32_t>(tensorpack._tensors_int32,
                                            this->_tensors_int32);
    TensorPack::_copy_tensor_list<int16_t>(tensorpack._tensors_int16,
                                           this->_tensors_int16);
    TensorPack::_copy_tensor_list<int8_t>(tensorpack._tensors_int8,
                                           this->_tensors_int8);
    TensorPack::_copy_tensor_list<uint16_t>(tensorpack._tensors_uint16,
                                            this->_tensors_uint16);
    TensorPack::_copy_tensor_list<uint8_t>(tensorpack._tensors_uint8,
                                            this->_tensors_uint8);
    this->_refresh_tensorbase_inventory();
}

TensorPack::TensorPack(TensorPack&& tensorpack)
{
    /* This is a move constructor for the TensorPack.
    */

    // Move all inventory data structures
    this->_tensors_double = std::move(tensorpack._tensors_double);
    this->_tensors_float = std::move(tensorpack._tensors_float);
    this->_tensors_int64 = std::move(tensorpack._tensors_int64);
    this->_tensors_int32 = std::move(tensorpack._tensors_int32);
    this->_tensors_int16 = std::move(tensorpack._tensors_int16);
    this->_tensors_int8 = std::move(tensorpack._tensors_int8);
    this->_tensors_uint16 = std::move(tensorpack._tensors_uint16);
    this->_tensors_uint8 = std::move(tensorpack._tensors_uint8);
    this->_all_tensors = std::move(tensorpack._all_tensors);
    this->_tensorbase_inventory = std::move(tensorpack._tensorbase_inventory);

    // Clear inventory data structors to make sure memory is not
    // freed
    tensorpack._tensors_double.clear();
    tensorpack._tensors_float.clear();
    tensorpack._tensors_int64.clear();
    tensorpack._tensors_int32.clear();
    tensorpack._tensors_int16.clear();
    tensorpack._tensors_int8.clear();
    tensorpack._tensors_uint16.clear();
    tensorpack._tensors_uint8.clear();
    tensorpack._all_tensors.clear();
    tensorpack._tensorbase_inventory.clear();
    return;
}

TensorPack& TensorPack::operator=(const TensorPack& tensorpack)
{
    /* This is the copy assignment operator that will
    copy the tensorpack into this.  Before copying,
    all current tensors in the tensorpack are deleted.
    */
    if (this != &tensorpack) {
        this->_delete_all_tensors();
        TensorPack::_copy_tensor_list<double>(tensorpack._tensors_double,
                                            this->_tensors_double);
        TensorPack::_copy_tensor_list<float>(tensorpack._tensors_float,
                                            this->_tensors_float);
        TensorPack::_copy_tensor_list<int64_t>(tensorpack._tensors_int64,
                                            this->_tensors_int64);
        TensorPack::_copy_tensor_list<int32_t>(tensorpack._tensors_int32,
                                            this->_tensors_int32);
        TensorPack::_copy_tensor_list<int16_t>(tensorpack._tensors_int16,
                                            this->_tensors_int16);
        TensorPack::_copy_tensor_list<int8_t>(tensorpack._tensors_int8,
                                            this->_tensors_int8);
        TensorPack::_copy_tensor_list<uint16_t>(tensorpack._tensors_uint16,
                                            this->_tensors_uint16);
        TensorPack::_copy_tensor_list<uint8_t>(tensorpack._tensors_uint8,
                                            this->_tensors_uint8);
        this->_refresh_tensorbase_inventory();
    }
    return *this;
}

TensorPack& TensorPack::operator=(TensorPack&& tensorpack)
{
    if (this != &tensorpack) {
        // Move all inventory data structures
        this->_tensors_double = std::move(tensorpack._tensors_double);
        this->_tensors_float = std::move(tensorpack._tensors_float);
        this->_tensors_int64 = std::move(tensorpack._tensors_int64);
        this->_tensors_int32 = std::move(tensorpack._tensors_int32);
        this->_tensors_int16 = std::move(tensorpack._tensors_int16);
        this->_tensors_int8 = std::move(tensorpack._tensors_int8);
        this->_tensors_uint16 = std::move(tensorpack._tensors_uint16);
        this->_tensors_uint8 = std::move(tensorpack._tensors_uint8);
        this->_all_tensors = std::move(tensorpack._all_tensors);
        this->_tensorbase_inventory = std::move(tensorpack._tensorbase_inventory);

        // Clear inventory data structors to make sure memory is not
        // freed
        tensorpack._tensors_double.clear();
        tensorpack._tensors_float.clear();
        tensorpack._tensors_int64.clear();
        tensorpack._tensors_int32.clear();
        tensorpack._tensors_int16.clear();
        tensorpack._tensors_int8.clear();
        tensorpack._tensors_uint16.clear();
        tensorpack._tensors_uint8.clear();
        tensorpack._all_tensors.clear();
        tensorpack._tensorbase_inventory.clear();
    }
    return *this;
}


TensorPack::~TensorPack()
{
    this->_delete_all_tensors();
}

void TensorPack::add_tensor(const std::string& name,
                            const std::string& type,
                            void* data,
                            const std::vector<int>& dims)
{
    /* This function adds a tensor with associated c_ptr for data.
    */
    if(this->_tensorbase_inventory.count(std::string(name))>0)
        throw std::runtime_error("The tensor " + std::string(name)
                                               + " already exists");

    if(DATATYPE_TENSOR_STR_DOUBLE.compare(type)==0) {
        Tensor<double>* ptr = new Tensor<double>(name, type, data, dims);
        this->_tensors_double.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_FLOAT.compare(type)==0) {
        Tensor<float>* ptr = new Tensor<float>(name, type, data, dims);
        this->_tensors_float.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_INT64.compare(type)==0) {
        Tensor<int64_t>* ptr = new Tensor<int64_t>(name, type, data, dims);
        this->_tensors_int64.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_INT32.compare(type)==0) {
        Tensor<int32_t>* ptr = new Tensor<int32_t>(name, type, data, dims);
        this->_tensors_int32.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_INT16.compare(type)==0) {
        Tensor<int16_t>* ptr = new Tensor<int16_t>(name, type, data, dims);
        this->_tensors_int16.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_INT8.compare(type)==0) {
        Tensor<int8_t>* ptr = new Tensor<int8_t>(name, type, data, dims);
        this->_tensors_int8.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_UINT16.compare(type)==0) {
        Tensor<uint16_t>* ptr = new Tensor<uint16_t>(name, type, data, dims);
        this->_tensors_uint16.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_UINT8.compare(type)==0) {
        Tensor<uint8_t>* ptr = new Tensor<uint8_t>(name, type, data, dims);
        this->_tensors_uint8.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else {
        throw std::runtime_error("Invalid tensor type in TensorPack: "+
                                 std::string(type));
    }
    return;
}

void TensorPack::add_tensor(const std::string& name,
                            const std::string& type,
                            const std::vector<int>& dims,
                            const std::string_view& buf)
{
    /* This function adds a tensor with associated data buffer
    */
    if(this->_tensorbase_inventory.count(std::string(name))>0)
        throw std::runtime_error("The tensor " + std::string(name)
                                               + " already exists");

    if(DATATYPE_TENSOR_STR_DOUBLE.compare(type)==0) {
        Tensor<double>* ptr = new Tensor<double>(name, type, dims, buf);
        this->_tensors_double.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_FLOAT.compare(type)==0) {
        Tensor<float>* ptr = new Tensor<float>(name, type, dims, buf);
        this->_tensors_float.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_INT64.compare(type)==0) {
        Tensor<int64_t>* ptr = new Tensor<int64_t>(name, type, dims, buf);
        this->_tensors_int64.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_INT32.compare(type)==0) {
        Tensor<int32_t>* ptr = new Tensor<int32_t>(name, type, dims, buf);
        this->_tensors_int32.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_INT16.compare(type)==0) {
        Tensor<int16_t>* ptr = new Tensor<int16_t>(name, type, dims, buf);
        this->_tensors_int16.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_INT8.compare(type)==0) {
        Tensor<int8_t>* ptr = new Tensor<int8_t>(name, type, dims, buf);
        this->_tensors_int8.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_UINT16.compare(type)==0) {
        Tensor<uint16_t>* ptr = new Tensor<uint16_t>(name, type, dims, buf);
        this->_tensors_uint16.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else if(DATATYPE_TENSOR_STR_UINT8.compare(type)==0) {
        Tensor<uint8_t>* ptr = new Tensor<uint8_t>(name, type, dims, buf);
        this->_tensors_uint8.push_front(ptr);
        this->_tensorbase_inventory[name] = ptr;
        this->_all_tensors.push_front(ptr);
    }
    else {
        throw std::runtime_error("Invalid tensor type in TensorPack: "+
                                 std::string(type));
    }
    return;
}

TensorBase* TensorPack::get_tensor(const std::string& name)
{
    /* Returns a pointer to the tensor by name
    */
    return this->_tensorbase_inventory[name];
}

void* TensorPack::get_tensor_data(const std::string& name)
{
    /* Returns a pointer to the tensor data
    memory space.
    */
    return this->_tensorbase_inventory[name]->get_data();
}

bool TensorPack::tensor_exists(const std::string& name)
{
    /* Check if a tensor exists by name
    */
    if(this->_tensorbase_inventory.count(name)>0)
        return true;
    else
        return false;
}

TensorPack::tensorbase_iterator TensorPack::tensor_begin()
{
    /* Return an iterator to the beginning of the tensors
    */
    return this->_all_tensors.begin();
}

TensorPack::tensorbase_iterator TensorPack::tensor_end()
{
    /* Return an iterator to the past the end tensor
    */
    return this->_all_tensors.end();
}

TensorPack::const_tensorbase_iterator TensorPack::tensor_cbegin()
{
    /* Return a constant iterator to the beginning of tensors
    */
    return this->_all_tensors.cbegin();
}

TensorPack::const_tensorbase_iterator TensorPack::tensor_cend()
{
    /* return a constant iterator to the past the end tensor
    */
    return this->_all_tensors.cend();
}


template <typename T>
void TensorPack::_copy_tensor_list(const std::forward_list<Tensor<T>*> &src_list,
                                   std::forward_list<Tensor<T>*> &dest_list)
{
    /* This function will copy a src_list Tensor list to the dest_list
    Tensor list.  The src_list is not cleared, but values are pushed to
    the front of the tensor list.  It is the responsibility of the caller
    to make sure both lists are in the correct state.
    */
    typename std::forward_list<Tensor<T>*>::const_iterator it =
                                                src_list.cbegin();
    typename std::forward_list<Tensor<T>*>::const_iterator it_end =
                                                dest_list.cend();

    while(it!=it_end)
    {
        dest_list.push_front(new Tensor<T>(**it));
        it++;
    }
    dest_list.reverse();
    return;
}

void TensorPack::_refresh_tensorbase_inventory()
{
    /* This function will clear the current tensorbase inventory
    and then add all tensors in the tensor lists to the
    tensorbase inventory
    */
    this->_tensorbase_inventory.clear();
    this->_all_tensors.clear();
    this->_add_to_tensorbase_inventory<double>(this->_tensors_double);
    this->_add_to_tensorbase_inventory<float>(this->_tensors_float);
    this->_add_to_tensorbase_inventory<int64_t>(this->_tensors_int64);
    this->_add_to_tensorbase_inventory<int32_t>(this->_tensors_int32);
    this->_add_to_tensorbase_inventory<int16_t>(this->_tensors_int16);
    this->_add_to_tensorbase_inventory<int8_t>(this->_tensors_int8);
    this->_add_to_tensorbase_inventory<uint16_t>(this->_tensors_uint16);
    this->_add_to_tensorbase_inventory<uint8_t>(this->_tensors_uint8);
}

template <typename T>
void TensorPack::_add_to_tensorbase_inventory(
                 const std::forward_list<Tensor<T>*>& tensor_list)
{
    /* This function will add entries in the tensor list
    into the tensorbase inventory
    */
    typename std::forward_list<Tensor<T>*>::const_iterator it =
                                            tensor_list.cbegin();
    typename std::forward_list<Tensor<T>*>::const_iterator it_end =
                                            tensor_list.cend();

    while(it!=it_end) {
        _tensorbase_inventory[(*it)->get_tensor_name()] = *it;
        _all_tensors.push_front(*it);
        it++;
    }

}

void TensorPack::_delete_all_tensors()
{
    /* This function deletes all tensors in each
    tensor list and clears the tensor inventory.
    */
    this->_delete_tensor_list<double>(this->_tensors_double);
    this->_delete_tensor_list<float>(this->_tensors_float);
    this->_delete_tensor_list<int64_t>(this->_tensors_int64);
    this->_delete_tensor_list<int32_t>(this->_tensors_int32);
    this->_delete_tensor_list<int16_t>(this->_tensors_int16);
    this->_delete_tensor_list<int8_t>(this->_tensors_int8);
    this->_delete_tensor_list<uint16_t>(this->_tensors_uint16);
    this->_delete_tensor_list<uint8_t>(this->_tensors_uint8);
    this->_tensorbase_inventory.clear();
    this->_all_tensors.clear();
}

template <typename T>
void TensorPack::_delete_tensor_list(std::forward_list<Tensor<T>*>& tensor_list)
{
    /* Deletes all Tensor objects held in a tensor list
    */
    typename std::forward_list<Tensor<T>*>::iterator it =
                                        tensor_list.begin();
    typename std::forward_list<Tensor<T>*>::iterator it_end =
                                        tensor_list.end();

    while(it!=it_end) {
        delete *it;
        it++;
    }
    return;
}