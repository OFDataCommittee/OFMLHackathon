#include "tensor.h"

Tensor::Tensor(const char* name, const char* type, void* data, std::vector<int> dims)
{
    /* The Tensor constructor makes a copy of the name, type, and dims associated
    with the tensor, but does not copy the data of the tensor.  It is assumed
    that the data is valid until the message is sent.
    */

    //TODO we need to have a custom error class
    if(std::strlen(name)==0)
        throw std::runtime_error("A name must "\
                                 "be provided for the tensor");

    if(TENSOR_DATATYPES.count(type) <= 0)
        throw std::runtime_error("Unsupported tensor data type " +
                                 std::string(type));

    if(!data)
        throw std::runtime_error("Must provide non-Null pointer to data.");

    if(dims.size()==0)
        throw std::runtime_error("Must provide a dimensions vector "\
                                 "with at least one dimension.");

    for(int i=0; i<dims.size(); i++) {
        if( dims[i] <= 0)
            throw std::runtime_error("All tensor dimensions must "\
                                     "be positive.");
    }

    this->name = std::string(name);
    this->type = std::string(type);
    this->data = data;
    this->dims = dims;

    this->_data_buf = 0;
    this->_buf_size = 0;

    // Set all function pointers in the constructor so switches
    // do not need to be done in any other section of client
    if(this->type.compare(DATATYPE_STR_FLOAT)==0) {
        this->_generate_data_buf_ptr=&Tensor::_generate_data_buf<float>;
    }
    else if(this->type.compare(DATATYPE_STR_DOUBLE)==0) {
        this->_generate_data_buf_ptr=&Tensor::_generate_data_buf<double>;
    }
    else if(this->type.compare(DATATYPE_STR_INT8)==0) {
        this->_generate_data_buf_ptr=&Tensor::_generate_data_buf<int8_t>;
    }
    else if(this->type.compare(DATATYPE_STR_INT16)==0) {
        this->_generate_data_buf_ptr=&Tensor::_generate_data_buf<int16_t>;
    }
    else if(this->type.compare(DATATYPE_STR_INT32)==0) {
        this->_generate_data_buf_ptr=&Tensor::_generate_data_buf<int32_t>;
    }
    else if(this->type.compare(DATATYPE_STR_INT64)==0) {
        this->_generate_data_buf_ptr=&Tensor::_generate_data_buf<int64_t>;
    }
    else if(this->type.compare(DATATYPE_STR_UINT8)==0) {
        this->_generate_data_buf_ptr=&Tensor::_generate_data_buf<uint8_t>;
    }
    else if(this->type.compare(DATATYPE_STR_UINT16)==0) {
        this->_generate_data_buf_ptr=&Tensor::_generate_data_buf<uint16_t>;
    }
}

Tensor::~Tensor()
{
    delete[] this->_data_buf;
}

Command Tensor::generate_send_command(std::string key_prefix, std::string key_suffix)
{
    /* This function generates the command for sending a tensor to the database
    */

    Command cmd;
    cmd.add_field("AI.TENSORSET");
    cmd.add_field(key_prefix + this->name + key_suffix);
    cmd.add_field(this->type);
    for(int i=0; i<dims.size(); i++)
        cmd.add_field(std::to_string(dims[i]));
    cmd.add_field("BLOB");
    cmd.add_field_ptr(this->_data_buf, this->_buf_size);
    //TODO need to add the blob value here
    return cmd;

}

template <typename T>
void Tensor::_generate_data_buf()
{
    /* This function will turn the tensor data into a binary string
    buffer that can be used for data transfer.
    */

    //TODO there is an optimization possible that if the data is contiguous in memory
    //(e.g. special fortran or numpy arrays) or if it is 1D data (all languages)
    //We can have data_buf just point to data and there will be absolutely no copies.
    //until we get into redis itself.

    int n_bytes = 1;
    for (int i = 0; i < this->dims.size(); i++) {
        n_bytes *= dims[i];
    }
    n_bytes *= sizeof(T);

    this->_buf_size = n_bytes;
    this->_data_buf = (char*)malloc(n_bytes);
    //TODO Now that this is in a Tensor class, we might be able to remove some of
    //the arguments being passed to clean it up, but may not be able to because
    //of recursive nature
    this->_copy_tensor_vals_to_buf<T>(this->data,
                                      &(this->dims[0]),
                                      this->dims.size(),
                                      (void*)this->_data_buf);
    return;
}

template <typename T>
void* Tensor::_copy_tensor_vals_to_buf(void* data, int* dims, int n_dims,
                                       void* buf)
{
    /* This function will copy the tensor data values into the
    binary buffer
    */

    //TODO we should check at some point that we don't exceed buf length
    //TODO test in multi dimensions with each dimension having a
    //different set of values make sure (dimensions are independent)
    if(n_dims > 1) {
        T** current = (T**) data;
        for(int i = 0; i < dims[0]; i++) {
          buf = this->_copy_tensor_vals_to_buf<T>(*current, &dims[1], n_dims-1, buf);
          current++;
        }
    }
    else {
        std::memcpy(buf, data, sizeof(T)*dims[0]);
        return &((T*)buf)[dims[0]];
    }
    return buf;
}

