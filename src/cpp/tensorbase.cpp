#include "tensorbase.h"

TensorBase::TensorBase(const std::string& name,
                       const std::string& type,
                       void* data,
                       const std::vector<size_t>& dims)
{
    /* The TensorBase constructor makes a copy of the name, type, and dims
    associated with the tensor, but does not copy the data of the tensor.
    It is assumed that the data is valid for the life of the tensor.
    */

    this->_check_constructor_input(name, type, dims);

    if(!data)
        throw std::runtime_error("Must provide non-Null "\
                                 "pointer to data.");

    this->_name = name;
    this->_type = type;
    this->_data = data;
    this->_dims = dims;
    this->_data_buf = 0;
    this->_buf_size = 0;
}

TensorBase::TensorBase(const std::string& name,
                       const std::string& type,
                       const std::vector<size_t>& dims,
                       const std::string_view& data_buf)
{
    /* This TensorBase constructor sets the data pointer to 0,
    and copies the provided data buffer.
    */
    this->_check_constructor_input(name, type, dims);
    this->_name = name;
    this->_type = type;
    this->_data = 0;
    this->_dims = dims;

    //TODO Look into if it is possible not copy the databuffer
    //but pass a Reply uniqueptr to this constructor.
    this->_buf_size = data_buf.size();
    this->_data_buf = (char*) malloc(sizeof(char) * this->_buf_size);
    data_buf.copy(this->_data_buf, this->_buf_size);
}

TensorBase::TensorBase(const TensorBase& tb)
{
    /* This is the copy constructor for tensorbase.
    It will allocate new memory and copy everything
    except for the data.  The data must be copied
    by the Tensor class, which is enforced by
    the function call at this end of this function.
    */
    this->_name = tb._name;
    this->_type = tb._type;
    this->_buf_size = tb._buf_size;
    this->_data_buf = 0;

    if(this->_buf_size>0) {
        this->_data_buf = (char*)malloc(this->_buf_size);
        std::memcpy(this->_data_buf, tb._data_buf,
                    this->_buf_size);
    }

    this->_dims = tb._dims;
    //TODO implement copy data
    //this->_copy_data(tb._data, tb._dims);
    return;
}

TensorBase& TensorBase::operator=(const TensorBase& tb)
{
    /* This function is the copy assignment operator.
    It will allocate new memory for the buffer.
    */
    this->_name = tb._name;
    this->_type = tb._type;
    this->_dims = tb._dims;
    this->_data = tb._data;

    if(this->_data_buf)
        free(this->_data_buf);

    this->_buf_size = tb._buf_size;
    if(this->_buf_size>0) {
        this->_data_buf = (char*)malloc(this->_buf_size);
        std::memcpy(this->_data_buf, tb._data_buf,
                    this->_buf_size);
    }

    this->_dims = tb._dims;
    //TODO implement copy data
    //this->_copy_data(tb._data, tb._dims);
    return *this;
}

TensorBase::~TensorBase()
{
    if(this->_buf_size>0)
        free(this->_data_buf);
}

std::string TensorBase::get_tensor_name()
{
    /* Return the tensor name
    */
    return this->_name;
}

std::string TensorBase::get_tensor_type()
{
    /* Return the tensor type
    */
   return this->_type;
}

std::vector<size_t> TensorBase::get_tensor_dims()
{
    /* Return the tensor dims
    */
   return this->_dims;
}

std::string_view TensorBase::get_data_buf()
{
    /* This function returns a std::string_view of tensor
    data translated into a data buffer.  If the data buffer
    has not yet been created, the data buffer will be
    created before returning.
    */
    if(!this->_data_buf) {
        this->_generate_data_buf();
    }
    return std::string_view(this->_data_buf, this->_buf_size);
}

void TensorBase::_check_constructor_input(const std::string& name,
                                          const std::string& type,
                                          std::vector<size_t> dims)
{
    /* This function checks the validity of constructor inputs.
    This was taken out of the constructor to reduce code
    duplication from multiple constructors.
    */

    if(name.size()==0)
        throw std::runtime_error("A name must "\
                                 "be provided for the tensor");

    if(name.compare(".meta")==0)
        throw std::runtime_error(".META is an internally reserved "\
                                 "name that is not allowed.");

    if(TENSOR_DATATYPES.count(type) == 0)
        throw std::runtime_error("Unsupported tensor data type " +
                                 std::string(type));

    if(dims.size()==0)
        throw std::runtime_error("Must provide a dimensions vector "\
                                 "with at least one dimension.");

    std::vector<size_t>::const_iterator it = dims.cbegin();
    std::vector<size_t>::const_iterator it_end = dims.cend();
    while(it!=it_end) {
        if((*it)<0) {
            throw std::runtime_error("All tensor dimensions must "\
                                     "be positive.");
        }
        it++;
    }
    return;
}