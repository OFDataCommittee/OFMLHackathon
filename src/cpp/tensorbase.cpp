#include "tensorbase.h"

TensorBase::TensorBase(const std::string& name,
                       const std::string& type,
                       void* data,
                       const std::vector<size_t>& dims,
                       const MemoryLayout mem_layout)
{
    /* The TensorBase constructor makes a copy of the
    name, type, and dims associated with the tensor.
    The provided data is copied into a memory space
    owned by the tensor.
    */

    this->_check_inputs(data, name, type, dims);

    this->_name = name;
    this->_type = type;
    this->_dims = dims;
    this->_data = 0;
}

TensorBase::TensorBase(const TensorBase& tb)
{
    /* This is the copy constructor for TensorBase.
    Copying of the data is left to the child classes.
    */
    this->_name = tb._name;
    this->_type = tb._type;
    this->_dims = tb._dims;
    this->_data = 0;
    return;
}

TensorBase::TensorBase(TensorBase&& tb)
{
    /* This is the move constructor for TensorBase.
    Moving of dynamically allocated tensor data
    memory and data pointers is the responsibility
    of the child class.
    */
    this->_name = std::move(tb._name);
    this->_type = std::move(tb._type);
    this->_dims = std::move(tb._dims);
    this->_data = 0;
    return;
}

TensorBase& TensorBase::operator=(const TensorBase& tb)
{
    /* This is the copy assignment operator for
    TensorBase. Copying of the data is left to
    the child class.
    */
    this->_name = tb._name;
    this->_type = tb._type;
    this->_dims = tb._dims;
    this->_data = 0;
    return *this;
}

TensorBase& TensorBase::operator=(TensorBase&& tb)
{
    /* This is the move assignment operator for
    TensorBase. Moving of dynamically allocated tensor
    data memory and data pointers is the responsibility
    of the child class.
    */
    if(this!=&tb) {
        this->_name = std::move(tb._name);
        this->_type = std::move(tb._type);
        this->_dims = std::move(tb._dims);
        this->_data = 0;
    }
    return *this;
}

TensorBase::~TensorBase()
{
}

std::string TensorBase::name()
{
    /* Return the tensor name.
    */
    return this->_name;
}

std::string TensorBase::type()
{
    /* Return the tensor type.
    */
   return this->_type;
}

std::vector<size_t> TensorBase::dims()
{
    /* Return the tensor dims
    */
   return this->_dims;
}

size_t TensorBase::num_values()
{
    /* Return the total number of values in the tensor
    */
    size_t n_values = this->_dims[0];
    for(size_t i=1; i<this->_dims.size(); i++) {
        n_values *= this->_dims[i];
    }
    return n_values;
}

void* TensorBase::data()
{
    /* This function returns a pointer to the
    tensor data.
    */
   return this->_data;
}

std::string_view TensorBase::buf()
{
    /* This function returns a std::string_view of tensor
    data translated into a data buffer.  If the data buffer
    has not yet been created, the data buffer will be
    created before returning.
    */
    return std::string_view((char*)this->_data,
                            this->_n_data_bytes());
}

inline void TensorBase::_check_inputs(const void* src_data,
                                      const std::string& name,
                                      const std::string& type,
                                      const std::vector<size_t>& dims)
{
    /* This function checks the validity of constructor
    inputs. This was taken out of the constructor to
    make the constructor actions more clear.
    */

    if(!src_data)
        throw std::runtime_error("Must provide non-Null "\
                                 "pointer to data.");


    if(name.size()==0)
        throw std::runtime_error("A name must be "\
                                 "provided for the tensor");

    if(name.compare(".meta")==0)
        throw std::runtime_error(".META is an internally "\
                                 "reserved name that is not "\
                                 "allowed.");

    if(TENSOR_DATATYPES.count(type) == 0)
        throw std::runtime_error("Unsupported tensor data type " +
                                 std::string(type));

    if(dims.size()==0)
        throw std::runtime_error("Must provide a dimensions "\
                                 "vector with at least one "\
                                 "dimension.");

    std::vector<size_t>::const_iterator it = dims.cbegin();
    std::vector<size_t>::const_iterator it_end = dims.cend();
    while(it!=it_end) {
        if((*it)<=0) {
            throw std::runtime_error("All tensor dimensions "\
                                     "must be positive.");
        }
        it++;
    }

    return;
}