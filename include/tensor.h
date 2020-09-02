#ifndef SMARTSIM_TENSOR_H
#define SMARTSIM_TENSOR_H

#include "stdlib.h"
#include <string>
#include <vector>
#include <unordered_set>
#include "command.h"

// Numeric data type of tensor elements that are allowed
static const char* DATATYPE_STR_FLOAT = "FLOAT";
static const char* DATATYPE_STR_DOUBLE = "DOUBLE";
static const char* DATATYPE_STR_INT8 = "INT8";
static const char* DATATYPE_STR_INT16 = "INT16";
static const char* DATATYPE_STR_INT32 = "INT32";
static const char* DATATYPE_STR_INT64 = "INT64";
static const char* DATATYPE_STR_UINT8 = "UINT8";
static const char* DATATYPE_STR_UINT16 = "UINT16";

static const std::unordered_set<const char*> TENSOR_DATATYPES {
    DATATYPE_STR_FLOAT,
    DATATYPE_STR_DOUBLE,
    DATATYPE_STR_INT8,
    DATATYPE_STR_INT16,
    DATATYPE_STR_INT32,
    DATATYPE_STR_INT64,
    DATATYPE_STR_UINT8,
    DATATYPE_STR_UINT16
};

///@file
///\brief The Tensor class encapsulating numeric data tensors
class Tensor;

class Tensor
{
    public:

        //! Tensor constructor
        Tensor(const char* name /*!< The name used to reference the tensor*/,
               const char* type /*!< The data type of the tensor*/,
               void* data /*!< A c_ptr to the raw data*/,
               std::vector<int> /*! The dimensions of the data*/
        );

        //! Tensor destructor
        ~Tensor();

        //! Generate Tensor send command
        //TODO move and improve this command
        Command generate_send_command(std::string key_prefix, std::string key_suffix);

        //! Tensor name
        std::string name;
        //! Tensor type
        std::string type;
        //! The dimensions of the tensor
        std::vector<int> dims;
        //! Pointer to the first memory address of the data
        void* data;

    private:

        //! The data buffer
        char* _data_buf;
        //! The length of the data buff in bytes
        unsigned long long _buf_size;

        //! Function to generate the data buffer from the data
        template<typename T> void _generate_data_buf();
        //! Function to copy values from tensor into buffer
        template<typename T> void* _copy_tensor_vals_to_buf(void* data,
                                                           int* dims, int n_dims,
                                                           void* buf);
        //! Function pointer to the templated version of _generate_data_buf based on tensor type.
        void (Tensor::*_generate_data_buf_ptr)();
};

#endif //SMARTSIM_TENSOR_H
