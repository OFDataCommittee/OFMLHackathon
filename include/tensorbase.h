#ifndef SMARTSIM_TENSORBASE_H
#define SMARTSIM_TENSORBASE_H

#include "stdlib.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <cstring>
#include <stdexcept>

// Numeric data type of tensor elements that are allowed
static std::string DATATYPE_TENSOR_STR_FLOAT = "FLOAT";
static std::string DATATYPE_TENSOR_STR_DOUBLE = "DOUBLE";
static std::string DATATYPE_TENSOR_STR_INT8 = "INT8";
static std::string DATATYPE_TENSOR_STR_INT16 = "INT16";
static std::string DATATYPE_TENSOR_STR_INT32 = "INT32";
static std::string DATATYPE_TENSOR_STR_INT64 = "INT64";
static std::string DATATYPE_TENSOR_STR_UINT8 = "UINT8";
static std::string DATATYPE_TENSOR_STR_UINT16 = "UINT16";

static const std::unordered_set<std::string> TENSOR_DATATYPES {
    DATATYPE_TENSOR_STR_FLOAT,
    DATATYPE_TENSOR_STR_DOUBLE,
    DATATYPE_TENSOR_STR_INT8,
    DATATYPE_TENSOR_STR_INT16,
    DATATYPE_TENSOR_STR_INT32,
    DATATYPE_TENSOR_STR_INT64,
    DATATYPE_TENSOR_STR_UINT8,
    DATATYPE_TENSOR_STR_UINT16
};

//An unordered_map of std::string type (key) and value
//integer that to cut down on strcmp throughout the code
static const int DOUBLE_TENSOR_TYPE = 1;
static const int FLOAT_TENSOR_TYPE = 2;
static const int INT64_TENSOR_TYPE = 3;
static const int INT32_TENSOR_TYPE = 4;
static const int INT16_TENSOR_TYPE = 5;
static const int INT8_TENSOR_TYPE = 6;
static const int UINT16_TENSOR_TYPE = 7;
static const int UINT8_TENSOR_TYPE = 8;

static const std::unordered_map<std::string, int>
    TENSOR_TYPE_MAP{
        {DATATYPE_TENSOR_STR_DOUBLE, DOUBLE_TENSOR_TYPE},
        {DATATYPE_TENSOR_STR_FLOAT, FLOAT_TENSOR_TYPE},
        {DATATYPE_TENSOR_STR_INT64, INT64_TENSOR_TYPE},
        {DATATYPE_TENSOR_STR_INT32, INT32_TENSOR_TYPE},
        {DATATYPE_TENSOR_STR_INT16, INT16_TENSOR_TYPE},
        {DATATYPE_TENSOR_STR_INT8, INT8_TENSOR_TYPE},
        {DATATYPE_TENSOR_STR_UINT16, UINT16_TENSOR_TYPE},
        {DATATYPE_TENSOR_STR_UINT8, UINT8_TENSOR_TYPE} };

///@file
///\brief The TensorBase class giving access to common Tensor methods and attributes
class TensorBase;

class TensorBase{

    public:
        //! TensorBase constructor using tensor data pointer
        TensorBase(const std::string& name /*!< The name used to reference the tensor*/,
                   const std::string& type /*!< The data type of the tensor*/,
                   void* data /*!< A c_ptr to the data of the tensor*/,
                   const std::vector<size_t>& dims /*! The dimensions of the data*/
                   );

        //! TensorBase constructor using data bufffer without tensor data pointer
        TensorBase(const std::string& name /*!< The name used to reference the tensor*/,
                   const std::string& type /*!< The data type of the tensor*/,
                   const std::vector<size_t>& dims /*! The dimensions of the data*/,
                   const std::string_view& data_buf /*! The data buffer*/
                   );

        //! Copy contrustor for Tensorbase
        TensorBase(const TensorBase& tb);

        //! Copy assignment operator for TensorBase
        TensorBase&  operator=(const TensorBase& tb);
        //! TensorBase destructor
        virtual ~TensorBase();

        //! Retrive the tensor name
        std::string get_tensor_name();

        //! Retreive the tensor type
        std::string get_tensor_type();

        //! Retrieve the tensor dims
        std::vector<size_t> get_tensor_dims();

        //! Get the tensor data as a buffer
        std::string_view get_data_buf();

        //! Get pointer to tensor memory space
        virtual void* get_data() = 0;

        //! Fill a user provided memory space with values from tensor buffer
        virtual void fill_data_from_buf(void* data /*!< Pointer to the allocated memory space*/,
                                       std::vector<size_t> dims /*!< The dimensions of the memory space*/,
                                       const std::string& type /*!< The datatype of the allocated memory space*/
                                       ) = 0;


        protected:

        //! Tensor name
        std::string _name;

        //! Tensor type
        std::string _type;

        //! Tensor dims
        std::vector<size_t> _dims;

        //! Pointer to the data memory space
        void* _data;

        //! The data buffer
        char* _data_buf;

        //! The length of the data buff in bytes
        size_t _buf_size;

        //! Function to generate the data buffer from the data
        virtual void _generate_data_buf() = 0;

        //TODO implement this
        //! Function to copy tensor data into this tensor data
        //virtual void _copy_data(void* data /*!< A c_ptr to the data to copy*/,
        //                        std::vector<int> dims /*! The dimensions of the data to copy*/
        //                        ) = 0;

        private:

        //! Function to check for errors in constructor inputs
        void _check_constructor_input(const std::string& name /*!< The name used to reference the tensor*/,
                                      const std::string& type /*!< The data type of the tensor*/,
                                      std::vector<size_t> dims /*! The dimensions of the data*/
                                      );

};
#endif //SMARTSIM_TENSORBASE_H
