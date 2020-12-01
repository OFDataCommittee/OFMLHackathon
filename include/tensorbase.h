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

enum MemoryLayout{
    nested=1,
    contiguous=2,
    fortran_nested=3,
    fortran_contiguous=4
};

///@file
///\brief The TensorBase class giving access to common Tensor methods and attributes
class TensorBase;

class TensorBase{

    public:
        //! TensorBase constructor
        TensorBase(const std::string& name /*!< The name used to reference the tensor*/,
                   const std::string& type /*!< The data type of the tensor*/,
                   void* data /*!< A c_ptr to the source data for the tensor*/,
                   const std::vector<size_t>& dims /*! The dimensions of the tensor*/,
                   const MemoryLayout mem_layout /*! The memory layout of the source data*/
                   );

        //! TensorBase destructor
        virtual ~TensorBase();

        //! Copy contrustor for TensorBase
        TensorBase(const TensorBase& tb);

        //! Move constructor for TensorBase
        TensorBase(TensorBase&& tb);

        //! Copy assignment operator for TensorBase
        TensorBase&  operator=(const TensorBase& tb);

        //! Move assignment operator for TensorBase
        TensorBase& operator=(TensorBase&& tb);

        //! Retrive the tensor name
        std::string name();

        //! Retreive the tensor type
        std::string type();

        //! Retrieve the tensor dims
        std::vector<size_t> dims();

        //! Get the total number of values in the tensor
        size_t num_values();

        //! Get the tensor data pointer
        void* data();

        //! Get the tensor data as a buf
        virtual std::string_view buf();

        //! Get pointer to the tensor memory space in the specific layout
        virtual void* data_view(const MemoryLayout mem_layout /*!< The MemoryLayout enum describing the layout of data view*/
                                ) = 0;

        //! Fill a user provided memory space with values from tensor data
        virtual void fill_mem_space(void* data /*!< Pointer to the allocated memory space*/,
                                    std::vector<size_t> dims /*!< The dimensions of the memory space*/
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

        //TODO implement this
        //! Function to copy tensor data into this tensor data
        //virtual void _copy_data(void* data /*!< A c_ptr to the data to copy*/,
        //                        std::vector<int> dims /*! The dimensions of the data to copy*/
        //                        ) = 0;

        private:

        //! Function to check for errors in constructor inputs
        inline void _check_inputs(const void* src_data /*!< A pointer to the data source for the tensor*/,
                                  const std::string& name /*!< The name used to reference the tensor*/,
                                  const std::string& type /*!< The data type of the tensor*/,
                                  const std::vector<size_t>& dims /*! The dimensions of the data*/
                                  );

        //! Set the tensor data from a src memory location
        virtual void _set_tensor_data(void* src_data,
                                      const std::vector<size_t>& dims,
                                      const MemoryLayout mem_layout) = 0;

        //! Get the total number of bytes of the Tensor data
        virtual size_t _n_data_bytes() = 0;
};
#endif //SMARTSIM_TENSORBASE_H
