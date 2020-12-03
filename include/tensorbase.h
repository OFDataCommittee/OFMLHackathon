#ifndef SMARTSIM_TENSORBASE_H
#define SMARTSIM_TENSORBASE_H

#include "stdlib.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <cstring>
#include <stdexcept>
#include "enums/cpp_tensor_type.h"
#include "enums/cpp_memory_layout.h"

/* The strings and unordered map below is used for
fast conversion between RedisAI string and enum
value
*/
// Numeric data type of tensor elements that are allowed
static std::string DATATYPE_TENSOR_STR_FLOAT = "FLOAT";
static std::string DATATYPE_TENSOR_STR_DOUBLE = "DOUBLE";
static std::string DATATYPE_TENSOR_STR_INT8 = "INT8";
static std::string DATATYPE_TENSOR_STR_INT16 = "INT16";
static std::string DATATYPE_TENSOR_STR_INT32 = "INT32";
static std::string DATATYPE_TENSOR_STR_INT64 = "INT64";
static std::string DATATYPE_TENSOR_STR_UINT8 = "UINT8";
static std::string DATATYPE_TENSOR_STR_UINT16 = "UINT16";

static const std::unordered_map<std::string, TensorType>
    TENSOR_TYPE_MAP{
        {DATATYPE_TENSOR_STR_DOUBLE, TensorType::dbl},
        {DATATYPE_TENSOR_STR_FLOAT, TensorType::flt},
        {DATATYPE_TENSOR_STR_INT64, TensorType::int64},
        {DATATYPE_TENSOR_STR_INT32, TensorType::int32},
        {DATATYPE_TENSOR_STR_INT16, TensorType::int16},
        {DATATYPE_TENSOR_STR_INT8, TensorType::int8},
        {DATATYPE_TENSOR_STR_UINT16, TensorType::uint16},
        {DATATYPE_TENSOR_STR_UINT8, TensorType::uint8} };

static const std::unordered_map<TensorType, std::string>
    TENSOR_STR_MAP{
        {TensorType::dbl, DATATYPE_TENSOR_STR_DOUBLE},
        {TensorType::flt, DATATYPE_TENSOR_STR_FLOAT},
        {TensorType::int64, DATATYPE_TENSOR_STR_INT64},
        {TensorType::int32, DATATYPE_TENSOR_STR_INT32},
        {TensorType::int16, DATATYPE_TENSOR_STR_INT16},
        {TensorType::int8, DATATYPE_TENSOR_STR_INT8},
        {TensorType::uint16, DATATYPE_TENSOR_STR_UINT16},
        {TensorType::uint8, DATATYPE_TENSOR_STR_UINT8} };

///@file
///\brief The TensorBase class giving access to common Tensor methods and attributes
class TensorBase;

class TensorBase{

    public:
        //! TensorBase constructor
        TensorBase(const std::string& name /*!< The name used to reference the tensor*/,
                   void* data /*!< A c_ptr to the source data for the tensor*/,
                   const std::vector<size_t>& dims /*! The dimensions of the tensor*/,
                   const TensorType type /*!< The data type of the tensor*/,
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

        //! Retreive the TensorType of the tensor
        TensorType type();

        //! Return a string representation of the TensorType
        std::string type_str();

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
        TensorType _type;

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
