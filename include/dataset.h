#ifndef SMARTSIM_DATASET_H
#define SMARTSIM_DATASET_H
#ifdef __cplusplus
#include "stdlib.h"
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <forward_list>
#include "tensor.h"
#include "command.h"
#include "metadata.h"
#include "memorylist.h"
#include "tensorpack.h"
#include "tensor.h"
#include "enums/cpp_tensor_type.h"
#include "enums/cpp_memory_layout.h"
#include "enums/cpp_metadata_type.h"

///@file
///\brief The DataSet class encapsulating numeric data and metadata.

class DataSet;

class DataSet
{
    public:

        //! DataSet constructor
        DataSet(const std::string& name /*!< The name used to reference the dataset*/
                );

        //! DataSet move constructor
        DataSet(DataSet&& dataset /*!< The dataset to move for construction*/);

        //! DataSet move assignment operator
        DataSet& operator=(DataSet&& dataset /*!< The dataset to move for assignment*/);

        //! Dataset constructor using serialized buffer
        DataSet(const std::string& name /*!< The name used to reference the dataset*/,
                char* buf /*!< The buffer used for object construction*/,
                size_t buf_size /*!< The buffer length*/
                );

        //! Add a tensor to the dataset
        void add_tensor(const std::string& name /*!< The name used to reference the tensor*/,
                        void* data /*!< A c_ptr to the data of the tensor*/,
                        const std::vector<size_t>& dims /*! The dimensions of the data*/,
                        const TensorType type /*!< The data type of the tensor*/,
                        const MemoryLayout mem_layout /*!< The MemoryLayout enum describing the layout of source data*/
                        );

        //! Add metadata field to the DataSet.  Default behavior is to append existing fields.
        void add_meta(const std::string& name /*!< The name used to reference the metadata*/,
                      const void* data /*!< A c_ptr to the metadata*/,
                      const MetaDataType type /*!< The data type of the metadata*/
                      );

        //! Get tensor data and return an allocated multi-dimensional array
        void get_tensor(const std::string& name /*!< The name used to reference the tensor*/,
                        void*& data /*!< A c_ptr to the tensor data */,
                        std::vector<size_t>& dims /*! The dimensions of the tensor retrieved*/,
                        TensorType& type /*!< The data type of the tensor*/,
                        const MemoryLayout mem_layout /*!< The MemoryLayout enum describing the layout of source data*/
                        );

        //! Get tensor data and return an allocated multi-dimensional array
        void get_tensor(const std::string& name /*!< The name used to reference the tensor*/,
                        void*& data /*!< A c_ptr to the tensor data */,
                        size_t*& dims /*! The dimensions of the tensor retrieved*/,
                        size_t& n_dims /*! The number of dimensions retrieved*/,
                        TensorType& type /*!< The data type of the tensor*/,
                        const MemoryLayout mem_layout /*!< The MemoryLayout enum describing the layout of source data*/
                        );

        //! Get tensor data and fill an already allocated array memory space
        void unpack_tensor(const std::string& name /*!< The name used to reference the tensor*/,
                           void* data /*!< A c_ptr to the data of the tensor*/,
                           const std::vector<size_t>& dims /*! The dimensions of the supplied memory space*/,
                           const TensorType type /*!< The data type corresponding to the user memory space*/,
                           const MemoryLayout mem_layout /*!< The MemoryLayout enum describing the layout of source data*/
                           );

        //! Get metadata field from the DataSet
        void get_meta(const std::string& name /*!< The name used to reference the metadata*/,
                      void*& data /*!< A c_ptr reference to the metadata*/,
                      size_t& length /*!< The length of the meta*/,
                      MetaDataType& type /*!< The data type of the metadata*/
                      );

        //! The name of the DataSet
        std::string name;

        typedef TensorPack::tensorbase_iterator tensor_iterator;
        typedef TensorPack::const_tensorbase_iterator const_tensor_iterator;

        //! Returns an iterator pointing to the tensor
        tensor_iterator tensor_begin();

        //! Returns an const_iterator pointing to the tensor
        const_tensor_iterator tensor_cbegin();

        //! Returns an iterator pointing to the past-the-end tensor
        tensor_iterator tensor_end();

        //! Returns an const_iterator pointing to the past-the-end tensor
        const_tensor_iterator tensor_cend();

        //! Return a serialization of the meatdata
        std::string_view get_metadata_buf();

        //! Return the data type of tensor to the user
        std::string get_tensor_type(const std::string& name /*!< The name used to reference the tensor*/
                                    );

        friend class SmartSimClient;
    protected:

        inline void _add_to_tensorpack(const std::string& name /*!< The name used to reference the tensor*/,
                                       void* data /*!< A c_ptr to the data of the tensor*/,
                                       const std::vector<size_t>& dims /*! The dimensions of the data*/,
                                       const TensorType type /*! The type of the tensor*/,
                                       const MemoryLayout mem_layout /*!< The MemoryLayout enum describing the layout of source data*/
                                       );
    private:
        //! The meta data object
        MetaData _metadata;
        //! The tensorpack object that holds all tensors
        TensorPack _tensorpack;
        //! MemoryList object to hold allocated size_t arrays stemming from get_tensor queries
        MemoryList<size_t> _dim_queries;
};
#endif
#endif //SMARTSIM_DATASET_H