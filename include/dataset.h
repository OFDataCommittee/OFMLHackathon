#ifndef SMARTSIM_DATASET_H
#define SMARTSIM_DATASET_H

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
        DataSet(DataSet&& dataset);

        //! DataSet move assignment operator
        DataSet& operator=(DataSet&& dataset);

        //! Dataset constructor using serialized buffer
        DataSet(const std::string& name /*!< The name used to reference the dataset*/,
                char* buf /*!< The buffer used for object construction*/,
                unsigned long long buf_size /*!< The buffer length*/
                );

        //! DataSet copy constructor
        DataSet(const DataSet& dataset /*!< The dataset to copy for construction*/
                );

        //! Add a tensor to the dataset
        void add_tensor(const std::string& name /*!< The name used to reference the tensor*/,
                        const std::string& type /*!< The data type of the tensor*/,
                        void* data /*!< A c_ptr to the data of the tensor*/,
                        std::vector<int> dims /*! The dimensions of the data*/
                        );

        //! Add metadata field to the DataSet.  Default behavior is to append existing fields.
        void add_meta(const std::string& name /*!< The name used to reference the metadata*/,
                      const std::string& type /*!< The data type of the metadata*/,
                      const void* data /*!< A c_ptr to the metadata*/
                      );

        //! Get tensor data and return an allocated multi-dimensional array
        void get_tensor(const std::string& name /*!< The name used to reference the tensor*/,
                        const std::string& type /*!< The data type of the tensor*/,
                        void*& data /*!< A c_ptr to the tensor data */,
                        std::vector<int>& dims /*! The dimensions of the tensor retrieved*/
                        );

        //! Get tensor data and fill an already allocated array
        void unpack_tensor(const std::string& name /*!< The name used to reference the tensor*/,
                           const std::string& type /*!< The data type of the tensor*/,
                           void* data /*!< A c_ptr to the data of the tensor*/,
                           std::vector<int> dims /*! The dimensions of the data*/
                           );

        //TODO rethink this function prototype.  Seems annoying ot have void** and int*
        //Maybe return std::pair<void* int> or maybe
        //! Get metadata field from the DataSet
        void get_meta(const std::string& name /*!< The name used to reference the metadata*/,
                      const std::string& type /*!< The data type of the metadata*/,
                      void*& data /*!< A c_ptr reference to the metadata*/,
                      int& length /*!< The length of the meta*/
                      );

        //! Method for adding a tensor when we only have the buffer and no data
        //! pointer.  This is meant to be used primarily by the client.
        void add_tensor_buf_only(const std::string& name /*!< The name used to reference the tensor*/,
                                 const std::string& type /*!< The data type of the tensor*/,
                                 std::vector<int> dims /*! The dimensions of the data*/,
                                 std::string_view buf /*!< A c_ptr to the data of the tensor*/
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

    private:
        //! The meta data object
        MetaData _metadata;
        //! The tensorpack object that holds all tensors
        TensorPack _tensorpack;
};
#endif //SMARTSIM_DATASET_H