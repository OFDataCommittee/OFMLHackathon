#ifndef SMARTSIM_TENSORPACK_H
#define SMARTSIM_TENSORPACK_H

#include "stdlib.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <forward_list>
#include "tensor.h"
#include "tensorbase.h"
#include "tensorlist.h"

///@file
///\brief The TensorPack class is a container for multiple tensors

namespace SILC {

class TensorPack;

class TensorPack
{
    public:

        //! Dataset constructor
        TensorPack() = default;

        //! Copy constructor
        TensorPack(const TensorPack& tensorpack);

        //! Move constructor
        TensorPack(TensorPack&& tensorpack) = default;

        //! Copy assignment operator
        TensorPack& operator=(const TensorPack& tensorpack);

        //! Move assignment operator
        TensorPack& operator=(TensorPack&& tensorpack) = default;

        //! TensorPack destructor
        ~TensorPack() = default;

        //! Add a tensor to the dataset
        void add_tensor(const std::string& name /*!< The name used to reference the tensor*/,
                        void* data /*!< A c_ptr to the data of the tensor*/,
                        const std::vector<size_t>& dims /*! The dimensions of the data*/,
                        const TensorType type /*!< The data type of the tensor*/,
                        const MemoryLayout mem_layout /*! The memory layout of the data*/
                        );

        //! Method to add a tensor object that has already been created on the heap.
        //! DO NOT add tensors allocated on the stack that may be deleted outside of
        //! the tensor pack.  This function will cast the TensorBase to the correct
        //! Tensor<T> type.
        void add_tensor(TensorBase* tensor /*!< Pointer to the tensor allocated on the heap*/
                        );

        //! Iterators for tensors
        typedef std::forward_list<TensorBase*>::iterator tensorbase_iterator;
        typedef std::forward_list<TensorBase*>::const_iterator const_tensorbase_iterator;

        //! Return a TensorBase pointer based on name
        TensorBase* get_tensor(const std::string& name /*!< The name used to reference the tensor*/
                               );

        //! Return a pointer to the tensor data memory space
        void* get_tensor_data(const std::string& name /*!< The name used to reference the tensor*/
                              );

        //! Returns boolean indicating if tensor by that name exists
        bool tensor_exists(const std::string& name /*!< The name used to reference the tensor*/
                           );

        //! Returns an iterator pointing to the tensor
        tensorbase_iterator tensor_begin();
        //! Returns an const_iterator pointing to the tensor
        const_tensorbase_iterator tensor_cbegin();
        //! Returns an iterator pointing to the past-the-end tensor
        tensorbase_iterator tensor_end();
        //! Returns an const_iterator pointing to the past-the-end tensor
        const_tensorbase_iterator tensor_cend();

    private:

        //! A forward list of all TensorBase to make iterating easier
        std::forward_list<TensorBase*> _all_tensors;

        //! A map used to query and retrieve tensors. We can only return
        //! the TensoreBase values since the Tensors are templated.
        std::unordered_map<std::string, TensorBase*> _tensorbase_inventory;

        //! TensorLists for memory management and operations on non-base class tensors
        TensorList<double> _tensors_double;
        TensorList<float> _tensors_float;
        TensorList<int64_t> _tensors_int64;
        TensorList<int32_t> _tensors_int32;
        TensorList<int16_t> _tensors_int16;
        TensorList<int8_t> _tensors_int8;
        TensorList<uint16_t> _tensors_uint16;
        TensorList<uint8_t> _tensors_uint8;

        //! Rebuild the tensor map and forward_list iterator
        void _rebuild_tensor_inventory();

        //! Add a TensorList to the tensor inventory
        template <typename T>
        void _add_tensorlist_to_inventory(TensorList<T>& t_list /*!< The TensorList to add*/
                                          );
};

} //namespace SILC

#endif //SMARTSIM_TENSORPACK_H