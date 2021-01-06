#ifndef SMARTSIM_TENSORPACK_H
#define SMARTSIM_TENSORPACK_H

#include "stdlib.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <forward_list>
#include "tensor.h"
#include "tensorbase.h"

///@file
///\brief The TensorPack class is a container for multiple tensors

namespace SILC {

class TensorPack;

class TensorPack
{
    public:

        //! Dataset constructor
        TensorPack();

        //! Copy constructor
        TensorPack(const TensorPack& tensorpack);

        //! Move constructor
        TensorPack(TensorPack&& tensorpack);

        //! Copy assignment operator
        TensorPack& operator=(const TensorPack& tensorpack);

        //! Move assignment operator
        TensorPack& operator=(TensorPack&& tensorpack);

        //! TensorPack destructor
        ~TensorPack();

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

        //! Forward lists of the actual tensor types that will be used
        //! for memory management and operations on non-base class tensors
        std::forward_list<Tensor<double>*> _tensors_double;
        std::forward_list<Tensor<float>*> _tensors_float;
        std::forward_list<Tensor<int64_t>*> _tensors_int64;
        std::forward_list<Tensor<int32_t>*> _tensors_int32;
        std::forward_list<Tensor<int16_t>*> _tensors_int16;
        std::forward_list<Tensor<int8_t>*> _tensors_int8;
        std::forward_list<Tensor<uint16_t>*> _tensors_uint16;
        std::forward_list<Tensor<uint8_t>*> _tensors_uint8;

        //! Copy a Tensor forward list
        template <typename T>
        static void _copy_tensor_list(
                const std::forward_list<Tensor<T>*>& src_list /*!< The source tensor list*/,
                std::forward_list<Tensor<T>*>& dest_list /*!< The destination tensor list*/);

        //! This function will clear and refresh the tensorbase inventory
        void _refresh_tensorbase_inventory();

        //! Add all entries in the Tensor list to the tensor inventory
        template <typename T>
        void _add_to_tensorbase_inventory(
            const std::forward_list<Tensor<T>*>& tensor_list /*!< The tensor list to add to the inventory*/);

        //! This deletes all tensors in each tensor list
        void _delete_all_tensors();

        //! Delete an individual tensor list
        template <typename T>
        void _delete_tensor_list(
            std::forward_list<Tensor<T>*>& tensor_list /*!< The tensor list to delete*/);
};

} //namespace SILC

#endif //SMARTSIM_TENSORPACK_H