#ifndef SMARTSIM_TENSOR_H
#define SMARTSIM_TENSOR_H

#include "stdlib.h"
#include <string>
#include <stdexcept>
#include "tensorbase.h"
#include "memorylist.h"

///@file
///\brief The Tensor class for data and buffer tensor operations

template <class T>
class Tensor : public TensorBase
{
    public:

        //! Tensor constructor
        Tensor(const std::string& name /*!< The name used to reference the tensor*/,
               void* data /*!< A c_ptr to the source data for the tensor*/,
               const std::vector<size_t>& dims /*! The dimensions of the tensor*/,
               const TensorType type /*!< The data type of the tensor*/,
               const MemoryLayout mem_layout /*! The memory layout of the source data*/
                );

        //! Tensor destructor
        virtual ~Tensor();

        //! Tensor copy constructor
        Tensor(const Tensor<T>& tensor);

        //! Tensor move constructor
        Tensor(Tensor<T>&& tensor);

        //! Tensor copy assignment operator
        Tensor<T>& operator=(const Tensor<T>& tensor);

        //! Tensor move assignment operator
        Tensor<T>& operator=(Tensor<T>&& tensor);

        //! Get pointer to the tensor memory space in the specific layout
        virtual void* data_view(const MemoryLayout mem_layout /*!< The MemoryLayout enum describing the layout of data view*/
                                );

        //! Fill a user provided memory space with values from tensor data
        virtual void fill_mem_space(void* data /*!< Pointer to the allocated memory space*/,
                                    std::vector<size_t> dims /*!< The dimensions of the memory space*/,
                                    MemoryLayout mem_layout /*!< The memory layout of the provided memory space*/
                                    );

    protected:

    private:

        //! Function to copy values from nested memory structure to contiguous memory structure
        void* _copy_nested_to_contiguous(void* src_data /*!< The nested data to copy to the contiguous memory location*/,
                                         const size_t* dims /*!< The dimensions of src_data*/,
                                         const size_t n_dims /*!< The number of dimensions in the src_data*/,
                                         void* dest_data /*!< The destination memory space (contiguous)*/
                                         );

        //! Function to copy from a flat, contiguous memory structure to a provided nested structure
        void _fill_nested_mem_with_data(void* data /*!< The destination nested space for the flat data*/,
                                        size_t* dims /*!< The dimensions of nested memory space*/,
                                        size_t n_dims /*!< The number of dimensions in dims*/,
                                        size_t& data_position /*!< The current position of copying the flat memory space*/,
                                        void* tensor_data /*!< The flat memory structure data*/
                                        );

        //! Builds nested array structure to point to the provided flat, contiguous memory space.  The space is returned via the data input pointer.
        T* _build_nested_memory(void** data /*!< A pointer to the pointer where data should be stored*/,
                                size_t* dims /*!< An array of integer dimension values*/,
                                size_t n_dims /*!< The number of dimension left to iterate through*/,
                                T* contiguous_mem /*!< The contiguous memory that the nested structure points to*/
                                );

        //! Set the tensor data from a src memory location
        virtual void _set_tensor_data(void* src_data /*!< A pointer to the source data being copied into the tensor*/,
                                      const std::vector<size_t>& dims /*!< The dimensions of the source data*/,
                                      const MemoryLayout mem_layout /*!< The layout of the source data memory structure*/
                                      );

        //! This function will copy a fortran array memory space (column major) to a c-style memory space layout (row major)
        void _f_to_c_memcpy(T* c_data,
                            T* f_data,
                            const std::vector<size_t>& dims);

        //! This function will copy a c-style array memory space (row major) to a fortran memory space layout (col major)
        void _c_to_f_memcpy(T* f_data,
                            T* c_data,
                            const std::vector<size_t>& dims);

        //! This is a recursive function used to copy fortran column major memory to c-style row major memory
        void _f_to_c(T* c_data,
                     T* f_data,
                     const std::vector<size_t>& dims,
                     std::vector<size_t> dim_positions,
                     size_t current_dim);

        //! This is a recursive function used to copy c-style row major memory to fortran column major memory
        void _c_to_f(T* f_data,
                     T* c_data,
                     const std::vector<size_t>& dims,
                     std::vector<size_t> dim_positions,
                     size_t current_dim);


        //! Calculate the contiguous array position for a column major position
        inline size_t _f_index(const std::vector<size_t>& dims,
                               const std::vector<size_t>& dim_positions);

        //! Calculate the contiguous array position for a column major position
        inline size_t _c_index(const std::vector<size_t>& dims,
                               const std::vector<size_t>& dim_positions);

        //! Get the total number of bytes of the Tensor data
        virtual size_t _n_data_bytes();

        //! Memory list that is used to hold recursively allocated
        //! when a data view is requested.
        MemoryList<T*> _ptr_mem_list;

        //! Memory list that is needed to retrun memory view of
        //! of tensor data in column major order because
        //! this operation is not done in place.
        MemoryList<T> _f_mem_views;
};

#include "tensor.tcc"

#endif //SMARTSIM_TENSOR_H
