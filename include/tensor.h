#ifndef SMARTSIM_TENSOR_H
#define SMARTSIM_TENSOR_H

#include "stdlib.h"
#include <string>
#include "tensorbase.h"
#include "memorylist.h"

///@file
///\brief The Tensor class for data and buffer tensor operations

template <class T>
class Tensor : public TensorBase
{
    public:

        //! Tensor constructor using tensor data pointer
        Tensor(const std::string& name /*!< The name used to reference the tensor*/,
               const std::string& type /*!< The data type of the tensor*/,
               void* data /*!< A c_ptr to the data of the tensor*/,
               const std::vector<int>& dims /*! The dimensions of the data*/
        );

        //! Tensor constructor using data bufffer without tensor data pointer
        Tensor(const std::string& name /*!< The name used to reference the tensor*/,
               const std::string& type /*!< The data type of the tensor*/,
               const std::vector<int>& dims /*! The dimensions of the data*/,
               const std::string_view& data_buf /*! The data buffer*/
        );

        //! Tensor destructor
        virtual ~Tensor();

        //! Tensor copy constructor
        Tensor(const Tensor<T>& tensor);

        //! Tensor copy assignment operator
        Tensor<T>& operator=(const Tensor<T>& tensor);

        //! Return a pointer to the tensor memory space
        virtual void* get_data();

        //! Fill a user provided memory space with values from tensor buffer
        virtual void fill_data_from_buf(void* data /*!< Pointer to the allocated memory space*/,
                                        std::vector<int> dims /*!< The dimensions of the memory space*/,
                                        const std::string& type /*!< The datatype of the allocated memory space*/
                                        );

    protected:

        //! Function to generate the data buffer from the data
        virtual void _generate_data_buf();

    private:

        //! Function to copy values from tensor into buffer
        void* _vals_to_buf(void* data /*!< The data to copy to buf tensor*/,
                           int* dims /*!< The dimensions of data*/,
                           int n_dims /*!< The number of dimensions in data*/,
                           void* buf /*!< The buffer to hold the data*/
                           );

        //! Function to copy values from buffer into tensor
        void _buf_to_data(void* data /*!< The data array to copy into*/,
                          int* dims /*!< The dimensions of data*/,
                          int n_dims /*!< The number of dimensions in data*/,
                          int& buf_position /*!< The current position of reading the buf*/,
                          void* buf /*!< The buf to read from*/
                          );

        //! Allocate memory that fits the tensor data dimensions
        void _allocate_data_memory(void** data /*!< A pointer to the pointer where data should be stored*/,
                                   int* dims /*!< An array of integer dimension values*/,
                                   int n_dims /*!< The number of dimension left to iterate through*/
                                   );

        //! Memory lists that are used to hold recursively allocated
        //! memory when a data pointer is requested and it does not
        //! exist.
        MemoryList<T> _numeric_mem_list;
        MemoryList<T*> _ptr_mem_list;
};

#include "tensor.tcc"

#endif //SMARTSIM_TENSOR_H
