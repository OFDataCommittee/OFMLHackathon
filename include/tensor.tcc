#ifndef SMARTSIM_TENSOR_TCC
#define SMARTSIM_TENSOR_TCC
#include <stdexcept>

template <class T>
Tensor<T>::Tensor(const std::string& name,
                  void* data,
                  const std::vector<size_t>& dims,
                  const TensorType type,
                  const MemoryLayout mem_layout) :
                  TensorBase(name, data, dims, type, mem_layout)
{
    /* Constructor for the Tensor class.
    */
    this->_set_tensor_data(data, dims, mem_layout);
}

template <class T>
Tensor<T>::~Tensor()
{
    /* Destructor for the Tensor class.
    */
    if(this->_data)
        free(this->_data);
}

template <class T>
Tensor<T>::Tensor(const Tensor<T>& tensor) : TensorBase(tensor)
{
    /* Copy constructor for Tensor.  The data
    and allocated pointers are copied.
    */
    this->_set_tensor_data(tensor._data, tensor._dims,
                           MemoryLayout::contiguous);
    this->_ptr_mem_list = tensor._ptr_mem_list;
}

template <class T>
Tensor<T>::Tensor(Tensor<T>&& tensor) : TensorBase(tensor)
{
    /* Move constructor for Tensor.  The data
    and allocated ptrs are moved and old data
    is left in a safe, but empty state.
    */

    this->_data = tensor._data;
    tensor._data = 0;
    this->_ptr_mem_list = std::move(tensor._ptr_mem_list);
}

template <class T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& tensor)
{
    /* Copy assignment operator for Tensor.  The data
    and allocated ptrs are copied.
    */
    this->TensorBase::operator=(tensor);
    this->_set_tensor_data(tensor._data, tensor._dims,
                           MemoryLayout::contiguous);
    this->_ptr_mem_list = tensor._ptr_mem_list;
    return *this;
}

template <class T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& tensor)
{
    /* Move assignment operator for Tensor.  The data
    and allocated ptrs are moved and old data
    is left in a safe, but empty state.
    */
    if(this!=&tensor) {
        this->_data = tensor._data;
        tensor._data = 0;
        this->_ptr_mem_list = std::move(tensor._ptr_mem_list);
    }
    return *this;
}

template <class T>
void* Tensor<T>::data_view(const MemoryLayout mem_layout)
{
    /* This function returns a pointer to a memory
    view of the underlying tensor data.  The
    view of the underlying tensor data depends
    on the mem_layout value.  The following values
    result in the following views:
    1) MemoryLayout::contiguous : The returned view
       pointer points to the first position in
       the tensor data array.  The caller is
       expected to only index into the view
       pointer with a single []
       MemoryLayout::nested : The returned view
       pointer points to a nested structure of
       pointers so that the caller can cast
       to a nested pointer structure and index
       with multiple [] operators.
    */

    void* ptr = 0;

    switch(mem_layout) {
        case(MemoryLayout::contiguous) :
            ptr = this->_data;
            break;
        case(MemoryLayout::nested) :
            this->_build_nested_memory(&ptr,
                                       this->_dims.data(),
                                       this->_dims.size(),
                                       (T*)this->_data);
            break;
        default :
            throw std::runtime_error(
                "Unsupported MemoryLayout "\
                "value in "\
                "Tensor<T>.data_view().");
    }
    return ptr;
}

template <class T>
void Tensor<T>::fill_mem_space(void* data,
                               std::vector<size_t> dims)
{
    /* This function will fill a supplied memory space
    (data) with the values in the tensor data array.
    */
    if(!this->_data)
        throw std::runtime_error("The tensor does not have "\
                                 "a data array to fill with.");

    if(dims.size() == 0)
        throw std::runtime_error("The dimensions must have "\
                                 "size greater than 0.");

    size_t n_values = 1;
    std::vector<size_t>::const_iterator it = dims.cbegin();
    std::vector<size_t>::const_iterator it_end = dims.cend();
    while(it!=it_end) {
        if((*it)<=0)
            throw std::runtime_error("All dimensions must "\
                                     "be greater than 0.");
        n_values*=(*it);
        it++;
    }

    if(n_values!=this->num_values())
        throw std::runtime_error("The provided dimensions do "\
                                  "not match the size of the "\
                                  "tensor data array");

    size_t starting_position = 0;

    this->_fill_nested_mem_with_data(data, dims.data(),
                                     dims.size(),
                                     starting_position,
                                     this->_data);
    return;
}

template <class T>
void* Tensor<T>::_copy_nested_to_contiguous(void* src_data,
                                            const size_t* dims,
                                            const size_t n_dims,
                                            void* dest_data)
{
    /* This function will copy the src_data, which is in a nested
    memory structure, to the dest_data memory space which is flat
    and contiguous.  The value returned by the first execution
    of this function will change the copy of dest_data and return
    a value that is not equal to the original source data value.
    As a result, the initial call of this function SHOULD NOT
    use the returned value.
    */

    if(n_dims > 1) {
        T** current = (T**)src_data;
        for(size_t i = 0; i < dims[0]; i++) {
          dest_data =
            this->_copy_nested_to_contiguous(*current, &dims[1],
                                             n_dims-1, dest_data);
          current++;
        }
    }
    else {
        std::memcpy(dest_data, src_data, sizeof(T)*dims[0]);
        return &((T*)dest_data)[dims[0]];
    }
    return dest_data;
}

template <class T>
void Tensor<T>::_fill_nested_mem_with_data(void* data,
                                           size_t* dims,
                                           size_t n_dims,
                                           size_t& data_position,
                                           void* tensor_data)
{
    /* This recursive function copies the tensor_data
    into the nested data memory space.  The caller
    should provide an initial value of 0 for data_position.
    */
    if(n_dims > 1) {
        T** current = (T**) data;
        for(size_t i = 0; i < dims[0]; i++) {
            this->_fill_nested_mem_with_data(
                    *current, &dims[1], n_dims-1,
                    data_position, tensor_data);
            current++;
        }
    }
    else {
        T* data_to_copy = &(((T*)tensor_data)[data_position]);
	    std::memcpy(data, data_to_copy, dims[0]*sizeof(T));
        data_position += dims[0];
    }
    return;
}

template <class T>
T* Tensor<T>::_build_nested_memory(void** data,
                                   size_t* dims,
                                   size_t n_dims,
                                   T* contiguous_mem)
{
    /* This function creates a nested tensor data
    structure that points to the underlying contiguous
    memory allocation of the data.  The initial caller
    SHOULD NOT use the return value.  The return value
    is for recursive value passing only.
    */
    if(n_dims>1) {
        T** new_data = this->_ptr_mem_list.allocate(dims[0]);
        (*data) = (void*)new_data;
        for(size_t i=0; i<dims[0]; i++)
            contiguous_mem =
                this->_build_nested_memory((void**)(&new_data[i]),
                                           &dims[1], n_dims-1,
                                           contiguous_mem);
    }
    else {
        (*data) = (T*)contiguous_mem;
        contiguous_mem += dims[0];
    }
    return contiguous_mem;
}

//! Set the tensor data from a src memory location
template <class T>
void Tensor<T>::_set_tensor_data(void* src_data,
                                 const std::vector<size_t>& dims,
                                 const MemoryLayout mem_layout)
{
    /* Set the tensor data from the src_data.  This involves
    a memcpy to a contiguous array.
    */

    if(this->_data)
        free(this->_data);

    size_t n_values = this->num_values();
    size_t n_bytes = n_values * sizeof(T);
    this->_data = malloc(n_bytes);

    switch(mem_layout) {
        case(MemoryLayout::contiguous) :
            std::memcpy(this->_data, src_data, n_bytes);
            break;
        case(MemoryLayout::nested) :
            this->_copy_nested_to_contiguous(src_data,
                                             dims.data(),
                                             dims.size(),
                                             this->_data);
            break;
    }
    return;
}

template <class T>
size_t Tensor<T>::_n_data_bytes()
{
    /* This function returns the total number
    of bytes in memory occupied by this->_data.
    */
    return this->num_values()*sizeof(T);
}
#endif //SMARTSIM_TENSOR_TCC