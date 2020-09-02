#ifndef SMARTSIM_TENSOR_TCC
#define SMARTSIM_TENSOR_TCC

template <class T>
Tensor<T>::Tensor(const std::string& name,
                  const std::string& type,
                  void* data,
                  const std::vector<int>& dims) :
                  TensorBase(name, type, data, dims)
{}

template <class T>
Tensor<T>::Tensor(const std::string& name,
                  const std::string& type,
                  const std::vector<int>& dims,
                  const std::string_view& data_buf) :
                  TensorBase(name, type, dims, data_buf)
{}

template <class T>
Tensor<T>::~Tensor()
{}

template <class T>
Tensor<T>::Tensor(const Tensor<T>& tensor) : TensorBase(tensor)
{}

template <class T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& tensor)
{
    TensorBase::operator=(tensor);
}

template <class T>
void* Tensor<T>::get_data()
{
    /* This function returns a pointer to the memory space
    of the tensor data.  If the pointer does not exist and
    a buffer does exist, memory will be allocated and the
    buffer coppied into the memory space.
    */
    if(this->_data)
        return this->_data;
    else {
        this->_allocate_data_memory(&(this->_data),
                                    this->_dims.data(),
                                    this->_dims.size());
        int start_position = 0;
        this->_buf_to_data(this->_data, this->_dims.data(),
                           this->_dims.size(), start_position,
                           this->_data_buf);
        return this->_data;
    }
}

template <class T>
void Tensor<T>::fill_data_from_buf(void* data, std::vector<int> dims,
                                   const std::string& type)
{
    /* This function will fill a supplied memory space
    (data) with the values in the tensor buffer.
    */
    if(!this->_data_buf)
        throw std::runtime_error("The tensor does not have "\
                                 "a databuf to fill with.");

    if(dims.size() == 0)
        throw std::runtime_error("The dimensions must have "\
                                 "size greater than 0.");

    int n_values = 1;
    for(int i=0; i<dims.size(); i++) {
        if(dims[i]<=0)
            throw std::runtime_error("All dimensions must "\
                                     "be greater than 0.");
        n_values*=dims[i];
    }

    int buf_vals = this->_buf_size / sizeof(T);
    if(n_values!=buf_vals)
        throw std::runtime_error("The provided dimensions do "\
                                  "not match the size of the "\
                                  "buffer");

    int buf_starting_position = 0;
    this->_buf_to_data(data, dims.data(),
                       dims.size(),
                       buf_starting_position,
                       this->_data_buf);
    return;
}

template <class T>
void Tensor<T>::_generate_data_buf()
{
    /* This function will turn the tensor data into a binary string
    buffer that can be used for data transfer.
    */

    /*TODO there is an optimization possible that if the data is
    contiguous in memory (e.g. special fortran or numpy arrays)
    or if it is 1D data (all languages) We can have data_buf
    just point to data and there will be absolutely no copies.
    until we get into redis itself.
    */
    int n_bytes = 1;
    for (int i = 0; i < this->_dims.size(); i++) {
        n_bytes *= this->_dims[i];
    }
    n_bytes *= sizeof(T);

    this->_buf_size = n_bytes;
    this->_data_buf = (char*)malloc(n_bytes);

    /*TODO Now that this is in a Tensor class, we might
    be able to remove some of the arguments being passed to
    clean it up, but may not be able to because of recursive
    nature
    */
    this->_vals_to_buf(this->_data, &(this->_dims[0]),
                       this->_dims.size(), (void*)this->_data_buf);
    return;
}

template <class T>
void* Tensor<T>::_vals_to_buf(void* data, int* dims, int n_dims,
                                       void* buf)
{
    /* This function will copy the tensor data values into the
    binary buffer
    */

    //TODO we should check at some point that we don't exceed buf length
    if(n_dims > 1) {
        T** current = (T**) data;
        for(int i = 0; i < dims[0]; i++) {
          buf = this->_vals_to_buf(*current, &dims[1], n_dims-1, buf);
          current++;
        }
    }
    else {
        std::memcpy(buf, data, sizeof(T)*dims[0]);
        return &((T*)buf)[dims[0]];
    }
    return buf;
}

template <class T>
void Tensor<T>::_buf_to_data(void* data, int* dims, int n_dims,
                             int& buf_position, void* buf)
{
    /* This recursive function copies data from the buf into
    a formatted data array
    */
    if(n_dims > 1) {
        T** current = (T**) data;
        for(int i = 0; i < dims[0]; i++) {
            this->_buf_to_data(*current, &dims[1], n_dims-1,
                               buf_position, buf);
            current++;
        }
    }
    else {
        T* buf_to_copy = &(((T*)buf)[buf_position]);
	std::memcpy(data, buf_to_copy, dims[0]*sizeof(T));
        buf_position += dims[0];
    }
    return;
}

template <class T>
void Tensor<T>::_allocate_data_memory(void** data, int* dims, int n_dims)
{
    /* This function recursively allocates a tensor data pointer
    */
    if(n_dims>1) {
        T** new_data = this->_ptr_mem_list.allocate(dims[0]);
        (*data) = (void*)new_data;
        for(int i=0; i<dims[0]; i++)
            this->_allocate_data_memory((void**)&(new_data[i]),
                                        &dims[1], n_dims-1);
    }
    else
    {
        T* new_data = this->_numeric_mem_list.allocate(dims[0]);
        (*data) = (void*)new_data;
    }

    return;
}

#endif //SMARTSIM_TENSOR_TCC