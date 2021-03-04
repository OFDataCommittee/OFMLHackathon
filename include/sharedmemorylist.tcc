#ifndef SILC_SHAREDMEMORYLIST_TCC
#define SILC_SHAREDMEMORYLIST_TCC

template <class T>
void SharedMemoryList<T>::add_allocation(size_t bytes, T* ptr)
{
    std::shared_ptr<T> s_ptr(ptr, free);
    this->_inventory.push_front(s_ptr);
    return;
}

template <class T>
T* SharedMemoryList<T>::allocate_bytes(size_t bytes)
{
    T* ptr = (T*)malloc(bytes);
    this->add_allocation(bytes, ptr);
    return ptr;
}

template <class T>
T* SharedMemoryList<T>::allocate(size_t n_values)
{
    size_t bytes = n_values * sizeof(T);
    return this->allocate_bytes(bytes);
}

#endif //SILC_SHAREDMEMORYLIST_TCC