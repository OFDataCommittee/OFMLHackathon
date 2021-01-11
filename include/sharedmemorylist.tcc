#ifndef SMARTSIM_SHAREMEMORYLIST_TCC
#define SMARTSIM_SHAREMEMORYLIST_TCC

template <class T>
void SharedMemoryList<T>::add_allocation(size_t bytes, T* ptr)
{
    /* Add an allocation made outside of the SharedMemoryList object
    to the memory managed by SharedMemoryList.
    */
    std::shared_ptr<T> s_ptr(ptr, free);
    this->_inventory.push_front(s_ptr);
    return;
}

template <class T>
T* SharedMemoryList<T>::allocate_bytes(size_t bytes)
{
    /* Create an allocation with the specified bytes
    add at to the SharedMemoryList inventory.
    */
    T* ptr = (T*)malloc(bytes);
    this->add_allocation(bytes, ptr);
    return ptr;
}

template <class T>
T* SharedMemoryList<T>::allocate(size_t n_values)
{
    /* Allocate memory for n_values of type T,
    and add that allocation to the SharedMemoryList
    inventory.
    */
    size_t bytes = n_values * sizeof(T);
    return this->allocate_bytes(bytes);
}

#endif //SMARTSIM_SHAREDMEMORYLIST_TCC