#ifndef SMARTSIM_MEMORYLIST_TCC
#define SMARTSIM_MEMORYLIST_TCC

template <class T>
MemoryList<T>::MemoryList()
{
}

template <class T>
MemoryList<T>::~MemoryList()
{
    this->_free_inventory();
}

template <class T>
MemoryList<T>::MemoryList(const MemoryList& memlist)
{
    /* This is the copy constructor for Memorylist.  New memory
    is allocated, and the values in the memlist are copied to the
    new MemoryList
    */
    typename std::forward_list<std::pair<bytesize, T*>>::const_iterator
        it = memlist._inventory.begin();
    typename std::forward_list<std::pair<bytesize, T*>>::const_iterator
        it_end = memlist._inventory.end();

    while(it!=it_end) {
        bytesize n_bytes = it->first;
        T* ptr = this->allocate_bytes(n_bytes);
        std::memcpy(ptr, it->second, n_bytes);
        it++;
    }
}

template <class T>
MemoryList<T>::MemoryList(MemoryList&& memlist)
{
    /* This is the move constructor for MemoryList.  The memory
    is items are moved and pointers in the original memory
    list are set to 0 to avoid the memory being freed.
    */
    this->_inventory = std::move(memlist->_inventory);
    memlist->_inventory.clear();
    return;
}

template <class T>
MemoryList<T>& MemoryList<T>::operator=(const MemoryList<T>& memlist)
{
    /* This function is the assignment operator.  First the internal
    inventory is cleared, then memory is allocated and data is copied.
    */
    this->_free_inventory();

    typename std::forward_list<std::pair<bytesize, T*>>::const_iterator
        it = memlist._inventory.begin();
    typename std::forward_list<std::pair<bytesize, T*>>::const_iterator
        it_end = memlist._inventory.end();
    while(it!=it_end) {
        bytesize n_bytes = it->first;
        T* ptr = this->allocate_bytes(n_bytes);
        std::memcpy(ptr, it->second, n_bytes);
        it++;
    }
    return *this;
}

template <class T>
MemoryList<T>& MemoryList<T>::operator=(MemoryList<T>&& memlist)
{
    /* This function is the move assignment operator.
    The internal inventory of memory allcations is moved,
    and then the original inventory is cleared to prevent
    the memory being freed.
    */
    if(this!=&memlist) {
        this->_inventory = std::move(memlist._inventory);
        memlist._inventory.clear();
    }
    return *this;
}

template <class T>
void MemoryList<T>::add_allocation(bytesize bytes, T* ptr)
{
    /* Add an allocation made outside of the MemoryList object
    to the memory managed by MemoryList.
    */
    this->_inventory.push_front({bytes, ptr});
    return;
}

template <class T>
T* MemoryList<T>::allocate_bytes(bytesize bytes)
{
    /* Create an allocation with the specified bytes
    add at to the MemoryList inventory.
    */
    T* ptr = (T*)malloc(bytes);
    this->add_allocation(bytes, ptr);
    return ptr;
}

template <class T>
T* MemoryList<T>::allocate(unsigned long long n_values)
{
    /* Allocate memory for n_values of type T,
    and add that allocation to the MemoryList inventory.
    */
    bytesize bytes = n_values * sizeof(T);
    return (this->allocate_bytes(bytes));
}

template <class T>
void MemoryList<T>::_free_inventory() {
    /* Free all of the allocated memory and remove
    from the inventory.
    */
    while(!(this->_inventory.empty()))
    {
        std::pair<bytesize, T*>& alloc = this->_inventory.front();
        free(alloc.second);
        this->_inventory.pop_front();
    }
}

#endif //SMARTSIM_METADATA_TCC