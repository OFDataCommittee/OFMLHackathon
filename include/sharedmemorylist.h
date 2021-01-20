#ifndef SMARTSIM_SHAREDMEMORYLIST_H
#define SMARTSIM_SHAREDMEMORYLIST_H

#include <forward_list>
#include <cstring>
#include <memory>

namespace SILC {

template <class T>
class SharedMemoryList {

    public:

    //! SharedMemoryList default constructor
    SharedMemoryList() = default;

    //! SharedMemoryList copy constructor
    SharedMemoryList(const SharedMemoryList<T>& memlist) = default;

    //! SharedMemoryList move constructor
    SharedMemoryList(SharedMemoryList<T>&& memlist) = default;

    //! SharedMemoryList copy assignment operator
    SharedMemoryList<T>& operator=(const SharedMemoryList<T>& memlist) = default;

    //! SharedMemoryList move assignment operator
    SharedMemoryList<T>& operator=(SharedMemoryList<T>&& memlist) = default;

    //! SharedMemoryList destructor
    ~SharedMemoryList() = default;

    //! Add a malloc memory allocation performed external to SharedMemoryList
    void add_allocation(size_t bytes, T* ptr);

    //! Perform a malloc based on total bytes and store in the inventory
    T* allocate_bytes(size_t bytes);

    //! Perform a malloc based on number of values and store in the inventory
    T* allocate(size_t n_values);

    private:

    //! Forward list to track allocation sizes and locations in memory
    typename std::forward_list<std::shared_ptr<T>> _inventory;

};

#include "sharedmemorylist.tcc"

} //namespace SILC

#endif //SMARTSIM_SHAREDMEMORYLIST_H

