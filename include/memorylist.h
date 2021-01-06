#ifndef SMARTSIM_MEMORYLIST_H
#define SMARTSIM_MEMORYLIST_H

#include <forward_list>
#include <cstring>

namespace SILC {

typedef unsigned long long bytesize;

template <class T>
class MemoryList {

    public:

    //! MemoryList default constructor
    MemoryList();

    //! MemoryList move constructor
    MemoryList(MemoryList&& memlist);

    //! MemoryList copy constructor
    MemoryList(const MemoryList& memlist);

    //! MemoryList copy assignment operator
    MemoryList<T>& operator=(const MemoryList<T>& memlist);

    //! MemoryList move assignment operator
    MemoryList<T>& operator=(MemoryList<T>&& memlist);

    //! MemoryList destructor
    ~MemoryList();

    //! Add a malloc memory allocation performed external to MemoryList
    void add_allocation(bytesize bytes, T* ptr);

    //! Perform a malloc based on total bytes and store in the inventory
    T* allocate_bytes(bytesize bytes);

    //! Perform a malloc based on number of values and store in the inventory
    T* allocate(unsigned long long n_values);

    private:

    //! Free allocated memory in inventory and empty the inventory object
    void _free_inventory();

    //! Forward list to track allocation sizes and locations in memory
    std::forward_list<std::pair<bytesize, T*>> _inventory;

};

#include "memorylist.tcc"

} //namespace SILC

#endif //SMARTSIM_METADATA_H

