/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Hewlett Packard Enterprise
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SMARTREDIS_SHAREDMEMORYLIST_H
#define SMARTREDIS_SHAREDMEMORYLIST_H

#include <forward_list>
#include <cstring>
#include <memory>

namespace SmartRedis {

/*!
*   \brief  The SharedMemoryList class allocates
*           and manages memory of type T.
*   \details This class is useful for repeated
*            allocations of ype T that need to be
*            managed.  The SharedMemoryList class uses
*            shared pointers which means that memory will
*            not be freed until all copies of a
*            SharedMemoryList are destroyed.
*   \tparam T The data type for allocation
*/
template <class T>
class SharedMemoryList {

    public:

    /*!
    *   \brief SharedMemoryList default constructor
    */
    SharedMemoryList() = default;

    /*!
    *   \brief SharedMemoryList default copy constructor
    *   \param memlist SharedMemoryList to copy for construction
    */
    SharedMemoryList(const SharedMemoryList<T>& memlist) = default;

    /*!
    *   \brief SharedMemoryList default move constructor
    *   \param memlist SharedMemoryList to move for construction
    */
    SharedMemoryList(SharedMemoryList<T>&& memlist) = default;

    /*!
    *   \brief SharedMemoryList default copy assignment operator
    *   \param memlist SharedMemoryList to copy for assignment
    *   \returns The SharedMemoryList that has been assigned values
    */
    SharedMemoryList<T>& operator=(const SharedMemoryList<T>& memlist) = default;

    /*!
    *   \brief SharedMemoryList default move assignment operator
    *   \param memlist SharedMemoryList to move for assignment
    *   \returns The SharedMemoryList that has been assigned values
    */
    SharedMemoryList<T>& operator=(SharedMemoryList<T>&& memlist) = default;

    /*!
    *   \brief SharedMemoryList default destructor
    */
    ~SharedMemoryList() = default;

    /*!
    *   \brief Add a malloc memory allocation performed
    *          external to SharedMemoryList
    *   \param bytes The number of bytes in the allocation
    *   \param ptr A pointer to the memory allocation
    */
    void add_allocation(size_t bytes, T* ptr);

    /*!
    *   \brief  Perform a malloc based on total
    *           bytes and store in the inventory
    *   \param bytes The number of bytes to allocate
    *   \returns A pointer to the memory allocation
    */
    //!
    T* allocate_bytes(size_t bytes);

    /*!
    *   \brief  Perform a malloc based on number of
    *           values and store in the inventory
    *   \param bytes The number of values to allocate
    *   \returns A pointer to the memory allocation
    */
    T* allocate(size_t n_values);

    private:
    /*!
    *   \brief  Forward list to track allocation
    *           sizes and locations in memory
    */
    typename std::forward_list<std::shared_ptr<T>> _inventory;

};

#include "sharedmemorylist.tcc"

} //namespace SmartRedis

#endif //SMARTREDIS_SHAREDMEMORYLIST_H

