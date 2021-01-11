#ifndef SMARTSIM_TENSORLIST_H
#define SMARTSIM_TENSORLIST_H

#include <forward_list>
#include "tensor.h"
#include <iostream>

namespace SILC {

template <class T>
class TensorList {

    public:

    //! TensorList default constructor
    TensorList() = default;

    //! TensorList move constructor
    TensorList(TensorList&& t_list);

    //! TensorList copy constructor
    TensorList(const TensorList& t_list);

    //! TensorList copy assignment operator
    TensorList<T>& operator=(const TensorList<T>& t_list);

    //! TensorList move assignment operator
    TensorList<T>& operator=(TensorList<T>&& t_list);

    //! TensorList destructor
    ~TensorList();

    //! Add a previously allocated Tensor
    void add_tensor(Tensor<T>* tensor);

    //! Add a previously allocated Tensor
    void add_tensor(TensorBase* tensor);

    //! Allocate Tensor
    Tensor<T>* allocate_tensor(const std::string& name,
                               void* data,
                               const std::vector<size_t>& dims,
                               const TensorType type,
                               const MemoryLayout mem_layout);

    //! Delete all of the Tensors in the TensorList, freeing all memory
    void clear();

    //! Iterators for tensors
    typedef typename std::forward_list<Tensor<T>*>::iterator iterator;
    typedef typename std::forward_list<Tensor<T>*>::const_iterator const_iterator;

    //! Returns an iterator pointing to the first tensor pointer
    iterator begin();

    //! Returns a const iterator pointing to the first tensor pointer
    const_iterator cbegin();

    //! Returns an iterator pointing to the past-the-end tensor pointer
    iterator end();

    //! Returns a const iterator pointing to the past-the-end tensor pointer
    const_iterator cend();

    private:

    //! Inventory of allocated Tensor objects
    std::forward_list<Tensor<T>*> _inventory;

    //! Free the memory associated with all Tensors in the TensorList
    inline void _free_tensors();

};

#include "tensorlist.tcc"

} //namespace SILC

#endif //SMARTSIM_MEMEORYLIST_H

