#ifndef SMARTSIM_TENSORLIST_TCC
#define SMARTSIM_TENSORLIST_TCC

template <class T>
TensorList<T>::~TensorList()
{
    /* TensorList destructor
    */
    this->_free_tensors();
}

template <class T>
TensorList<T>::TensorList(const TensorList& t_list)
{
    /* This is the copy constructor for Memorylist.
    A deep copy is performed for all Tensors
    in the TensorList t_list.
    */
    typename std::forward_list<Tensor<T>*>::const_iterator it =
        t_list._inventory.cbegin();
    typename std::forward_list<Tensor<T>*>::const_iterator it_end =
        t_list._inventory.cend();

    while(it!=it_end) {
        this->_inventory.push_front(new Tensor<T>(**it));
        it++;
    }
    this->_inventory.reverse();
}

template <class T>
TensorList<T>::TensorList(TensorList&& t_list)
{
    /* This is the move constructor for TensorList.
    */
    this->_inventory = std::move(t_list._inventory);
}

template <class T>
TensorList<T>& TensorList<T>::operator=(const TensorList<T>& t_list)
{
    /* This function is the assignment operator.
    */
    if(this!=&t_list) {
        this->_free_tensors();

        typename std::forward_list<Tensor<T>*>::const_iterator it =
        t_list._inventory.cbegin();
        typename std::forward_list<Tensor<T>*>::const_iterator it_end =
        t_list._inventory.cend();

        while(it!=it_end) {
            this->_inventory.push_front(new Tensor<T>(**it));
            it++;
        }
        this->_inventory.reverse();
    }
    return *this;
}

template <class T>
TensorList<T>& TensorList<T>::operator=(TensorList<T>&& t_list)
{
    /* This function is the move assignment operator.
    */

    if(this!=&t_list) {
        this->_free_tensors();
        this->_inventory = std::move(t_list._inventory);
    }
    return *this;
}

template <class T>
void TensorList<T>::add_tensor(Tensor<T>* tensor)
{
    /* Add a Tensor allocated outside of the TensorList
    object to the memory managed by TensorList.
    */
    this->_inventory.push_front(tensor);
    return;
}

template <class T>
void TensorList<T>::add_tensor(TensorBase* tensor) {
    /* Add a TensorBase tensor to the TensorList.
    The TensorBase pointer will be recast as a
    Tensor<T>* type before being added.
    */
    this->_inventory.push_front((Tensor<T>*)tensor);
}

template <class T>
Tensor<T>* TensorList<T>::allocate_tensor(const std::string& name,
                                    void* data,
                                    const std::vector<size_t>& dims,
                                    const TensorType type,
                                    const MemoryLayout mem_layout)
{
    /* Allocate and add a tensor to the TensorList
    */
    this->_inventory.push_front(new Tensor<T>(name, data, dims,
                                              type, mem_layout));
    return this->_inventory.front();
}

template <typename T>
void TensorList<T>::clear() {
    /* This function clears the TensorList inventory
    and frees all memory associated with the Tensors.
    */
    this->_free_tensors();
}

template <typename T>
typename TensorList<T>::iterator TensorList<T>::begin() {
    /* Returns an iterator pointing to the
    first tensor.
    */
    return this->_inventory.begin();
}

template <typename T>
typename TensorList<T>::const_iterator TensorList<T>::cbegin() {
    /* Returns a const iterator pointing to the
    first tensor.
    */
    return this->_inventory.cbegin();
}

template <typename T>
typename TensorList<T>::iterator TensorList<T>::end(){
    /* Returns an iterator pointing to the
    past-the-end tensor.
    */
    return this->_inventory.end();
}

template <typename T>
typename TensorList<T>::const_iterator TensorList<T>::cend(){
    /* Returns a const iterator pointing to the
    past-the-end tensor.
    */
    return this->_inventory.cend();
}

template <typename T>
inline void TensorList<T>::_free_tensors() {
    /* This function frees the memory associated
    with all of the Tensors in the TensorList.
    */
    while(!this->_inventory.empty()) {
        delete this->_inventory.front();
        this->_inventory.pop_front();
    }
}

#endif //SMARTSIM_TENSORLIST_TCC