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

#ifndef SMARTREDIS_TENSORLIST_TCC
#define SMARTREDIS_TENSORLIST_TCC

template <class T>
TensorList<T>::~TensorList()
{
    this->_free_tensors();
}

template <class T>
TensorList<T>::TensorList(const TensorList& t_list)
{
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
    this->_inventory = std::move(t_list._inventory);
}

template <class T>
TensorList<T>& TensorList<T>::operator=(const TensorList<T>& t_list)
{
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
    if(this!=&t_list) {
        this->_free_tensors();
        this->_inventory = std::move(t_list._inventory);
    }
    return *this;
}

template <class T>
void TensorList<T>::add_tensor(Tensor<T>* tensor)
{
    this->_inventory.push_front(tensor);
    return;
}

template <class T>
void TensorList<T>::add_tensor(TensorBase* tensor) {
    this->_inventory.push_front((Tensor<T>*)tensor);
}

template <class T>
Tensor<T>* TensorList<T>::allocate_tensor(const std::string& name,
                                    void* data,
                                    const std::vector<size_t>& dims,
                                    const TensorType type,
                                    const MemoryLayout mem_layout)
{
    this->_inventory.push_front(new Tensor<T>(name, data, dims,
                                              type, mem_layout));
    return this->_inventory.front();
}

template <typename T>
void TensorList<T>::clear() {
    this->_free_tensors();
}

template <typename T>
typename TensorList<T>::iterator TensorList<T>::begin() {
    return this->_inventory.begin();
}

template <typename T>
typename TensorList<T>::const_iterator TensorList<T>::cbegin() {
    return this->_inventory.cbegin();
}

template <typename T>
typename TensorList<T>::iterator TensorList<T>::end(){
    return this->_inventory.end();
}

template <typename T>
typename TensorList<T>::const_iterator TensorList<T>::cend(){
    return this->_inventory.cend();
}

template <typename T>
inline void TensorList<T>::_free_tensors() {
    while(!this->_inventory.empty()) {
        delete this->_inventory.front();
        this->_inventory.pop_front();
    }
}

#endif //SMARTREDIS_TENSORLIST_TCC