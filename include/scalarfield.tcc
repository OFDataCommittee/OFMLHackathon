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

#ifndef SMARTREDIS_SCALARFIELD_TCC
#define SMARTREDIS_SCALARFIELD_TCC

template <class T>
ScalarField<T>::ScalarField(const std::string& name,
                            MetaDataType type) :
    MetadataField(name, type)
{
    return;
}

template <class T>
ScalarField<T>::ScalarField(const std::string& name,
                            MetaDataType type,
                            const std::vector<T>& vals) :
    MetadataField(name, type)
{
    this->_vals = vals;
    return;
}

template <class T>
ScalarField<T>::ScalarField(const std::string& name,
                            MetaDataType type,
                            std::vector<T>&& vals) :
    MetadataField(name, type)
{
    this->_vals = std::move(vals);
    return;
}

template <class T>
std::string ScalarField<T>::serialize()
{
    return MetadataBuffer::generate_scalar_buf<T>(this->type(),
                                                  this->_vals);
}

template <class T>
void ScalarField<T>::append(const void* value)
{
    this->_vals.push_back(*((T*)(value)));
    return;
}

template <class T>
size_t ScalarField<T>::size()
{
    return this->_vals.size();
}

template <class T>
void ScalarField<T>::clear()
{
    this->_vals.clear();
    return;
}

template <class T>
void* ScalarField<T>::data()
{
    return this->_vals.data();
}

template <class T>
const std::vector<T>& ScalarField<T>::immutable_values()
{
    return this->_vals;
}

#endif //SMARTREDIS_SCALARFIELD_TCC