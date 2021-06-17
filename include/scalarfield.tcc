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