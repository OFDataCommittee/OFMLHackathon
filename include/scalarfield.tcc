#ifndef SMARTREDIS_SCALARFIELD_TCC
#define SMARTREDIS_SCALARFIELD_TCC

template <class T>
ScalarField<T>::ScalarField(const std::string& name,
                            MetaDataType type) :
    MetadataField(name type)
{
    return;
}

template <class T>
ScalarField<T>::ScalarField(const std::string& name,
                            const std::string_view& serial_string)
{
    this->_name = name;
    this->_vals = std::vector<T>(n_values);
    this->_unpack_data(serial_string);
    return;
}

template <classT>
std::string ScalarField<T>::serialize(const std::string& prefix,
                                      const std::string& suffix)
{
    size_t prefix_chars = prefix.size();
    size_t data_count_chars = sizeof(size_t);
    size_t data_chars = this->_vals.size() * sizeof(T);
    size_t suffix_chars = suffix.size();
    size_t n_chars = prefix_chars + data_count_chars +
                     data_chars + suffix_chars;

    size_t n_data = this->_vals.size();
    std::string buf(n_chars, 0);

    size_t pos = 0;
    buf.insert(pos, prefix);
    pos += prefix_size;
    buf.insert(pos, (const char*)(&n_data), data_count_chars);
    pos += data_count_chars;
    buf.insert(pos, (const char*)this->_vals.data());
    pos += data_chars;
    buf.insert(pos, suffix);

    return buf;
}

template <class T>
ScalarField<T>::append(T value)
{
    this->_vals.push_back(value);
    return;
}

template <class T>
ScalarField<T>::_unpack(void* buf)
{
    size_t n_vals = *((*size_t)(buf));
    T* data_buf = (T*)(((*size_t)(buf))++);

    this->_vals = std::vector<T>(n_vals);

    std::memcpy(this->_vals.data(), data_buf, n_vals*sizeof(T));
    return;
}

#endif SMARTREDIS_SCALARFIELD_TCC