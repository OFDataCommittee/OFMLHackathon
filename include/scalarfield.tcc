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
                            const std::string_view& serial_string) :
    MetadataField(name, serial_string)
{
    this->_unpack(serial_string);
    return;
}

template <class T>
std::string ScalarField<T>::serialize()
{
    /*
    *   The serialized string has the format described
    *   below.
    *
    *   Field           Field size
    *   -------         ----------
    *   type_id         1 byte
    *
    *   n_values        sizeof(size_t)
    *
    *   data content     sizeof(T) * values.size()
    */

    if(sizeof(int8_t) != sizeof(char))
        throw std::runtime_error("Metadata is not supported on "\
                                 "systems with char length not "\
                                 "equal to one byte.");

    // Number of bytes needed for the type identifier
    size_t type_bytes = sizeof(int8_t);
    // Number of bytes needed for value count
    size_t count_bytes = sizeof(size_t);
    // Number of bytes needed for the string values
    size_t data_bytes = sizeof(T) * this->_vals.size();

    size_t n_bytes = type_bytes + count_bytes +
                     data_bytes;

    size_t n_chars = n_bytes / sizeof(char);

    std::string buf(n_chars, 0);
    //std::cout<<"buf.size at beg "<<buf.size()<<std::endl;
    size_t pos = 0;

    // Add the type ID
    int8_t type_id = (int8_t)this->_type;
    n_chars = sizeof(int8_t)/sizeof(char);
    this->_place_buf_chars(buf, pos, (char*)(&type_id), n_chars);
    pos += n_chars;

    // Add the number of values
    size_t n_vals = this->_vals.size();
    n_chars = sizeof(size_t)/sizeof(char);
    this->_place_buf_chars(buf, pos, (char*)(&n_vals), n_chars);
    pos += n_chars;

    // Add the number of the values
    n_chars = sizeof(T)/sizeof(char) * this->_vals.size();
    T* v_data = this->_vals.data();
    this->_place_buf_chars(buf, pos, (char*)(v_data), n_chars);
    //std::cout<<"buf.size at end "<<buf.size()<<std::endl;
    return buf;
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
void ScalarField<T>::_unpack(const std::string_view& buf)
{
    void* data = (void*)(buf.data());

    int8_t type = *((int8_t*)data);
    data = ((int8_t*)data) + 1;
    if(type!=(int8_t)this->type())
        throw std::runtime_error("The buffer scalar metadata type "\
                                 "does not match the object type "\
                                 "being used to interpret it.");

    size_t n_vals = *((size_t*)(data));
    data = ((size_t*)data) + 1;

    this->_vals = std::vector<T>(n_vals);

    std::memcpy(this->_vals.data(), (T*)data,
                n_vals*sizeof(T));
    return;
}

template <typename T>
void ScalarField<T>::_place_buf_chars(std::string& buf,
                                      size_t pos,
                                      char* buf_chars,
                                      size_t n_chars)
{
    for(size_t i=0; i<n_chars; i++) {
        //std::cout<<"Putting char "<<(*buf_chars)<<std::endl;
        buf[pos] = *buf_chars;
        pos++;
        buf_chars++;
    }
    return;
}

#endif SMARTREDIS_SCALARFIELD_TCC