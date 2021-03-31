#ifndef SMARTREDIS_COMMAND_TCC
#define SMARTREDIS_COMMAND_TCC

template <class T>
void Command::add_fields(const std::vector<T>& fields, bool is_key)
{
    for(int i=0; i<fields.size(); i++) {
        this->add_field(std::to_string(fields[i]), is_key);
    }
    return;
}

#endif //SMARTREDIS_COMMAND_TCC