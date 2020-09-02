#ifndef SMARTSIM_COMMAND_TCC
#define SMARTSIM_COMMAND_TCC

template <class T>
void Command::add_fields(const std::vector<T>& fields)
{
    for(int i=0; i<fields.size(); i++) {
        this->add_field(std::to_string(fields[i]));
    }
    return;
}

#endif //SMARTSIM_COMMAND_TCC