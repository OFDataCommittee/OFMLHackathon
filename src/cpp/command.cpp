#include "command.h"

using namespace SILC;

Command::~Command()
{
    std::vector<char*>::iterator it = this->_local_fields.begin();
    std::vector<char*>::iterator it_end = this->_local_fields.end();
    for(; it!=it_end; it++) {
        free(*it);
        (*it) = 0;
    }
    return;
}

void Command::add_field(std::string field, bool is_key)
{
    /* Copy the field string into a char* that is stored
    locally in the Command object.  The new string is not
    null terminated because the fields vector is of type
    string_view which stores the length of the string.
    If is_key is true, the key will be added to the command
    keys.
    */

    int field_size = field.size();
    char* f = (char*) malloc(sizeof(char)*(field_size+1));
    field.copy(f, field_size, 0);
    f[field_size]=0;
    this->_local_fields.push_back(f);
    this->_fields.push_back(std::string_view(f, field_size));

    if(is_key)
        this->_cmd_keys[std::string_view(f, field_size)] =
            this->_fields.size()-1;

    return;
}

void Command::add_field(char* field, bool is_key)
{
    /* Copy the field char* into a char* that is stored
    locally in the Command object.  The new string is not
    null terminated because the fields vector is of type
    string_view which stores the length of the string.
    If is_key is true, the key will be added to the command
    keys.
    */

    int field_size = std::strlen(field);
    char* f = (char*) malloc(sizeof(char)*(field_size));
    std::memcpy(f, field, sizeof(char)*field_size);
    this->_local_fields.push_back(f);
    this->_fields.push_back(std::string_view(f, field_size));

    if(is_key)
        this->_cmd_keys[std::string_view(f, field_size)] =
            this->_fields.size()-1;

    return;
}

void Command::add_field_ptr(char* field, size_t field_size)
{
    /* This function adds a field to the fields data
    structure without copying the data.  This means
    that the memory needs to be valid when it is later
    accessed.  This function should be used for very large
    fields.  Field pointers cannot act as Command keys.
    */
    this->_fields.push_back(std::string_view(field, field_size));
    return;
}

void Command::add_field_ptr(std::string_view field)
{
    /* This function adds a field to the fields data
    structure without copying the data.  This means
    that the memory needs to be valid when it is later
    accessed.  This function should be used for very large
    fields.  If is_key is true, the key will be added to the
    command keys.  Field pointers cannot act as Command keys.
    */
    this->_fields.push_back(field);
    return;
}

void Command::add_fields(const std::vector<std::string>& fields, bool is_key)
{
    /* Copy field strings into a char* that is stored
    locally in the Command object.  The new string is not
    null terminated because the fields vector is of type
    string_view which stores the length of the string
    */
    for(int i=0; i<fields.size(); i++)
        this->add_field(fields[i], is_key);
    return;
}

std::string Command::first_field()
{
    return std::string(this->begin()->data(),
                       this->begin()->size());
}

std::string Command::to_string()
{
    Command::iterator it = this->begin();
    Command::iterator it_end = this->end();

    std::string output;
    while(it!=it_end) {
        output += " " + std::string(it->data(), it->size());
        it++;
    }
    return output;
}

Command::iterator Command::begin()
{
    return this->_fields.begin();
}

Command::const_iterator Command::cbegin()
{
    return this->_fields.cbegin();
}

Command::iterator Command::end()
{
    return this->_fields.end();
}

Command::const_iterator Command::cend()
{
    return this->_fields.cend();
}

bool Command::has_keys()
{
    return (this->_cmd_keys.size()>0);
}

std::vector<std::string> Command::get_keys() {
    /* This function returns a vector of key copies
    to the user.  Original keys are not returned
    to the user because memory management
    becomes complicated for std::string_view that
    may need to grow or decrease in size.
    */
    std::vector<std::string> keys;

    std::unordered_map<std::string_view,size_t>::iterator it =
        this->_cmd_keys.begin();
    std::unordered_map<std::string_view,size_t>::iterator it_end;
        this->_cmd_keys.end();

    while(it!=it_end) {
        keys.push_back(std::string(it->first.data(), it->first.length()));
        it++;
    }
    return keys;

}

void Command::update_key(std::string old_key,
                         std::string new_key)
{
    /* This function will change a value of a
    key in the command.  This function will preserve
    the order of the command fields.
    */
    std::string_view old_key_sv(old_key.data(), old_key.length());
    size_t key_index = this->_cmd_keys.at(old_key_sv);

    //Allocated memory and copy new_key
    int new_key_size = new_key.size();
    char* f = (char*) malloc(sizeof(char)*(new_key_size+1));
    new_key.copy(f, new_key_size, 0);
    f[new_key_size]=0;
    std::string_view new_key_sv(f, new_key_size);

    //Swap string_view, dellaocate old memory and
    //track new memory
    this->_fields[key_index].swap(new_key_sv);
    free(this->_local_fields[key_index]);
    this->_local_fields[key_index] = f;
    return;
}