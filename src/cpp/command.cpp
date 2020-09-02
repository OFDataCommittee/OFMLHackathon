#include "command.h"

Command::Command()
{}

Command::~Command()
{
    std::forward_list<char*>::iterator it = this->_local_fields.begin();
    while(it != this->_local_fields.end()) {
        delete[] (*it);
        (*it) = 0;
        it++;
    }
    return;
}

void Command::add_field(std::string field)
{
    /* Copy the field string into a char* that is stored
    locally in the Command object.  The new string is not
    null terminated because the fields vector is of type
    string_view which stores the length of the string
    */

    int field_size = field.size();
    char* f = (char*) malloc(sizeof(char)*(field_size));
    field.copy(f, field_size, 0);
    this->_local_fields.push_front(f);
    this->_fields.push_back(std::string_view(f, field_size));
    return;
}

void Command::add_field(char* field)
{
    /* Copy the field char* into a char* that is stored
    locally in the Command object.  The new string is not
    null terminated because the fields vector is of type
    string_view which stores the length of the string
    */

    int field_size = std::strlen(field);
    char* f = (char*) malloc(sizeof(char)*(field_size));
    std::memcpy(f, field, sizeof(char)*field_size);
    this->_local_fields.push_front(f);
    this->_fields.push_back(std::string_view(f, field_size));
    return;
}

void Command::add_field_ptr(char* field, unsigned long long field_size)
{
    /* This function adds a field to the fields data
    structure without copying the data.  This means
    that the memory needs to be valid when it is later
    accessed.  This function should be used for very large
    fields.
    */

    this->_fields.push_back(std::string_view(field, field_size));
    return;
}

Command::iterator Command::begin()
{
    /* Returns a iterator pointing to the first
    field in the command
    */
    return this->_fields.begin();
}

Command::const_iterator Command::cbegin()
{
    /* Returns a const_iterator pointing to the first
    field in the command
    */
    return this->_fields.cbegin();
}

Command::iterator Command::end()
{
    /* Returns a iterator pointing to the past-the-end
    field in the command
    */
    return this->_fields.end();
}

Command::const_iterator Command::cend()
{
    /* Returns a const_iterator pointing to the past-the-end
    field in the command
    */
    return this->_fields.cend();
}