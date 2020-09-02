#include "command.h"

Command::Command()
{
}

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

void Command::add_field(std::string field)
{
    /* Copy the field string into a char* that is stored
    locally in the Command object.  The new string is not
    null terminated because the fields vector is of type
    string_view which stores the length of the string
    */

    int field_size = field.size();
    char* f = (char*) malloc(sizeof(char)*(field_size+1));
    field.copy(f, field_size, 0);
    f[field_size]=0;
    this->_local_fields.push_back(f);
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
    this->_local_fields.push_back(f);
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

void Command::add_field_ptr(std::string_view field)
{
    /* This function adds a field to the fields data
    structure without copying the data.  This means
    that the memory needs to be valid when it is later
    accessed.  This function should be used for very large
    fields.
    */

    this->_fields.push_back(field);
    return;
}


void Command::add_fields(const std::vector<std::string>& fields)
{
    /* Copy field strings into a char* that is stored
    locally in the Command object.  The new string is not
    null terminated because the fields vector is of type
    string_view which stores the length of the string
    */
    for(int i=0; i<fields.size(); i++)
        this->add_field(fields[i]);
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