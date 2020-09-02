#ifndef SMARTSIM_COMMAND_H
#define SMARTSIM_COMMAND_H

#include "stdlib.h"
#include <string>
#include <vector>
#include <forward_list>

///@file
///\brief The Command class for constructing data transfer commands
class Command;

class Command
{
    private:

        std::vector<std::string_view> _fields;
        std::forward_list<char*> _local_fields;

    public:

        typedef std::vector<std::string_view>::iterator iterator;
        typedef std::vector<std::string_view>::const_iterator const_iterator;

        //! Command constructor
        Command();

        //! Command destructor
        ~Command();

        //! Add a field to the command from a string value
        void add_field(std::string field);
        //! Add a field to the command from a char* value
        void add_field(char* field);
        //! Add a field to the command for data managed by another object
        void add_field_ptr(char* field, unsigned long long field_size);

        //! Returns an iterator pointing to the first field in the command
        iterator begin();
        //! Returns an const_iterator pointing to the first field in the command
        const_iterator cbegin();
        //! Returns an iterator pointing to the past-the-end field in the command
        iterator end();
        //! Returns an const_iterator pointing to the past-the-end field in the command
        const_iterator cend();


};
#endif //SMARTSIM_COMMAND_H