#ifndef SMARTSIM_COMMAND_H
#define SMARTSIM_COMMAND_H

#include "stdlib.h"
#include <string>
#include <vector>
#include <cstring>
#include <queue>
#include <forward_list>
#include <iostream>

///@file
///\brief The Command class for constructing data transfer commands

namespace SILC {

class Command;

class Command
{
    private:

        std::vector<std::string_view> _fields;
        std::vector<char*> _local_fields;

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
        //! Add a field to the command for data managed by another object
        void add_field_ptr(std::string_view field);
        //! Add more than one string field
        void add_fields(const std::vector<std::string>& fields);
        //! Add more than one field
        template <class T>
        void add_fields(const std::vector<T>& fields);

        //! Return a copy of the first field for information purposes
        std::string first_field();
        //! Return a single string of the command
        std::string to_string();

        //! Returns an iterator pointing to the first field in the command
        iterator begin();
        //! Returns an const_iterator pointing to the first field in the command
        const_iterator cbegin();
        //! Returns an iterator pointing to the past-the-end field in the command
        iterator end();
        //! Returns an const_iterator pointing to the past-the-end field in the command
        const_iterator cend();
};

#include "command.tcc"

} //namespace SILC

#endif //SMARTSIM_COMMAND_H
