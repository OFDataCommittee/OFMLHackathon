#ifndef SMARTSIM_COMMAND_H
#define SMARTSIM_COMMAND_H

#include "stdlib.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <iostream>

///@file
///\brief The Command class for constructing data transfer commands

namespace SILC {

class Command;

class Command
{
    public:

        typedef std::vector<std::string_view>::iterator iterator;
        typedef std::vector<std::string_view>::const_iterator const_iterator;

        //! Command constructor
        Command();

        //! Command copy constructor
        Command(const Command& cmd) = delete;

        //! Command move constructor
        Command(Command&& cmd) = default;

        //! Command copy assignment operator
        Command& operator=(const Command& cmd) = delete;

        //! Command move assignment operator
        Command& operator=(Command&& cmd) = default;

        //! Command destructor
        ~Command();

        //! Add a field to the command from a string value
        void add_field(std::string field, bool is_key=false);
        //! Add a field to the command from a char* value
        void add_field(char* field, bool is_key=false);
        //! Add a field to the command for data managed by another object
        void add_field_ptr(char* field, size_t field_size);
        //! Add a field to the command for data managed by another object
        void add_field_ptr(std::string_view field);

        //! Add more than one string field
        void add_fields(const std::vector<std::string>& fields, bool is_key=false);
        //! Add more than one field
        template <class T>
        void add_fields(const std::vector<T>& fields, bool is_key=false);

        //! Return true if the Command has any keys
        bool has_keys();

        //! Get a vector containing a copy of all Command keys
        std::vector<std::string> get_keys();

        //! Update a Command key value
        void update_key(std::string old_key,
                        std::string new_key);

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

    private:

        std::vector<std::string_view> _fields;
        std::vector<char*>_local_fields;

        //! Unordered map of std::string_view to the index of _fields
        //! and _local_fields for the std::string_view
        std::unordered_map<std::string_view, size_t> _cmd_keys;
};

#include "command.tcc"

} //namespace SILC

#endif //SMARTSIM_COMMAND_H
