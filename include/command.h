/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Hewlett Packard Enterprise
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SMARTREDIS_COMMAND_H
#define SMARTREDIS_COMMAND_H

#include "stdlib.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <iostream>

///@file

namespace SmartRedis {

class Command;

/*!
*   \brief The Command class constructs Client commands.
*   \details The Command class has multiple methods for adding
*          fields to the Command.  The Command.add_field()
*          methods will copy the underlying field data,
*          while the Command.add_field_ptr() methods
*          will only maintain a pointer to the field data.
*          The Command.add_field_ptr() methods are ideal
*          for large field values.
*/
class Command
{
    public:
        /*!
        *   \brief Default Command constructor
        */
        Command() = default;

        /*!
        *   \brief Command copy constructor
        *   \param cmd The Command to copy for construction
        */
        Command(const Command& cmd);

        /*!
        *   \brief Command default move constructor
        */
        Command(Command&& cmd) = default;

        /*!
        *   \brief Command copy assignment operator
        *   \param cmd The Command to copy for assignment
        */
        Command& operator=(const Command& cmd);

        /*!
        *   \brief Command move assignment operator
        */
        Command& operator=(Command&& cmd) = default;

        /*!
        *   \brief Command destructor
        */
        ~Command();

        /*!
        *   \brief Add a field to the Command from a string.
        *   \details The string field value is copied to the
        *            Command.
        *   \param field The field to add to the Command
        *   \param is_key Boolean indicating if the field
        *                 should be treated as a key for the
        *                 Command.
        */
        void add_field(std::string field,
                       bool is_key=false);

        /*!
        *   \brief Add a field to the Command from a c-string.
        *   \details The c-string field value is copied to the
        *            Command.
        *   \param field The field to add to the Command
        *   \param is_key Boolean indicating if the field
        *                 should be treated as a key for the
        *                 Command.
        */
        void add_field(char* field, bool is_key=false);

        /*!
        *   \brief Add a field to the Command from a const c-string.
        *   \details The const c-string field value is copied
        *            to the Command.
        *   \param field The field to add to the Command
        *   \param is_key Boolean indicating if the field
        *                 should be treated as a key for the
        *                 Command.
        */
        void add_field(const char* field, bool is_key=false);

        /*!
        *   \brief Add a field to the Command from a c-string.
        *   \details The c-string will not be copied to the
        *            Command object.  A pointer is kept that
        *            points to the c-string location in
        *            memory.  As a result, the c-string
        *            memory must be valid up until the
        *            execution of the Command.  A field
        *            that is not copied cannot be a Command
        *            key.
        *   \param field The field to add to the Command
        *   \param field_size The length of the c-string
        */
        void add_field_ptr(char* field, size_t field_size);

        /*!
        *   \brief Add a field to the Command from a
        *          std::string_view.
        *   \details The std::string_view data will not be copied
        *            to the Command object.  The internal
        *            data structure will maintain a pointer
        *            to the std::string_view data location in
        *            memory.  As a result, the std::string_view
        *            memory must be valid up until the
        *            execution of the Command.  A field
        *            that is not copied cannot be a Command
        *            key.
        *   \param field The field to add to the Command
        */
        void add_field_ptr(std::string_view field);

        /*!
        *   \brief Add fields to the Command
        *          from a vector of strings.
        *   \details The string field values are copied to the
        *            Command.
        *   \param fields The fields to add to the Command
        *   \param is_key Boolean indicating if the all
        *                 of the fields are Command keys
        */
        void add_fields(const std::vector<std::string>& fields, bool is_key=false);


        /*!
        *   \brief Add fields to the Command
        *          from a vector of type T
        *   \details The field values are copied to the
        *            Command.  The type T must be convertable
        *            to a string via std::to_string.
        *   \tparam T Any type that can be converted
        *             to a string via std::to_string.
        *
        *   \param fields The fields to add to the Command
        *   \param is_key Boolean indicating if the all
        *                 of the fields are Command keys
        */
        template <class T>
        void add_fields(const std::vector<T>& fields, bool is_key=false);

        /*!
        *   \brief Return true if the Command has keys
        *   \returns True if the Command has keys, otherwise
        *            False
        */
        bool has_keys();

        /*!
        *   \brief Return a copy of all Command keys
        *   \returns std::vector of Command keys
        */
        std::vector<std::string> get_keys();

        /*!
        *   \brief Change a Command key value
        *   \param old_key The value of the old key field
        *   \param new_key The value of the new key field
        */
        void update_key(std::string old_key,
                        std::string new_key);

        /*!
        *   \brief Set address and port for command
        *          to be executed on
        *   \param address Address of database
        *   \param port Port of database
        */
        void set_exec_address_port(std::string address,
                                   uint16_t port);

        /*!
        *   \brief Get address that command will be
        *          to be executed on
        *   \return std::string of address
        *           if an address hasn't been set,
        *                 returns an empty string
        */
        std::string get_address();

        /*!
        *   \brief Get port that command will be
        *          to be executed on
        *   \return uint16_t of port
        *           if port hasn't been set, returns 0
        */
        uint16_t get_port();

        /*!
        *   \brief Get the value of the field field
        *   \returns std::string of the first Command field
        */
        std::string first_field();


        /*!
        *   \brief Get a string of the entire Command
        *   \returns std::string concatenating all Command
        *            fields
        */
        std::string to_string();

        /*!
        *   \typedef An iterator type for iterating
        *            over all Command fields
        */
        typedef std::vector<std::string_view>::iterator iterator;

        /*!
        *   \typedef An iterator type for iterating
        *            over all Command fields
        */
        typedef std::vector<std::string_view>::const_iterator const_iterator;


        /*!
        *   \brief Returns an iterator pointing to the
        *          first field in the Command
        *   \returns Command iterator to the first field
        */
        iterator begin();

        /*!
        *   \brief Returns a const iterator pointing to the
        *          first field in the Command
        *   \returns const Command iterator to the first field
        */
        const_iterator cbegin();

        /*!
        *   \brief Returns an iterator pointing to the
        *          past-the-end field in the Command
        *   \returns Command iterator to the past-the-end field
        */
        iterator end();


        /*!
        *   \brief Returns a const iterator pointing to the
        *          past-the-end field in the Command
        *   \returns const Command iterator to the past-the-end field
        */
        const_iterator cend();

    private:

        /*!
        *   \brief All local fields and
        *          pointer fields in the order that
        *          they were added to the Command
        */
        std::vector<std::string_view> _fields;

        /*!
        *   \brief All of the fields whose memory was
                   allocated by Command, along with
                   their associated index in _fields
        */
        std::vector<std::pair<char*, size_t> > _local_fields;

        /*!
        *   \brief All of the fields whose memory was not
        *          allocated by Command, along with their
        *          associated index in _fields
        */
        std::vector<std::pair<char*, size_t> > _ptr_fields;

        /*!
        *   \brief Address of database node
        */
        std::string _address;

        /*!
        *   \brief Port of database node
        */
        uint64_t _port;

        /*!
        *   \brief Unordered map of std::string_view to
        *          the index of _fields and _local_fields
        *          for the std::string_view
        */
        std::unordered_map<std::string_view, size_t> _cmd_keys;

        /*!
        *   \brief Helper function for emptying the Command
        */
        void make_empty();
};

#include "command.tcc"

} //namespace SmartRedis

#endif //SMARTREDIS_COMMAND_H
