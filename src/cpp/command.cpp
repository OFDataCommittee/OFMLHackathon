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

#include "command.h"

using namespace SmartRedis;

Command::Command(const Command& cmd)
{
    *this = cmd;
}

Command& Command::operator=(const Command& cmd)
{
    if(this!=&cmd) {
        make_empty();

        this->_fields.resize(cmd._fields.size());

        // copy pointer fields and put into _fields
        this->_ptr_fields = cmd._ptr_fields;
        std::vector<std::pair<char*, size_t> >::const_iterator ptr_it =
            this->_ptr_fields.begin();
        std::vector<std::pair<char*, size_t> >::const_iterator ptr_it_end =
            this->_ptr_fields.end();
        for(; ptr_it != ptr_it_end; ptr_it++)
            this->_fields[ptr_it->second] = ptr_it->first;

        // copy _local_fields and _cmd_keys, and put the fields into _fields
        std::vector<std::pair<char*, size_t> >::const_iterator local_it =
            cmd._local_fields.begin();
        std::vector<std::pair<char*, size_t> >::const_iterator local_it_end =
            cmd._local_fields.end();
        for(; local_it != local_it_end; local_it++) {
            // allocate memory and copy a local field
            int field_size = cmd._fields[local_it->second].size();
            char* f = (char*) malloc(sizeof(char)*field_size);
            std::memcpy(f, local_it->first, sizeof(char)*field_size);
            this->_local_fields.push_back(std::pair<char*, size_t>
                                         (f, local_it->second));
            this->_fields[local_it->second] = std::string_view(f, field_size);

            if(cmd._cmd_keys.count(cmd._fields[local_it->second]) != 0) {
                // copy a command key
                this->_cmd_keys[std::string_view(f, field_size)] =
                    local_it->second;
            }
        }
    }
    return *this;
}

Command::~Command()
{
    make_empty();
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
    this->_local_fields.push_back({f, _fields.size()});
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
    this->_local_fields.push_back({f, _fields.size()});
    this->_fields.push_back(std::string_view(f, field_size));

    if(is_key)
        this->_cmd_keys[std::string_view(f, field_size)] =
            this->_fields.size()-1;

    return;
}

void Command::add_field(const char* field, bool is_key)
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
    this->_local_fields.push_back({f, _fields.size()});
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
    this->_ptr_fields.push_back({field, _fields.size()});
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
    this->_ptr_fields.push_back({(char*)field.data(), _fields.size()});

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
    for(int i=0; i<fields.size(); i++) {
        this->add_field(fields[i], is_key);
    }
    return;
}

std::string Command::first_field()
{
    if(this->begin() == this->end())
        throw std::runtime_error("No fields exist in the Command.");
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

void Command::make_empty()
{
    std::vector<std::pair<char*, size_t> >::iterator it =
        this->_local_fields.begin();
    std::vector<std::pair<char*, size_t> >::iterator it_end =
        this->_local_fields.end();
    for(; it!=it_end; it++) {
        free((*it).first);
        *it = {NULL,0};
    }
    this->_local_fields.clear();
    this->_ptr_fields.clear();
    this->_cmd_keys.clear();
    this->_fields.clear();
}

void Command::set_exec_address_port(std::string address,
                                    uint16_t port)
{
    this -> _address = address;
    this -> _port = port;
    return;
}

std::string Command::get_address()
{
    return this -> _address;
}

uint16_t Command::get_port()
{
    return this -> _port;
}