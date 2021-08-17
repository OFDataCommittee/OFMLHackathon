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

#include "commandlist.h"

using namespace SmartRedis;

// CommandList copy constructor
CommandList::CommandList(const CommandList& cmd_lst)
{
    std::vector<Command*>::const_iterator c_it = cmd_lst._commands.cbegin();
    for ( ; c_it != cmd_lst._commands.cend(); c_it++) {
        Command* curr_command = new Command(**c_it);
        this->_commands.push_back(curr_command);
    }
}

// CommandList copy assignment operator
CommandList& CommandList::operator=(const CommandList& cmd_lst)
{
    // Check for self-assignment
    if (this == &cmd_lst)
        return *this;

    // Remove my old contents
    std::vector<Command*>::iterator it = this->_commands.begin();
    for (; it != this->_commands.end(); it++)
        delete (*it);
    this->_commands.clear();

    // Copy over the new contents
    std::vector<Command*>::const_iterator c_it = cmd_lst._commands.begin();
    for ( ; c_it != cmd_lst._commands.end(); c_it++) {
        Command* curr_command = new Command(**c_it);
        this->_commands.push_back(curr_command);
    }

    // Done
    return *this;
}

// Default CommandList destructor
CommandList::~CommandList()
{
    std::vector<Command*>::iterator it = this->_commands.begin();
    for (; it != this->_commands.end(); it++)
        delete (*it);
}

// Dynamically allocate a new Command and return a pointer to the new Command
Command* CommandList::add_command()
{
    this->_commands.push_back(new Command());
    return this->_commands.back();
}

// Returns an iterator pointing to the first Command
CommandList::iterator CommandList::begin()
{
    return this->_commands.begin();
}

// Returns a const iterator pointing to the first Command
CommandList::const_iterator CommandList::cbegin()
{
    return this->_commands.cbegin();
}

// Returns an iterator pointing to the last Command
CommandList::iterator CommandList::end()
{
    return this->_commands.end();
}

// Returns a const iterator pointing to the last Command
CommandList::const_iterator CommandList::cend()
{
    return this->_commands.cend();
}

// EOF
