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

#ifndef SMARTREDIS_KEYEDCOMMAND_H
#define SMARTREDIS_KEYEDCOMMAND_H

#include "command.h"


///@file

namespace SmartRedis {

class RedisServer;

/*!
*   \brief The KeyedCommand intermediary class constructs Client
*          commands with keys.
*   \details The KeyedCommand class has multiple methods for dealing
*            with keyed commands.
*/
class KeyedCommand : public Command
{
    public:

        /*!
        *   \brief Return a copy of all Command keys
        *   \returns std::vector of Command keys
        */
        std::vector<std::string> get_keys();

        virtual CommandReply runme(RedisServer * r);
};

/*!
*   \brief The MultiKeyCommand class constructs Client
*          commands with multiple keys. This is a subclass
*          of the KeyedCommand class.
*/
class MultiKeyCommand : public KeyedCommand
{
    public:
	    virtual CommandReply runme(RedisServer * r);
};


/*!
*   \brief The SingleKeyCommand class constructs Client
*          commands with only one key. This is a subclass
*          of the KeyedCommand class.
*/
class SingleKeyCommand : public KeyedCommand
{
    public:
	    virtual CommandReply runme(RedisServer * r);
};

/*!
*   \brief The MultiCommandCommand class constructs Client
*          commands with multiple commands. This is a subclass
*          of the KeyedCommand class.
*/
class MultiCommandCommand : public KeyedCommand
{
    public:
        virtual CommandReply runme(RedisServer * r);
};

} //namespace SmartRedis

#endif //KEYEDCOMMAND