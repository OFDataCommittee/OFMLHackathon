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
        *   \brief Default KeyedCommand constructor
        */
        KeyedCommand() = default;

        /*!
        *   \brief KeyedCommand copy constructor
        *   \param cmd The KeyedCommand to copy for construction
        */
        KeyedCommand(const KeyedCommand& cmd) = default;

        /*!
        *   \brief KeyedCommand default move constructor
        */
        KeyedCommand(KeyedCommand&& cmd) = default;

        /*!
        *   \brief KeyedCommand copy assignment operator
        *   \param cmd The KeyedCommand to copy for assignment
        */
        KeyedCommand& operator=(const KeyedCommand& cmd) = default;

        /*!
        *   \brief KeyedCommand move assignment operator
        */
        KeyedCommand& operator=(KeyedCommand&& cmd) = default;

        /*!
        *   \brief KeyedCommand destructor
        */
        virtual ~KeyedCommand() = default;

        /*!
        *   \brief Deep copy operator
        *   \details This method creates a new derived
        *            type Command and returns a Command*
        *            pointer.  The new derived type is
        *            allocated on the heap.  Contents
        *            are copied using the copy assignment
        *            operator for the derived type. This is meant
        *            to provide functionality to deep
        *            copy a Command.
        *   \returns A pointer to dynamically allocated
        *            derived type cast to parent Command
        *            type.
        */
        virtual Command* clone() = 0;

        /*!
        *   \brief Run this Command on the RedisServer.
        *   \param server A pointer to the RedisServer
        */
        virtual CommandReply run_me(RedisServer* server) = 0;
};

} //namespace SmartRedis

#endif //KEYEDCOMMAND