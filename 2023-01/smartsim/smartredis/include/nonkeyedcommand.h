/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

#ifndef SMARTREDIS_NONKEYEDCOMMAND_H
#define SMARTREDIS_NONKEYEDCOMMAND_H

#include "command.h"

///@file

namespace SmartRedis {

class RedisServer;

/*!
*   \brief The NonKeyedCommand intermediary class constructs Client
*          commands without keys. These commands use db node addresses.
*   \details The KeyedCommand class has multiple methods for dealing
*            with non-keyed commands.
*/
class NonKeyedCommand : public Command
{
    public:
        /*!
        *   \brief Default NonKeyedCommand constructor
        */
        NonKeyedCommand() = default;

        /*!
        *   \brief NonKeyedCommand copy constructor
        *   \param cmd The NonKeyedCommand to copy for construction
        */
        NonKeyedCommand(const NonKeyedCommand& cmd) = default;

        /*!
        *   \brief NonKeyedCommand default move constructor
        */
        NonKeyedCommand(NonKeyedCommand&& cmd) = default;

        /*!
        *   \brief NonKeyedCommand copy assignment operator
        *   \param cmd The NonKeyedCommand to copy for assignment
        */
        NonKeyedCommand& operator=(const NonKeyedCommand& cmd) = default;

        /*!
        *   \brief KeyedCommand move assignment operator
        */
        NonKeyedCommand& operator=(NonKeyedCommand&& cmd) = default;

        /*!
        *   \brief KeyedCommand destructor
        */
        virtual ~NonKeyedCommand() = default;

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
        *   \brief Set address and port for command
        *          to be executed on
        *   \param address Address of database
        *   \param port Port of database
        */
        void set_exec_address_port(std::string address,
                                   uint64_t port);

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
        *   \return uint64_t of port
        */
        uint64_t get_port();

        /*!
        *   \brief Run this Command on the RedisServer.
        *   \param server A pointer to the RedisServer
        */
        virtual CommandReply run_me(RedisServer* server) = 0;

    private:
        /*!
        *   \brief Address of database node
        */
        std::string _address;

        /*!
        *   \brief Port of database node
        */
        uint64_t _port;
};

} //namespace SmartRedis

#endif //NONKEYEDCOMMAND