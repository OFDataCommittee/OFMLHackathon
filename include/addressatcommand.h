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

#ifndef SMARTREDIS_ADDRESSATCOMMAND_H
#define SMARTREDIS_ADDRESSATCOMMAND_H

#include "nonkeyedcommand.h"
#include "srexception.h"

///@file

namespace SmartRedis {

class RedisServer;

/*!
*   \brief The AddressAtCommand class constructs non-keyed Client
*          commands that address the user given db node(s).
*          This is a subclass of the NonKeyedCommand class.
*/
class AddressAtCommand : public NonKeyedCommand
{
    public:
        /*!
        *   \brief Run this Command on the RedisServer.
        *   \param server A pointer to the RedisServer
        */
        virtual CommandReply run_me(RedisServer* server);

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
        virtual Command* clone();

        /*!
        *   \brief Returns host of database node
        *   \param address The address of the database node
        *   \returns The host of the database node
        *   \throw RuntimeException if ':' is at the start of address,
        *          or if ':' is not found in address; BadAllocException
        *          if allocating storage for the host string fails.
        */
        inline std::string parse_host(std::string address)
        {
            std::string host;
            size_t end_position = address.find(":");

            if (end_position == 0 || end_position == std::string::npos) {
                throw SRRuntimeException(std::string(address) +
                                         " is not a valid database node address.");
            }

            try {
                host = address.substr(0, end_position);
            }
            catch (std::bad_alloc& ba) {
                throw SRBadAllocException(ba.what());
            }
            return host;
        }

        /*!
        *   \brief Returns port of database node
        *   \param address The address of the database node
        *   \returns The port of the database node
        *   \throw RuntimeException if ':' is at the end of the address,
        *          or if ':' is not found in address, or if the port
        *          conversion to uint64_t cannot be performed, or if the
        *          string representation of the port is out of the range of
        *          representable values by an uint_64; BadAllocException
        *          in the event of a memory allocation failure
        */
        inline uint64_t parse_port(std::string address)
        {
            size_t start_position = address.find(":");
            if ((start_position >= address.size() - 1) || (start_position == std::string::npos)) {
                throw SRRuntimeException(std::string(address) +
                                         " is not a valid database node address.");
            }

            uint64_t port;

            try {
                std::string port_string = address.substr(start_position + 1);
                port = std::stoul(port_string, nullptr, 0);
            }
            catch (std::bad_alloc& ba) {
                throw SRBadAllocException(ba.what());
            }
            catch (std::invalid_argument& ia) {
                throw SRRuntimeException(ia.what());
            }
            catch (std::out_of_range& oor) {
                throw SRRuntimeException(oor.what());
            }
            return port;
        }

};

} //namespace SmartRedis

#endif //ADDRESSATCOMMAND