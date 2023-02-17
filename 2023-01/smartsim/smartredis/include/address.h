/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

#ifndef SMARTREDIS_ADDRESS_H
#define SMARTREDIS_ADDRESS_H

#include <stdio.h>
#include <stdlib.h>

///@file

#ifdef __cplusplus
#include <string>
namespace SmartRedis {

/*!
*   \brief  SRAddress: generalized representation of a server address,
*           encompassing both TCP (address:port) and Unix Domain Socket
*           (filename) style addresses
*/
class SRAddress
{
    public:
    /*!
    *   \brief SRAddress default constructor
    */
    SRAddress() : _is_tcp(true), _tcp_host(""), _tcp_port(0) {}

    /*!
    *   \brief SRAddress constructor
    *   \param addr_spec The address (string form) of a server
    */
    SRAddress(const std::string& addr_spec);

    /*!
    *   \brief Default SRAddress copy constructor
    *   \param other The address to copy
    */
    SRAddress(const SRAddress& other) = default;

    /*!
    *   \brief Comparison operator
    *   \param other The address to compare
    */
    bool operator==(const SRAddress& other) const;

    /*!
    *   \brief Convert an address to string form
    *   \param add_tcp_protocol Add "tcp://" protocol prefix to output if true
    *   \returns Stringified address specification
    */
    virtual std::string to_string(bool add_tcp_protocol = false) const;

    /*!
    *   \brief  Is this a TCP address? (If not, it's a Unix Domain Socket Address)
    */
    bool _is_tcp;

    /*!
    *   \brief  The TCP host (undefined for UDS)
    */
    std::string _tcp_host;

    /*!
    *   \brief  The TCP port (undefined for UDS)
    */
    uint16_t _tcp_port;

    /*!
    *   \brief  The UDS filespec (undefined for TCP)
    */
    std::string _uds_file;
};

} // namespace SmartRedis

#endif // __cplusplus
#endif // SMARTREDIS_ADDRESS_H
