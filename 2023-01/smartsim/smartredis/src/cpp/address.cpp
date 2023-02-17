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

#include <iostream>
#include "address.h"
#include "srexception.h"

using namespace SmartRedis;

// SRAddress constructor
SRAddress::SRAddress(const std::string& addr_spec)
{
    // Check address type
    // Unix Domain Socket (UDS) addresses are prefixed with "unix://""
    std::string prefix("unix://");
    _is_tcp = (addr_spec.compare(0, prefix.size(), prefix) != 0);

    // Parse the incoming address specification
    if (_is_tcp) { // TCP address
        // Strip off tcp:// if present
        std::string address(addr_spec);
        prefix = "tcp://";
        if (address.compare(0, prefix.size(), prefix) == 0) {
            address = address.substr(prefix.size());
        }

        // TCP addresses are of the form host:port, so split the two halves
        size_t colon_position = address.find(":");
        if (colon_position == 0 || (colon_position >= address.size() - 1) ||
            colon_position == std::string::npos) {
            throw SRRuntimeException(address +
                                     " is not a valid database node address.");
        }

        try {
            _tcp_host = address.substr(0, colon_position);
            std::string port_string = address.substr(colon_position + 1);
            _tcp_port = std::stoul(port_string, nullptr, 0);
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
    }
    else { // UDS address
        _uds_file = addr_spec.substr(prefix.size());
    }
}

// Comparison operator
bool SRAddress::operator==(const SRAddress& other) const
{
    return (
        (_is_tcp == other._is_tcp) &&
        (_is_tcp ? _tcp_host == other._tcp_host : true) &&
        (_is_tcp ? _tcp_port == other._tcp_port : true) &&
        (_is_tcp ? true : _uds_file == other._uds_file)
    );
}


// Convert to a string
std::string SRAddress::to_string(bool add_tcp_protocol /* = false*/) const
{
    if (_is_tcp) {
        std::string result(add_tcp_protocol ? "tcp://" : "");
        return result + _tcp_host + ":" + std::to_string(_tcp_port);
    }
    else {
        return "unix://" + _uds_file;
    }
}
