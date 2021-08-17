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

#include "dbnode.h"

using namespace SmartRedis;

// DBNode constructor
DBNode::DBNode()
{
   this->name = "";
   this->ip = "";
   this->port = -1;
   this->lower_hash_slot = -1;
   this->upper_hash_slot = -1;
}

// DBNode constructor with connection and hash slot information.
DBNode::DBNode(std::string _ip, std::string _name,
               uint64_t _port, uint64_t l_slot,
               uint64_t u_slot, std::string _prefix)
    : name(_name), ip(_ip), port(_port), lower_hash_slot(l_slot), upper_hash_slot(u_slot),
      prefix(_prefix)
{
    // NOP
}

// Less than operator. Returns True if the lower hash slot of this node
// is less than the other lower hash slot.
bool DBNode::operator<(const DBNode& db_node) const
{
    return this->lower_hash_slot < db_node.lower_hash_slot;
}

// EOF
