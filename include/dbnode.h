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

#ifndef SMARTREDIS_DBNODE_H
#define SMARTREDIS_DBNODE_H

#include "stdlib.h"
#include <string>

///@file

namespace SmartRedis {

class DBNode;

//@file
/*!
*   \brief The DBNode class stores connection and hash slot
*          information for the database node.
*/
class DBNode{

    public:

        /*!
        *   \brief DBNode constructor
        */
        DBNode();

        /*!
        *   \brief Default DBNode copy constructor
        *   \param dbnode The DBNode to copy for construction
        */
        DBNode(const DBNode& dbnode) = default;

        /*!
        *   \brief Default DBNode move constructor
        *   \param dbnode The DBNode to move for construction
        */
        DBNode(DBNode&& dbnode) = default;

        /*!
        *   \brief Default DBNode copy assignment operator
        *   \param dbnode The DBNode to copy assign
        *   \returns Reference to the DBNode that has been assigned
        */
        DBNode& operator=(const DBNode& dbnode) = default;

        /*!
        *   \brief Default DBNode move assignment operator
        *   \param dbnode The DBNode to move assign
        *   \returns Reference to the DBNode that has been assigned
        */
        DBNode& operator=(DBNode&& dbnode) = default;

        /*!
        *   \brief DBNode constructor with connection
        *          and hash slot information.
        *   \param name The name of the DBNode
        *   \param ip The IP address of the DBNode
        *   \param port The port of the DBNode
        *   \param l_slot The lower hash slot of the DBNode
        *   \param u_slot The upper hash slot of the DBNode
        *   \param prefix A prefix that can be placed into
        *                 a hash tag that can be placed in
        *                 front of a key to ensure placement
        *                 on this DBNode.
        *
        */
        DBNode(std::string name,
               std::string ip,
               uint64_t port,
               uint64_t l_slot,
               uint64_t u_slot,
               std::string prefix
               );

        /*!
        *   \brief Default DBNode destructor
        */
        ~DBNode() = default;

        /*!
        *   \brief Less than operator.  Returns True
        *          if the lower hash slot of this
        *          node is less than the other lower
        *          hash slot.
        *   \param db_nodes DBNode to compare to
        */
        bool operator<(const DBNode& db_node) const;

        /*!
        *   \brief The IP address of the DBNode
        */
        std::string ip;

        /*!
        *   \brief The port of the DBNode
        */
        uint64_t port;

        /*!
        *   \brief The name of the DBNode
        */
        std::string name;

        /*!
        *   \brief The lower hash slot of the DBNode
        */
        uint64_t lower_hash_slot;

        /*!
        *   \brief The upper hash slot of the DBNode
        */
        uint64_t upper_hash_slot;

        /*!
        *   \brief A prefix that can be placed into
        *          a hash tag that can be placed in
        *          front of a key to ensure placement
        *          on this DBNode.
        */
        std::string prefix;

};

} //namespace SmartRedis

#endif //SMARTREDIS_DBNODE_H
