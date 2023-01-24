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

#ifndef SMARTREDIS_PIPELINEREPLY_H
#define SMARTREDIS_PIPELINEREPLY_H

#include "stdlib.h"
#include <sw/redis++/redis++.h>
#include <iostream>
#include <vector>
#include <queue>

#include "commandreply.h"

namespace SmartRedis {

class PipelineReply;

/*!
*   \brief Redis++ pipeline command reply type
*/
typedef sw::redis::QueuedReplies QueuedReplies;

///@file
/*!
*   \brief The PipelineReply class stores and processes replies
*          from pipelined commands
*   \details PipelineReply is built around processing
*            redis-plus-plus QueuedReplies.  The redis-plus-plus
*            generic pipeline execution returns a QueuedReplies
*            object.  The QueuedReplies object consists of a
*            vector of unique_ptr to redisReply objects.
*            Because this vector is a private member variable without
*            sufficient access methods, the underlying unique_ptr
*            cannot be moved out of the QueuedReplies vector.  As
*            a result, this class is limited in some functionality
*            compared to CommandReply.  Additionally,
*            because of the implementation of QueuedReplies,
*            PipelineReply is limited to only move semantics (no
*            copy constructor or copy assignment operator).
*/
class PipelineReply {

    public:

        /*!
        *   \brief Default PipelineReply constructor
        */
        PipelineReply() = default;

        /*!
        *   \brief Default PipelineReply destructor
        */
        ~PipelineReply() = default;

        /*!
        *   \brief PipelineReply copy constructor (not allowed)
        *   \param reply The PipelineReply to copy
        */
        PipelineReply(const PipelineReply& reply) = delete;

        /*!
        *   \brief PipelineReply copy assignment operator (not allowed)
        *   \param reply The PipelineReply to copy
        *   \returns PipelineReply reference
        */
        PipelineReply& operator=(PipelineReply& reply) = delete;

        /*!
        *   \brief PipelineReply move constructor
        *   \param reply The PipelineReply for construction
        */
        PipelineReply(PipelineReply&& reply) = default;

        /*!
        *   \brief PipelineReply move assignment operator
        *   \param reply The PipelineReply for assignment
        *   \returns PipelineReply reference
        */
        PipelineReply& operator=(PipelineReply&& reply) = default;

        /*!
        *   \brief PipelineReply move constructor using QueuedReplies
        *   \param reply The QueuedReplies used for construction
        */
        PipelineReply(QueuedReplies&& reply);

        /*!
        *   \brief PipelineReply move assignment operator
        *          using QueuedReplies
        *   \param reply The QueuedReplies for assignment
        *   \returns PipelineReply reference
        */
        PipelineReply& operator=(QueuedReplies&& reply);

        /*!
        *   \brief Add PipelineReply contents to a PipelineReply object
        *          via move semantics
        *   \param reply PipelineReply object
        */
        void operator+=(PipelineReply&& reply);

        /*!
        *   \brief Index operator for PipelineReply that will
        *          return the indexed element of the PipelineReply.
        *          Note that this is a shallow copy and the referenced
        *          memory is only valid as long as the PipelineReply
        *          has not been destroyed.
        *   \param index Index to retrieve
        *   \returns The indexed CommandReply
        */
        CommandReply operator[](size_t index);

        /*!
        *   \brief Get the number of CommandReply in the PipelineReply
        *   \returns The number of CommandReply in the PipelineReply
        */
        size_t size();

        /*!
        *   \brief Return if any of the CommandReply in the PipelineReply
        *          has an error
        *   \returns True if any CommandReply has an error, otherwise false
        */
        bool has_error();

        /*!
        *   \brief Reorder the stored pipeline command reply messages based
        *          on the provided index vector.
        *   \details This function allows a user to reorder the stored command
        *            replies in the cases where the operator [] should align
        *            to some other sequence not maintained by the append()
        *            function.
        *   \param index_order A std::vector<size_t> of indices where each
        *                      entry should be moved
        *   \throw SmartRedis::InternalError if indices does not match
        *          the length of the internal container for replies
        */
        void reorder(std::vector<size_t> index_order);

    private:

        /*!
        *   \brief The original redis-plus-plus object containing the
        *          pipeline response.  This is held here just
        *          for memory management.
        */
        std::vector<sw::redis::QueuedReplies> _queued_replies;

        /*!
        *   \brief This is an aggregate vector of all of the
        *          redisReply references in _queued_replies.
        *          This is used for simpler random access of replies.
        */
        std::vector<redisReply*> _all_replies;

        /*!
        *   \brief Add a QueuedReplies object ot the internal
        *          inventory of QueuedReplies.  This also
        *          adds the replies contained within the
        *          QueuedReplies object to the internal inventory
        *          of replies.
        *   \param reply The QueuedReplies object to add
        */
        void _add_queuedreplies(QueuedReplies&& reply);

};

} //namespace SmartRedis

#endif //SMARTREDIS_PIPELINEREPLY_H
