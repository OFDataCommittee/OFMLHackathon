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

#include "pipelinereply.h"
#include "srexception.h"

using namespace SmartRedis;

// Move constructor with QueuedReplies
PipelineReply::PipelineReply(QueuedReplies&& reply)
{
    _add_queuedreplies(std::forward<QueuedReplies>(reply));
}

// Move assignment with QueuedReplies
PipelineReply& PipelineReply::operator=(QueuedReplies&& reply)
{
    _queued_replies.clear();
    _all_replies.clear();
    _add_queuedreplies(std::forward<QueuedReplies>(reply));
    return *this;
}

// Add PipelineReply content to Pipeline ojbect via move semantics
void PipelineReply::operator+=(PipelineReply&& reply)
{
    for (size_t i = 0; i < reply._queued_replies.size(); i++) {
        _add_queuedreplies(std::move(reply._queued_replies[i]));
    }
    reply._queued_replies.clear();
    reply._all_replies.clear();
}

// Return a shallow copy of an entry in the PipelineReply
CommandReply PipelineReply::operator[](size_t index)
{
    if (index > _all_replies.size()) {
        throw SRInternalException("An attempt was made to access index " +
                                  std::to_string(index) +
                                  " of the PipelineReply, which is beyond the "\
                                  " PipelineReply length of " +
                                  std::to_string(_all_replies.size()));
    }

    return CommandReply::shallow_clone(_all_replies[index]);
}

// Get the number of CommandReply in the PipelineReply
size_t PipelineReply::size()
{
    return _all_replies.size();
}

// Check if any CommandReply has error
bool PipelineReply::has_error()
{
    for (size_t i = 0; i < _all_replies.size(); i++) {
        if (CommandReply::shallow_clone(_all_replies[i]).has_error() > 0)
            return true;
    }
    return false;
}


// Reorder the internal order of pipeline command replies
void PipelineReply::reorder(std::vector<size_t> index_order)
{
    for (size_t i = 0; i < index_order.size(); i++) {
        while(i != index_order[i]) {
            size_t swap_index = index_order[i];
            std::swap(_all_replies[i], _all_replies[swap_index]);
            std::swap(index_order[i], index_order[swap_index]);
        }
    }
}

// Add the QueuedReplies object to the inventory
void PipelineReply::_add_queuedreplies(QueuedReplies&& reply)
{
    // Move the QueuedReplies into the inventory
    _queued_replies.push_back(std::forward<QueuedReplies>(reply));

    // Add redisReply contained in the QueuedReplies into the inventory
    size_t n_replies = _queued_replies.back().size();
    for (size_t i = 0; i < n_replies; i++) {
        _all_replies.push_back(&(_queued_replies.back().get(i)));
    }
}