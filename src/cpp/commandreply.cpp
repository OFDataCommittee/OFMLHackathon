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

#include "commandreply.h"

using namespace SmartRedis;

// CommandReply constructor from a redisReply.
CommandReply::CommandReply(redisReply* reply)
{
  this->_uptr_reply = 0;
  this->_reply = reply;
}

// Move constructor with RedisReplyUPtr as input
CommandReply::CommandReply(RedisReplyUPtr&& reply)
{
  this->_uptr_reply = std::move(reply);
  this->_reply = this->_uptr_reply.get();
}

// Move constructor with redisReply as input
CommandReply::CommandReply(redisReply*&& reply)
{
  this->_uptr_reply = 0;
  this->_reply = std::move(reply);
}

// Move constructor with CommandReply as input
CommandReply::CommandReply(CommandReply&& reply)
{
  if (this != &reply) {
    this->_uptr_reply = std::move(reply._uptr_reply);
    this->_reply = this->_uptr_reply.get();
  }
}

// Move assignment operator with RedisReplyUPtr as input
CommandReply& CommandReply::operator=(RedisReplyUPtr&& reply)
{
    this->_uptr_reply = std::move(reply);
    this->_reply = this->_uptr_reply.get();
    return *this;
}

// Move assignment operator with redisReply as input.
CommandReply& CommandReply::operator=(redisReply*&& reply)
{
    this->_uptr_reply = 0;
    this->_reply = std::move(reply);
    return *this;
}

// Move assignment operator with CommandReply as input.
CommandReply& CommandReply::operator=(CommandReply&& reply)
{
  if( this != &reply) {
    this->_uptr_reply = std::move(reply._uptr_reply);
    this->_reply = this->_uptr_reply.get();
  }
  return *this;
}

// Get the string field of the reply
char* CommandReply::str()
{
  if (this->_reply->type != REDIS_REPLY_STRING) {
    throw std::runtime_error("A pointer to the reply str "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  }
  return this->_reply->str;
}

// Get the integer field of the reply
long long CommandReply::integer()
{
  if (this->_reply->type != REDIS_REPLY_INTEGER) {
    throw std::runtime_error("The reply integer "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  }
  return this->_reply->integer;
}

// Get the double field of the reply
double CommandReply::dbl()
{
  if (this->_reply->type!=REDIS_REPLY_DOUBLE) {
    throw std::runtime_error("The reply double "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  }
  return this->_reply->dval;
}

// Index operator for CommandReply that will return the indexed element of
// the CommandReply if there are multiple elements
CommandReply CommandReply::operator[](int index)
{
  if (this->_reply->type!=REDIS_REPLY_ARRAY) {
    throw std::runtime_error("The reply cannot be indexed "\
                             "because the reply type is " +
                             this->redis_reply_type());
  }
  return CommandReply(this->_reply->element[index]);
}

// Get the length of the CommandReply string field
size_t CommandReply::str_len()
{
  if (this->_reply->type!=REDIS_REPLY_STRING) {
    throw std::runtime_error("The length of the reply str "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  }
  return this->_reply->len;
}

// Get the number of elements in the CommandReply
size_t CommandReply::n_elements()
{
  if (this->_reply->type!=REDIS_REPLY_ARRAY) {
    throw std::runtime_error("The number of elements "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  }
  return this->_reply->elements;
}

// Return the number of errors in the CommandReply and any nested CommandReply
int CommandReply::has_error()
{
    int num_errors = 0;
    if (this->_reply->type == REDIS_REPLY_ERROR)
        num_errors++;
    else if (this->_reply->type == REDIS_REPLY_ARRAY) {
        for (size_t i = 0; i < this->_reply->elements; i++) {
          CommandReply tmp = (*this)[i];
          num_errors += tmp.has_error();
        }
    }
    return num_errors;
}

// This will print any errors in the CommandReply or nested CommandReply.
void CommandReply::print_reply_error()
{
    if (this->_reply->type == REDIS_REPLY_ERROR) {
        std::string_view error(this->_reply->str, this->_reply->len);
        std::cout << error << std::endl;
    }
    else if (this->_reply->type == REDIS_REPLY_ARRAY) {
        for (size_t i = 0; i < this->_reply->elements; i++) {
          CommandReply tmp = (*this)[i];
          tmp.print_reply_error();
        }
    }
    return;
}

// Return the type of the CommandReply in the form of a string.
std::string CommandReply::redis_reply_type()
{
  switch (this->_reply->type) {
    case REDIS_REPLY_STRING:
      return "REDIS_REPLY_STRING";
    case REDIS_REPLY_ARRAY:
      return "REDIS_REPLY_ARRAY";
    case REDIS_REPLY_INTEGER:
      return "REDIS_REPLY_INTEGER";
    case REDIS_REPLY_NIL:
      return "REDIS_REPLY_NIL";
    case REDIS_REPLY_STATUS:
      return "REDIS_REPLY_STATUS";
    case REDIS_REPLY_ERROR:
      return "REDIS_REPLY_ERROR";
    case REDIS_REPLY_DOUBLE:
      return "REDIS_REPLY_DOUBLE";
    case REDIS_REPLY_BOOL:
      return "REDIS_REPLY_BOOL";
    case REDIS_REPLY_MAP:
      return "REDIS_REPLY_MAP";
    case REDIS_REPLY_SET:
      return "REDIS_REPLY_SET";
    case REDIS_REPLY_ATTR:
      return "REDIS_REPLY_ATTR";
    case REDIS_REPLY_PUSH:
      return "REDIS_REPLY_PUSH";
    case REDIS_REPLY_BIGNUM:
      return "REDIS_REPLY_BIGNUM";
    case REDIS_REPLY_VERB:
      return "REDIS_REPLY_VERB";
    default:
      throw std::runtime_error("Invalid Redis reply type");
  }
}

// Print the reply structure of the CommandReply
void CommandReply::print_reply_structure(std::string index_tracker)
{
  // TODO these recursive functions can't use 'this' unless
  // we have a constructor that takes redisReply*
  std::cout << index_tracker + " type: " << this->redis_reply_type()
            << std::endl;
  switch (this->_reply->type) {
    case REDIS_REPLY_STRING:
      std::cout << index_tracker + " value: "
               << std::string(this->str(), this->str_len()) <<std::endl;
      break;
    case REDIS_REPLY_ARRAY:
      for (size_t i = 0; i < this->n_elements(); i++) {
        std::string r_prefix = index_tracker + "[" + std::to_string(i) + "]";
        CommandReply tmp = (*this)[i];
        tmp.print_reply_structure(r_prefix);
      }
      break;
    case REDIS_REPLY_INTEGER:
      std::cout << index_tracker + " value: " << this->_reply->integer
                << std::endl;
      break;
    case REDIS_REPLY_DOUBLE:
      std::cout << index_tracker + " value: " << this->_reply->dval
                << std::endl;
      break;
    case REDIS_REPLY_ERROR:
      std::cout << index_tracker + " value: "
                << std::string(this->str(), this->str_len()) << std::endl;
      break;
    case REDIS_REPLY_BOOL:
      std::cout << index_tracker + " value: " << this->_reply->integer
                << std::endl;
      break;
    default:
      std::cout << index_tracker << " value type not supported." << std::endl;
  }
}

// EOF
