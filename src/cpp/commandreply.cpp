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

CommandReply::CommandReply(redisReply* reply)
{
  this->_uptr_reply = 0;
  this->_reply = reply;
}

CommandReply::CommandReply(RedisReplyUPtr&& reply)
{
  this->_uptr_reply = std::move(reply);
  this->_reply = this->_uptr_reply.get();
  return;
}

CommandReply::CommandReply(redisReply*&& reply)
{
  this->_uptr_reply = 0;
  this->_reply = std::move(reply);
  return;
}

CommandReply::CommandReply(CommandReply&& reply)
{
  this->_uptr_reply = std::move(reply._uptr_reply);
  this->_reply = this->_uptr_reply.get();
  return;
}

CommandReply& CommandReply::operator=(RedisReplyUPtr&& reply)
{
    this->_uptr_reply = std::move(reply);
    this->_reply = this->_uptr_reply.get();
    return *this;
}

CommandReply& CommandReply::operator=(redisReply*&& reply)
{
    this->_uptr_reply = 0;
    this->_reply = std::move(reply);
    return *this;
}

CommandReply& CommandReply::operator=(CommandReply&& reply)
{
  if(this!=&reply) {
    this->_uptr_reply = std::move(reply._uptr_reply);
    this->_reply = this->_uptr_reply.get();
  }
  return *this;
}

char* CommandReply::str()
{
  if(this->_reply->type!=REDIS_REPLY_STRING)
    throw std::runtime_error("A pointer to the reply str "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  return this->_reply->str;
}

long long CommandReply::integer()
{
  if(this->_reply->type!=REDIS_REPLY_INTEGER)
    throw std::runtime_error("The reply integer "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  return this->_reply->integer;
}

double CommandReply::dbl()
{
  if(this->_reply->type!=REDIS_REPLY_DOUBLE)
    throw std::runtime_error("The reply double "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  return this->_reply->dval;
}

CommandReply CommandReply::operator[](int index)
{
  if(this->_reply->type!=REDIS_REPLY_ARRAY)
    throw std::runtime_error("The reply cannot be indexed "\
                             "because the reply type is " +
                             this->redis_reply_type());
  return CommandReply(this->_reply->element[index]);
}

size_t CommandReply::str_len()
{
  if(this->_reply->type!=REDIS_REPLY_STRING)
    throw std::runtime_error("The length of the reply str "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  return this->_reply->len;
}

size_t CommandReply::n_elements()
{
  if(this->_reply->type!=REDIS_REPLY_ARRAY)
    throw std::runtime_error("The number of elements "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  return this->_reply->elements;
}

int CommandReply::has_error()
{
    int num_errors = 0;
    if(this->_reply->type == REDIS_REPLY_ERROR)
        num_errors++;
    else if(this->_reply->type == REDIS_REPLY_ARRAY) {
        for(size_t i=0; i<this->_reply->elements; i++) {
          CommandReply tmp = (*this)[i];
          num_errors += tmp.has_error();
        }
    }
    return num_errors;
}

void CommandReply::print_reply_error()
{
    if(this->_reply->type == REDIS_REPLY_ERROR) {
        std::string_view error(this->_reply->str,
                               this->_reply->len);
        std::cout<<error<<std::endl;
    }
    else if(this->_reply->type == REDIS_REPLY_ARRAY) {
        for(size_t i=0; i<this->_reply->elements; i++) {
          CommandReply tmp = (*this)[i];
          tmp.print_reply_error();
        }
    }
    return;
}

std::string CommandReply::redis_reply_type()
{
  switch (this->_reply->type) {
    case REDIS_REPLY_STRING:
      return "REDIS_REPLY_STRING";
      break;
    case REDIS_REPLY_ARRAY:
      return "REDIS_REPLY_ARRAY";
      break;
    case REDIS_REPLY_INTEGER:
      return "REDIS_REPLY_INTEGER";
      break;
    case REDIS_REPLY_NIL:
      return "REDIS_REPLY_NIL";
      break;
    case REDIS_REPLY_STATUS:
      return "REDIS_REPLY_STATUS";
      break;
    case REDIS_REPLY_ERROR:
      return "REDIS_REPLY_ERROR";
      break;
    case REDIS_REPLY_DOUBLE:
      return "REDIS_REPLY_DOUBLE";
      break;
    case REDIS_REPLY_BOOL:
      return "REDIS_REPLY_BOOL";
      break;
    case REDIS_REPLY_MAP:
      return "REDIS_REPLY_MAP";
      break;
    case REDIS_REPLY_SET:
      return "REDIS_REPLY_SET";
      break;
    case REDIS_REPLY_ATTR:
      return "REDIS_REPLY_ATTR";
      break;
    case REDIS_REPLY_PUSH:
      return "REDIS_REPLY_PUSH";
      break;
    case REDIS_REPLY_BIGNUM:
      return "REDIS_REPLY_BIGNUM";
      break;
    case REDIS_REPLY_VERB:
      return "REDIS_REPLY_VERB";
      break;
    default:
      throw std::runtime_error("Invalid Redis reply type");
  }
}

void CommandReply::print_reply_structure(std::string index_tracker)
{
  //TODO these recursive functions can't use 'this' unless
  //we have a constructor that takes redisReply*
  std::cout<<index_tracker + " type: "
           <<this->redis_reply_type()<<std::endl;
  switch(this->_reply->type) {
    case REDIS_REPLY_STRING:
      std::cout<<index_tracker + " value: "
               <<std::string(this->str(),
                             this->str_len())
               <<std::endl;
      break;
    case REDIS_REPLY_ARRAY:
      for(size_t i=0; i<this->n_elements(); i++) {
        std::string r_prefix = index_tracker + "[" +
                               std::to_string(i) + "]";
        CommandReply tmp = (*this)[i];
        tmp.print_reply_structure(r_prefix);
      }
      break;
    case REDIS_REPLY_INTEGER:
      std::cout<<index_tracker + " value: "
               <<this->_reply->integer<<std::endl;
      break;
    case REDIS_REPLY_DOUBLE:
      std::cout<<index_tracker + " value: "
               <<this->_reply->dval<<std::endl;
      break;
    case REDIS_REPLY_ERROR:
      std::cout<<index_tracker + " value: "
               <<std::string(this->str(), this->str_len())
               <<std::endl;
      break;
    case REDIS_REPLY_BOOL:
      std::cout<<index_tracker + " value: "
               <<this->_reply->integer<<std::endl;
      break;
    default:
      std::cout<<index_tracker
               <<" value type not supported."<<std::endl;
  }
  return;
}