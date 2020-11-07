#include "commandreply.h"

CommandReply::CommandReply()
{
}

CommandReply::CommandReply(redisReply* reply)
{

  this->_uptr_reply = 0;
  this->_reply = reply;
}

CommandReply::~CommandReply()
{
}

CommandReply::CommandReply(RedisReplyUPtr&& reply)
{
  /* Move constructor with redisReply unique_ptr
  */
  this->_uptr_reply = std::move(reply);
  this->_reply = this->_uptr_reply.get();
  return;
}

CommandReply::CommandReply(redisReply*&& reply)
{
  /* Move constructor with redisReply ptr
  */
  this->_uptr_reply = 0;
  this->_reply = std::move(reply);
  return;
}

CommandReply::CommandReply(CommandReply&& reply)
{
  /* Move constructor with CommandReply
  */
  this->_uptr_reply = std::move(reply._uptr_reply);
  this->_reply = this->_uptr_reply.get();
  return;
}

CommandReply& CommandReply::operator=(RedisReplyUPtr&& reply)
{
    /* Move assignment operator with redisReply unique_ptr
    */
    this->_uptr_reply = std::move(reply);
    this->_reply = this->_uptr_reply.get();
    return *this;
}

CommandReply& CommandReply::operator=(redisReply*&& reply)
{
    /* Move assignment operator with redisReply ptr
    */
    this->_uptr_reply = 0;
    this->_reply = std::move(reply);
    return *this;
}

CommandReply& CommandReply::operator=(CommandReply&& reply)
{
  /* Move assignment operator with CommandReply
  */
  if(this!=&reply) {
    this->_uptr_reply = std::move(reply._uptr_reply);
    this->_reply = this->_uptr_reply.get();
  }
  return *this;
}

char* CommandReply::str()
{
  /* This function returns a pointer to the
  str reply.  If the reply is not a string
  type, an error will be thrown.
  */
  if(this->_reply->type!=REDIS_REPLY_STRING)
    throw std::runtime_error("A pointer to the reply str "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  return this->_reply->str;
}

long long CommandReply::integer()
{
  /* This function returns a pointer to the
  str reply.  If the reply is not a string
  type, an error will be thrown.
  */
  if(this->_reply->type!=REDIS_REPLY_INTEGER)
    throw std::runtime_error("The reply integer "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  return this->_reply->integer;
}

double CommandReply::dbl()
{
  /* This function returns a pointer to the
  str reply.  If the reply is not a string
  type, an error will be thrown.
  */
  if(this->_reply->type!=REDIS_REPLY_DOUBLE)
    throw std::runtime_error("The reply double "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  return this->_reply->dval;
}

CommandReply CommandReply::operator[](int index)
{
  /* This function returns the redisReply in the
  index position of the elements array.
  */
  if(this->_reply->type!=REDIS_REPLY_ARRAY)
    throw std::runtime_error("The reply cannot be indexed "\
                             "because the reply type is " +
                             this->redis_reply_type());
  return CommandReply(this->_reply->element[index]);
}

size_t CommandReply::str_len()
{
  /* This function returns the len of the
  str reply.  If the reply is not a string
  type, an error will be thrown.
  */
  if(this->_reply->type!=REDIS_REPLY_STRING)
    throw std::runtime_error("The length of the reply str "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  return this->_reply->len;
}

size_t CommandReply::n_elements()
{
  /* This function returns the number of elements
  in the redis reply
  */
  if(this->_reply->type!=REDIS_REPLY_ARRAY)
    throw std::runtime_error("The number of elements "\
                             "cannot be returned because the "\
                             "the reply type is " +
                             this->redis_reply_type());
  return this->_reply->elements;
}

int CommandReply::has_error()
{
    /* This function checks to see if the reply or any
    of the sub replies contains an error.  The number
    of errors is returned.
    */
    int num_errors = 0;
    if(this->_reply->type == REDIS_REPLY_ERROR)
        num_errors++;
    else if(this->_reply->type == REDIS_REPLY_ARRAY) {
        for(int i=0; i<this->_reply->elements; i++) {
          CommandReply tmp = (*this)[i];
          num_errors += tmp.has_error();
        }
    }
    return num_errors;
}

void CommandReply::print_reply_error()
{
    /* This function will print error replies
    and sub element error replies to std::cout
    */
    if(this->_reply->type == REDIS_REPLY_ERROR) {
        std::string_view error(this->_reply->str,
                               this->_reply->len);
        std::cout<<error<<std::endl;
    }
    else if(this->_reply->type == REDIS_REPLY_ARRAY) {
        for(int i=0; i<this->_reply->elements; i++) {
          CommandReply tmp = (*this)[i];
          tmp.print_reply_error();
        }
    }
    return;
}

std::string CommandReply::redis_reply_type()
{
  /* This function returns the redis reply type as a string
  */
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
  /* This function prints out the nested reply structure
  */
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
      for(int i=0; i<this->n_elements(); i++) {
        std::string r_prefix = index_tracker + "[" + std::to_string(i) + "]";
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