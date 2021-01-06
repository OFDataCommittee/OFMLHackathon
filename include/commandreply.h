#ifndef SMARTSIM_COMMANDREPLY_H
#define SMARTSIM_COMMANDREPLY_H

#include "stdlib.h"
#include <sw/redis++/redis++.h>
#include <iostream>

namespace SILC {

class CommandReply;

typedef std::unique_ptr<redisReply, sw::redis::ReplyDeleter>
        RedisReplyUPtr;

///@file
///\brief This class stores and processes Command replies
class CommandReply {

    public:
        //! CommandReply default constructor
        CommandReply();

        //!Command reply destructor
        ~CommandReply();

        //Move constructor with redisReply unique ptr
        CommandReply(RedisReplyUPtr&& reply);

        //Move constructor with redisReply ptr
        CommandReply(redisReply*&& reply);

        //Move constructor with CommandReply
        CommandReply(CommandReply&& reply);

        //! Move assignment operator with RHS as redisReply unique ptr
        CommandReply& operator=(RedisReplyUPtr&& reply
                       );

        //! Move assignment operator with RHS as redisReply ptr
        CommandReply& operator=(redisReply*&& reply);

        // //! Move assignment operator with RHS as CommandReply
        CommandReply& operator=(CommandReply&& reply);

        CommandReply operator[](int index);

        //! Return the reply string
        char* str();

        //! Return the integer reply
        long long integer();

        //! Return the double reply
        double dbl();

        //! Return the len of the string reply
        size_t str_len();

        //! Return the number of elements in a reply
        size_t n_elements();

        //! Check to see if the reply contains any errors
        int has_error();

        //! Print error reply text to std::cout
        void print_reply_error();

        //! Return the reply type as a string
        std::string redis_reply_type();

        //! Print the type of each element in the reply
        void print_reply_structure(std::string index_tracker="reply[0]");


    private:

        //! CommandReply constructor from redisReply
        CommandReply(redisReply* reply);

        RedisReplyUPtr _uptr_reply;
        redisReply* _reply;

        //! Helper function to print nested reply structures
        void _print_nested_reply_structure(redisReply* reply,
                                           std::string index_tracker);
};

} //namespace SILC

#endif //SMARTSIM_COMMANDREPLY_H