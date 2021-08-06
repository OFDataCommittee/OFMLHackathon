#include "../../../third-party/catch/catch.hpp"
#include "commandreply.h"

using namespace SmartRedis;

// utility function for quickly creating RedisReplyUPtr
RedisReplyUPtr create_reply_uptr(redisReply* reply)
{
    return std::unique_ptr<redisReply, sw::redis::ReplyDeleter>
           (reply, sw::redis::ReplyDeleter());
}

SCENARIO("Testing CommandReply object", "[CommandReply]")
{

    GIVEN("A CommandReply object with type REDIS_REPLY_INTEGER")
    {
        redisReply* reply = new redisReply;
        reply->type = REDIS_REPLY_INTEGER;
        reply->integer = 50;
        CommandReply cmd_reply = std::move(create_reply_uptr(reply));

        THEN("The reply type and integer can be retrieved")
        {
            CHECK("REDIS_REPLY_INTEGER" == cmd_reply.redis_reply_type());
            CHECK(reply->integer == cmd_reply.integer());
            cmd_reply.print_reply_structure("0");
        }

        AND_THEN("Various methods will throw errors since the CommandReply"
                 "object doesn't have the correct type for those methods")
        {
            CHECK_THROWS_AS(cmd_reply.str(), std::runtime_error);
            CHECK_THROWS_AS(cmd_reply.dbl(), std::runtime_error);
            CHECK_THROWS_AS(cmd_reply[1], std::runtime_error);
            CHECK_THROWS_AS(cmd_reply.str_len(), std::runtime_error);
            CHECK_THROWS_AS(cmd_reply.n_elements(), std::runtime_error);
        }
    }

    GIVEN("A CommandReply object with type REDIS_REPLY_BOOL")
    {
        redisReply* reply = new redisReply;
        reply->type = REDIS_REPLY_BOOL;
        CommandReply cmd_reply = std::move(create_reply_uptr(reply));

        THEN("The reply type can be retrieved")
        {
            CHECK("REDIS_REPLY_BOOL" == cmd_reply.redis_reply_type());
            cmd_reply.print_reply_structure("1");
        }

        THEN("Cannot call integer method on a REDIS_REPLY_BOOL")
        {
            CHECK_THROWS_AS(cmd_reply.integer(), std::runtime_error);
        }
    }

    GIVEN("Given a CommandReply object without a redis reply type")
    {
        redisReply* reply = new redisReply;
        reply->type = std::numeric_limits<int>::max();
        CommandReply cmd_reply = std::move(create_reply_uptr(reply));

        THEN("An error is thrown when the redis reply"
             "type is attempted to be retrieved")
        {
            CHECK_THROWS_AS(cmd_reply.redis_reply_type(), std::runtime_error);
        }
    }
}