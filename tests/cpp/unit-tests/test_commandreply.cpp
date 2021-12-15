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

#include "../../../third-party/catch/catch.hpp"
#include "commandreply.h"
#include "srexception.h"

using namespace SmartRedis;

/*
*   ----------------------------
*   HELPER FUNCTIONS FOR TESTING
*   ----------------------------
*/

// utility function for quickly creating RedisReplyUPtr
RedisReplyUPtr create_reply_uptr(redisReply* reply)
{
    return std::unique_ptr<redisReply, sw::redis::ReplyDeleter>
           (reply, sw::redis::ReplyDeleter());
}

// utility function for building up a simple rediReply e.g. REDIS_REPLY_INTEGER
void fill_reply_integer(redisReply*& reply, int val)
{
    reply->type = REDIS_REPLY_INTEGER;
    reply->integer = val;
}

// utility function for build up REDIS_REPLY_STRING
// and REDIS_REPLY_DOUBLE and REDIS_REPLY_ERROR
void fill_reply_str(redisReply*& reply, int type, char const* str,
                    size_t len, double val=0.0)
{
    if(type != REDIS_REPLY_STRING &&
       type != REDIS_REPLY_STATUS &&
       type != REDIS_REPLY_DOUBLE &&
       type != REDIS_REPLY_ERROR &&
       type != REDIS_REPLY_BIGNUM &&
       type != REDIS_REPLY_VERB)
        return;
    reply->type = type;
    reply->len = len;
    if(len > 0) {
        reply->str = new char[len];
        std::strcpy(reply->str, str);
        if(type == REDIS_REPLY_DOUBLE)
            reply->dval = val;
    }
    else
        reply->str = NULL;
}

// utility function for building up redisReply of type REDIS_REPLY_ARRAY
void fill_reply_array(redisReply*& reply, int num_of_children)
{
    reply->type = REDIS_REPLY_ARRAY;
    reply->elements = num_of_children;
    if(num_of_children > 0) {
        reply->element = new redisReply*[num_of_children];
        for(size_t i=0; i<num_of_children; i++) {
            // memory has been allocated. Ensure it is filled in
            // a subsequent call to a utility function
            reply->element[i] = new redisReply;
        }
    }
    else
        reply->element = NULL;
}

/*
*   -----
*   TESTS
*   -----
*/

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

        AND_THEN("Various methods will throw errors since the CommandReply "
                 "object doesn't have the correct type for those methods")
        {
            CHECK_THROWS_AS(cmd_reply.str(), RuntimeException);
            CHECK_THROWS_AS(cmd_reply.dbl(), RuntimeException);
            CHECK_THROWS_AS(cmd_reply[1], RuntimeException);
            CHECK_THROWS_AS(cmd_reply.str_len(), RuntimeException);
            CHECK_THROWS_AS(cmd_reply.n_elements(), RuntimeException);
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
            CHECK_THROWS_AS(cmd_reply.integer(), RuntimeException);
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
            CHECK_THROWS_AS(cmd_reply.redis_reply_type(), RuntimeException);
        }
    }
}

SCENARIO("Test CommandReply copy assignment operator and copy "
         "constructor on simple REDIS_REPLY_TYPES", "[CommandReply]")
{

    GIVEN("A CommandReply")
    {
        redisReply* reply = new redisReply;
        fill_reply_integer(reply, 70);
        CommandReply rvalue = std::move(create_reply_uptr(reply));

        WHEN("The CommandReply is copied with the copy constructor")
        {
            CommandReply lvalue(rvalue);

            THEN("The two CommandReply objects have the same structure")
            {
                CHECK(lvalue.redis_reply_type() == rvalue.redis_reply_type());
            }
        }
    }

    AND_GIVEN("Two CommandReplys")
    {
        redisReply* r_reply = new redisReply;
        fill_reply_integer(r_reply, 70);
        CommandReply rvalue = std::move(create_reply_uptr(r_reply));

        redisReply* l_reply = new redisReply;
        l_reply->type = REDIS_REPLY_ARRAY;
        l_reply->elements = 0;
        l_reply->element = NULL;
        CommandReply lvalue = std::move(create_reply_uptr(l_reply));

        WHEN("rvalue is copied into lvalue with the copy assignment operator")
        {
            lvalue = rvalue;

            THEN("The two CommandReply objects have the same structure")
            {
                CHECK(lvalue.integer() == 70);
                CHECK(rvalue.integer() == 70);
            }
        }
    }
}

SCENARIO("Test CommandReply::has_error", "[CommandReply]")
{

    GIVEN("A parent and child redisReply")
    {
        char const* str = "ERR";
        redisReply* parent_reply = new redisReply;
        redisReply* child_reply = new redisReply;
        fill_reply_array(parent_reply, 1);
        fill_reply_str(child_reply, REDIS_REPLY_ERROR, str, 4);
        parent_reply->element[0] = child_reply;

        WHEN("A CommandReply is constructed and copied")
        {
            CommandReply rvalue = std::move(create_reply_uptr(parent_reply));
            CommandReply lvalue = rvalue;

            THEN("The child error is copied correctly")
            {
                CHECK(lvalue.has_error() == 1);
            }
        }
    }

}

SCENARIO("CommandReply copy assignment operator preserves the state of the "
         "rvalue and the lvalue when one of the objects are deleted", "[CommandReply]")
{

    GIVEN("Two dynamically allocated CommandReply. One with a complex "
          "redisReply, and the other with a simple redisReply")
    {
        // create complex CommandReply
        char const* strs[] = {"zero", "one", "two", "three", "four", "five",
                              "six", "seven", "eight", "nine", "ten", "eleven",
                              "twelve", "thirteen", "fourteen", "fifteen"};
        int lens[] = {5, 4, 4, 6, 5, 5, 4, 6, 6, 5, 4, 7, 7, 9, 9, 8};

        /* create a redisReply that has 4 children, each being
           REDIS_REPLY_ARRAY. Each child has four of its own children,
           each being REDIS_REPLY_STR. In total, the base redisReply
           has 16 ancestors. This is a perfect 4-ary tree of height 3. */
        redisReply* reply = new redisReply;
        fill_reply_array(reply, 4);
        for(size_t i=0; i<reply->elements; i++) {
            fill_reply_array(reply->element[i], 4);
            for(size_t j=0; j<reply->element[i]->elements; j++)
                fill_reply_str(reply->element[i]->element[j],
                               REDIS_REPLY_STRING,
                               strs[4*i+j],
                               lens[4*i+j]);
        }
        CommandReply* rvalue = new CommandReply(create_reply_uptr(reply));

        // create simple CommmandReply
        char const* str = "10.0";
        redisReply* simple_reply = new redisReply;
        fill_reply_str(simple_reply, REDIS_REPLY_DOUBLE, str, 5, 10.0);
        CommandReply* lvalue =
            new CommandReply(create_reply_uptr(simple_reply));

        WHEN("The original CommandReply is copied and then deleted")
        {
            *lvalue = *rvalue;
            delete rvalue;

            THEN("The state of the copy is preserved")
            {
                REQUIRE(lvalue->redis_reply_type() == "REDIS_REPLY_ARRAY");
                REQUIRE(lvalue->n_elements() == 4);
                for(size_t i=0; i<lvalue->n_elements(); i++) {
                    CommandReply child = lvalue[0][i];
                    REQUIRE(child.redis_reply_type() == "REDIS_REPLY_ARRAY");
                    REQUIRE(child.n_elements() == 4);
                    for(size_t j=0; j<child.n_elements(); j++) {
                        CommandReply g_child = child[j];
                        REQUIRE(g_child.redis_reply_type() ==
                               "REDIS_REPLY_STRING");
                        CHECK(g_child.str_len() == lens[4*i+j]);
                        CHECK(std::strcmp(g_child.str(), strs[4*i+j]) == 0);
                    }
                }
                delete lvalue;
            }
        }
        WHEN("The original CommandReply is copied "
             "and then the copy is deleted")
        {
            *lvalue = *rvalue;
            delete lvalue;

            THEN("The state of the CommandReply that was copied is preserved")
            {
                REQUIRE(rvalue->redis_reply_type() == "REDIS_REPLY_ARRAY");
                REQUIRE(rvalue->n_elements() == 4);
                for(size_t i=0; i<rvalue->n_elements(); i++) {
                    CommandReply child = rvalue[0][i];
                    REQUIRE(child.redis_reply_type() == "REDIS_REPLY_ARRAY");
                    REQUIRE(child.n_elements() == 4);
                    for(size_t j=0; j<child.n_elements(); j++) {
                        CommandReply g_child = child[j];
                        REQUIRE(g_child.redis_reply_type() ==
                               "REDIS_REPLY_STRING");
                        CHECK(g_child.str_len() == lens[4*i+j]);
                        CHECK(std::strcmp(g_child.str(), strs[4*i+j]) == 0);
                    }
                }
                delete rvalue;
                rvalue = NULL;
            }
        }
    }
}

SCENARIO("Simple tests on CommandReply constructors that use redisReply*", "[CommandReply]")
{

    GIVEN("A redisReply")
    {
        char const* str = "100.0";
        size_t str_len = 6;
        redisReply* reply = new redisReply;
        fill_reply_str(reply, REDIS_REPLY_DOUBLE, str, 6, 100.0);

        WHEN("A CommandReply is constructed with the "
             "redisReply that is later deleted")
        {

            CommandReply cmd_reply(reply);
            delete reply;

            THEN("The state of the CommandReply is preserved")
            {
                REQUIRE(cmd_reply.redis_reply_type() == "REDIS_REPLY_DOUBLE");
                CHECK(cmd_reply.dbl() == 100.0);
                CHECK(cmd_reply.dbl_str() == std::string(str, str_len));
            }
        }

        AND_WHEN("A CommandReply is set equal to a "
                 "redisReply that is later deleted")
        {
            redisReply* simple_reply = new redisReply;
            fill_reply_integer(simple_reply, 5);
            CommandReply cmd_reply(simple_reply);
            delete simple_reply;
            cmd_reply = reply;
            delete reply;

            THEN("The state of the CommandReply is preserved")
            {
                REQUIRE(cmd_reply.redis_reply_type() == "REDIS_REPLY_DOUBLE");
                CHECK(cmd_reply.dbl() == 100.0);
                CHECK(cmd_reply.dbl_str() == std::string(str, str_len));

            }
        }
    }
}

SCENARIO("Test CommandReply copy constructor with an inconsistent redisReply", "[CommandReply]")
{

    GIVEN("An inconsistent redisReply where its 'elements' doesn't "\
          "correspond to its 'element'")
    {

    redisReply* reply = new redisReply;
    reply->type = REDIS_REPLY_ARRAY;
    reply->elements = 5;
    reply->element = NULL;

        WHEN("The CommandReply is constructed with an inconsistent redisReply")
        {

            THEN("An error is thrown during construction")
            {
                CommandReply cmd_reply;
                CHECK_THROWS_AS(cmd_reply = reply, RuntimeException);

                delete reply;
            }
        }
    }
}

SCENARIO("Test CommandReply's redisReply deep copy on a shallow copy", "[CommandReply]")
{

    GIVEN("A CommandReply with redisReply type REDIS_REPLY_ARRAY")
    {
        char const* strs[] = {"zero", "one"};
        int lens[] = {5, 4};
        // reply is a REDIS_REPLY_ARRAY of length 2 where each element is a REDIS_REPLY_STRING
        redisReply* reply = new redisReply;
        fill_reply_array(reply, 2);
        fill_reply_str(reply->element[0], REDIS_REPLY_STRING, strs[0], lens[0]);
        fill_reply_str(reply->element[1], REDIS_REPLY_STRING, strs[1], lens[1]);

        CommandReply* cmd_reply = new CommandReply(create_reply_uptr(reply));

        WHEN("A second CommandReply is constructed by shallowly copying a redisReply from the first CommandReply")
        {
            CommandReply shallow_cmd_reply = (*cmd_reply)[0];
            CHECK(shallow_cmd_reply.str_len() == lens[0]);
            CHECK(std::strcmp(shallow_cmd_reply.str(), strs[0]) == 0);
            CHECK(shallow_cmd_reply.redis_reply_type() == "REDIS_REPLY_STRING");

            THEN("A third CommandReply can deeply copy the shallow copy")
            {
                CommandReply deep_cmd_reply = shallow_cmd_reply;

                // Ensure the deep copy of the shallow copy has the correct data
                CHECK(deep_cmd_reply.str_len() == lens[0]);
                CHECK(std::strcmp(deep_cmd_reply.str(), strs[0]) == 0);

                // Ensure deep_cmd_reply is independent of cmd_reply
                delete cmd_reply;
                CHECK(deep_cmd_reply.str_len() == lens[0]);
                CHECK(std::strcmp(deep_cmd_reply.str(), strs[0]) == 0);
            }
        }
    }
}

SCENARIO("Test CommandReply string retrieval for non REDIS_REPLY_STRING", "[CommandReply]")
{
    char const* strs[] = {"OK", "42.5", "99999999999999999999", "Verbatim string"};
    int lens[] = {3, 5, 21, 16};
    int types[] = {REDIS_REPLY_STATUS, REDIS_REPLY_DOUBLE, REDIS_REPLY_BIGNUM, REDIS_REPLY_VERB};

    WHEN("A redisReply array is populated with strings for DOUBLE, ERROR, BIGNUM, and VERB")
    {
        redisReply* reply_array = new redisReply;
        fill_reply_array(reply_array, 4);
        for (size_t i = 0; i < reply_array->elements; i++)
            fill_reply_str(reply_array->element[i], types[i], strs[i], lens[i]);

        CommandReply* cmd_reply = new CommandReply(create_reply_uptr(reply_array));

        THEN("The strings can be retrieved")
        {

            CHECK(cmd_reply[0][0].status_str() == std::string(strs[0], lens[0]));
            CHECK(cmd_reply[0][1].dbl_str() == std::string(strs[1], lens[1]));
            CHECK(cmd_reply[0][2].bignum_str() == std::string(strs[2], lens[2]));
            CHECK(cmd_reply[0][3].verb_str() == std::string(strs[3], lens[3]));
        }

        AND_THEN("Errors are thrown if string retrieval methods "
                 "are called on an incompatible redisReply type")
        {
            // Calling string retrieval methods on a REDIS_REPLY_ARRAY
            CHECK_THROWS_AS(cmd_reply[0].status_str(), RuntimeException);
            CHECK_THROWS_AS(cmd_reply[0].dbl_str(), RuntimeException);
            CHECK_THROWS_AS(cmd_reply[0].bignum_str(), RuntimeException);
            CHECK_THROWS_AS(cmd_reply[0].verb_str(), RuntimeException);
        }
        delete cmd_reply;
    }
}


SCENARIO("Test REDIS_REPLY_ERROR retrieval from a CommandReply", "[CommandReply]")
{
    /*
            CommanReply (ARRAY)     LEVEL 0
            /    |    \
        ARRAY   DBL   ERR1          LEVEL 1
      /   |  \
    DBL ERR2 ARRAY                  LEVEL 2
               |
              ERR3                  LEVEL 3
    */
    char const* strs[3] = {"ERR1", "ERR2", "ERR3"};
    int str_len = 5;
    double dbl_val = 1998.0;

    GIVEN("A CommandReply with three REDIS_REPLY_ERROR")
    {
        redisReply* reply = new redisReply;
        fill_reply_array(reply, 3);
        // fill LEVEL 1
        fill_reply_array(reply->element[0], 3);
        fill_reply_str(reply->element[1], REDIS_REPLY_DOUBLE, "1998.0", 7, dbl_val);
        fill_reply_str(reply->element[2], REDIS_REPLY_ERROR, strs[0], str_len);
        // fill LEVEL 2
        fill_reply_str(reply->element[0]->element[0], REDIS_REPLY_DOUBLE, "1998.0", 7, dbl_val);
        fill_reply_str(reply->element[0]->element[1], REDIS_REPLY_ERROR, strs[1], str_len);
        fill_reply_array(reply->element[0]->element[2], 1);
        // fill LEVEL 3
        fill_reply_str(reply->element[0]->element[2]->element[0], REDIS_REPLY_ERROR, strs[2], str_len);

        CommandReply* cmd_reply = new CommandReply(create_reply_uptr(reply));

        WHEN("The errors are retrieved")
        {
            std::vector<std::string> errors = cmd_reply->get_reply_errors();

            THEN("The retrieved errors are as expected")
            {
                for (size_t i = 0; i < sizeof(strs)/sizeof(strs[0]); i++)
                    CHECK(errors.at(i) == std::string(strs[i], str_len));
                delete cmd_reply;
            }
        }
    }
}