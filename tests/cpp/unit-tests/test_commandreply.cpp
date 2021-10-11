#include "../../../third-party/catch/catch.hpp"
#include "commandreply.h"

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
       type != REDIS_REPLY_DOUBLE &&
       type != REDIS_REPLY_ERROR)
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
                // uncomment the following lines once CommandReply::str
                // and CommandReply::str_len are fixed
                // CHECK(cmd_reply.str_len() == 6);
                // CHECK(std::strcmp(cmd_reply.str(), str) == 0);
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
                // uncomment the following lines once CommandReply::str
                // and CommandReply::str_len are fixed
                // CHECK(cmd_reply.str_len() == 6);
                // CHECK(std::strcmp(cmd_reply.str(), str) == 0);

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
                CHECK_THROWS_AS(cmd_reply = reply, std::runtime_error);

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