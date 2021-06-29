#include "catch.hpp"
#include "commandlist.h"

using namespace SmartRedis;

SCENARIO("Testing CommandList object", "[CommandList]")
{
    GIVEN("A CommandList object")
    {
        CommandList cmd_lst;
        WHEN("Commands are added to the CommandList")
        {
            cmd_lst.add_command();
            cmd_lst.add_command();
            cmd_lst.add_command();
            THEN("The commands can be iterated over")
            {
                CommandList::iterator it = cmd_lst.begin();
                CommandList::iterator it_end = cmd_lst.end();
                while (it != it_end) {
                    CHECK((**it).has_keys() == false);
                    it++;
                }
                CommandList::const_iterator c_it = cmd_lst.cbegin();
                CommandList::const_iterator c_it_end = cmd_lst.cend();
                while (c_it != c_it_end) {
                    CHECK((**c_it).has_keys() == false);
                    c_it++;
                }
            }
            THEN("Fields can be added to the commands")
            {
                std::vector<std::string> keys = {"AI.TENSORSET", "EXISTS", "DEL"};
                CommandList::iterator it = cmd_lst.begin();
                CommandList::iterator it_end = cmd_lst.end();
                int i = 0;
                // Add fields to each Command in the CommandList
                while (it != it_end) {
                    Command& cmd = **(it);
                    cmd.add_field(keys.at(i), true);
                    it++;
                    i++;
                }
                it = cmd_lst.begin();
                it_end = cmd_lst.end();
                i = 0;
                // Verify that the fields have been correctly added to each Command
                while (it != it_end) {
                    CHECK((**it).get_keys().front() == keys.at(i));
                    it++;
                    i++;
                }
            }
        }
    }
}