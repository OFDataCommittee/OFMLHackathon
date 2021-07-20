#include "../../../third-party/catch/catch.hpp"
#include "commandlist.h"

using namespace SmartRedis;

SCENARIO("Testing CommandList object", "[CommandList]")
{
    GIVEN("A CommandList object")
    {
        CommandList cmd_lst;
        WHEN("Commands are added to the CommandList")
        {
            Command* cmd_1 = cmd_lst.add_command();
            Command* cmd_2 = cmd_lst.add_command();
            Command* cmd_3 = cmd_lst.add_command();
            std::vector<std::string> keys = {"AI.TENSORSET", "EXISTS", "DEL"};
            cmd_1->add_field(keys[0], true);
            cmd_2->add_field(keys[1], true);
            cmd_3->add_field(keys[2], true);
            THEN("The commands can be iterated over")
            {
                CommandList::iterator it = cmd_lst.begin();
                CommandList::iterator it_end = cmd_lst.end();
                int i = 0;
                while (it != it_end) {
                    REQUIRE((**it).has_keys() == true);
                    CHECK((**it).first_field() == keys[i]);
                    it++;
                    i++;
                }
                i = 0;
                CommandList::const_iterator c_it = cmd_lst.cbegin();
                CommandList::const_iterator c_it_end = cmd_lst.cend();
                while (c_it != c_it_end) {
                    CHECK((**c_it).has_keys() == true);
                    CHECK((**c_it).first_field() == keys[i]);
                    c_it++;
                    i++;
                }
            }
            AND_THEN("A new CommandList object can be constructed with the move constructor")
            {
                CommandList cmd_lst_moved(std::move(cmd_lst));
                // Ensure the cmd_lst has been moved into cmd_lst_moved
                CommandList::const_iterator c_it = cmd_lst_moved.cbegin();
                CommandList::const_iterator c_it_end = cmd_lst_moved.cend();
                int i = 0;
                while (c_it != c_it_end) {
                    CHECK((**c_it).has_keys() == true);
                    CHECK((**c_it).first_field() == keys[i]);
                    c_it++;
                    i++;
                }

            }
            AND_THEN("A new CommandList object can be constructed with the move assignment operator")
            {
                CommandList cmd_lst_moved;
                cmd_lst_moved.add_command();
                cmd_lst_moved = std::move(cmd_lst);
                // Ensure the cmd_lst has been moved into cmd_lst_moved
                CommandList::const_iterator c_it = cmd_lst_moved.cbegin();
                CommandList::const_iterator c_it_end = cmd_lst_moved.cend();
                int i = 0;
                while (c_it != c_it_end) {
                    CHECK((**c_it).has_keys() == true);
                    CHECK((**c_it).first_field() == keys[i]);
                    c_it++;
                    i++;
                }
            }
            AND_THEN("A new CommandList object can be constructed with the copy constructor")
            {
                CommandList* cmd_lst_cpy = new CommandList(cmd_lst);

                // Ensure the two CommandLists are the same after copying
                std::vector<Command*>::const_iterator lst_it = cmd_lst.cbegin();
                std::vector<Command*>::const_iterator lst_it_end = cmd_lst.cend();
                std::vector<Command*>::const_iterator lst_it_cpy = cmd_lst_cpy->cbegin();
                std::vector<Command*>::const_iterator lst_it_end_cpy = cmd_lst_cpy->cend();
                while (lst_it != lst_it_end) {
                    if (lst_it_cpy == lst_it_end_cpy)
                        REQUIRE(false);
                    // Ensure each field in the current Command is the same
                    Command::const_iterator cmd_it = (**lst_it).cbegin();
                    Command::const_iterator cmd_it_end = (**lst_it).cend();
                    Command::const_iterator cmd_it_cpy = (**lst_it_cpy).cbegin();
                    Command::const_iterator cmd_it_end_cpy = (**lst_it_cpy).cend();
                    while (cmd_it != cmd_it_end) {
                        REQUIRE(cmd_it_cpy != cmd_it_end_cpy);
                        CHECK(*cmd_it == *cmd_it_cpy);
                        cmd_it++;
                        cmd_it_cpy++;
                    }
                    CHECK(cmd_it_cpy == cmd_it_cpy);
                    lst_it++;
                    lst_it_cpy++;
                }
                CHECK(lst_it_cpy == lst_it_end_cpy);

                delete cmd_lst_cpy;

                // Ensure that the state of the original CommandList object is preserved
                int cmd_count = 0;
                lst_it = cmd_lst.cbegin();
                while((lst_it != lst_it_end) && (cmd_count < keys.size())) {
                    CHECK((*lst_it)->first_field() == keys[cmd_count]);
                    lst_it++;
                    cmd_count++;
                }
                CHECK(cmd_count == keys.size());
            }
            AND_THEN("The CommandList object can be copied with the assignment operator")
            {
                CommandList* cmd_lst_cpy = new CommandList;
                Command* tmp_cmd = cmd_lst_cpy->add_command();
                tmp_cmd->add_field("tmp_field", true);
                *cmd_lst_cpy = cmd_lst;

                // Ensure the two CommandLists are the same after copying
                std::vector<Command*>::const_iterator lst_it = cmd_lst.cbegin();
                std::vector<Command*>::const_iterator lst_it_end = cmd_lst.cend();
                std::vector<Command*>::const_iterator lst_it_cpy = cmd_lst_cpy->cbegin();
                std::vector<Command*>::const_iterator lst_it_end_cpy = cmd_lst_cpy->cend();
                while (lst_it != lst_it_end) {
                    REQUIRE(lst_it_cpy != lst_it_end_cpy);
                    // Ensure each field in the current Commands are the same
                    Command::const_iterator cmd_it = (**lst_it).cbegin();
                    Command::const_iterator cmd_it_end = (**lst_it).cend();
                    Command::const_iterator cmd_it_cpy = (**lst_it_cpy).cbegin();
                    Command::const_iterator cmd_it_end_cpy = (**lst_it_cpy).cend();
                    while (cmd_it != cmd_it_end) {
                        REQUIRE(cmd_it_cpy != cmd_it_end_cpy);
                        CHECK(*cmd_it == *cmd_it_cpy);
                        cmd_it++;
                        cmd_it_cpy++;
                    }
                    CHECK(cmd_it_cpy == cmd_it_cpy);
                    lst_it++;
                    lst_it_cpy++;
                }
                CHECK(lst_it_cpy == lst_it_end_cpy);

                delete cmd_lst_cpy;

                // Ensure that the state of the original CommandList object is preserved
                int cmd_count = 0;
                lst_it = cmd_lst.cbegin();
                while((lst_it != lst_it_end) && (cmd_count < keys.size())) {
                    CHECK((*lst_it)->first_field() == keys[cmd_count]);
                    lst_it++;
                    cmd_count++;
                }
                CHECK(cmd_count == keys.size());
            }
            THEN("Fields can be added to the commands")
            {
                CommandList::iterator it = cmd_lst.begin();
                CommandList::iterator it_end = cmd_lst.end();
                it = cmd_lst.begin();
                it_end = cmd_lst.end();
                int i = 0;
                // Verify that the fields have been correctly added to each Command
                while (it != it_end) {
                    CHECK((**it).get_keys().front() == keys.at(i));
                    it++;
                    i++;
                }
            }
        }
    }
    GIVEN("A CommandList object on the heap with three Commands")
    {
        CommandList* cmd_lst = new CommandList;
        Command* cmd_1 = cmd_lst->add_command();
        Command* cmd_2 = cmd_lst->add_command();
        Command* cmd_3 = cmd_lst->add_command();
        std::vector<std::string> keys = {"AI.TENSORSET", "EXISTS", "DEL"};
        cmd_1->add_field(keys[0], true);
        cmd_2->add_field(keys[1], true);
        cmd_3->add_field(keys[2], true);
        WHEN("A new CommandList is created usig the assignment operator and then the original is deleted")
        {
            CommandList* cmd_lst_cpy = new CommandList;
            Command* tmp_cmd = cmd_lst_cpy->add_command();
            tmp_cmd->add_field("tmp_field", true);
            *cmd_lst_cpy = *cmd_lst;
            delete cmd_lst;
            THEN("The state of the new CommandList is preserved")
            {
                // Ensure that the state of the original CommandList object is preserved
                int cmd_count = 0;
                std::vector<Command*>::const_iterator it = cmd_lst_cpy->cbegin();
                std::vector<Command*>::const_iterator it_end = cmd_lst_cpy->cend();
                while((it != it_end) && (cmd_count < keys.size())) {
                    CHECK((*it)->first_field() == keys[cmd_count]);
                    it++;
                    cmd_count++;
                }
                CHECK(cmd_count == keys.size());
            }
        }
    }
}