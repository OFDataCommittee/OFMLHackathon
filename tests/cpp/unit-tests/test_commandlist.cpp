#include "../../../third-party/catch/catch.hpp"
#include "commandlist.h"
#include "singlekeycommand.h"
#include "multikeycommand.h"
#include "compoundcommand.h"
#include "addressatcommand.h"
#include "addressanycommand.h"
#include "client.h"

using namespace SmartRedis;

SCENARIO("Testing CommandList object", "[CommandList]")
{

    GIVEN("A CommandList object")
    {
        CommandList cmd_lst;

        WHEN("Different commands with fields are added to the CommandList")
        {
            SingleKeyCommand* sk_cmd = cmd_lst.add_command<SingleKeyCommand>();
            MultiKeyCommand* mk_cmd = cmd_lst.add_command<MultiKeyCommand>();
            CompoundCommand* c_cmd = cmd_lst.add_command<CompoundCommand>();
            AddressAtCommand* aat_cmd = cmd_lst.add_command<AddressAtCommand>();
            AddressAnyCommand* aany_cmd = cmd_lst.add_command<AddressAnyCommand>();

            std::vector<std::string> fields = {"TAG", "BLOB", "TAG", "INFO", "FLUSHALL"};
            sk_cmd->add_field(fields[0], true);
            mk_cmd->add_field(fields[1], true);
            c_cmd->add_field(fields[2], true);
            aat_cmd->add_field(fields[3]);
            aany_cmd->add_field(fields[4]);

            THEN("The commands in the CommandList can be iterated "
                 "over with regular and const iterators")
            {
                int field_index = 0;

                CommandList::iterator it = cmd_lst.begin();
                CommandList::iterator it_end = cmd_lst.end();

                while (it != it_end) {
                    REQUIRE(field_index < fields.size());

                    CHECK((**it).first_field() == fields[field_index]);
                    it++;
                    field_index++;
                }

                field_index = 0;

                CommandList::const_iterator c_it = cmd_lst.cbegin();
                CommandList::const_iterator c_it_end = cmd_lst.cend();

                while (c_it != c_it_end) {
                    REQUIRE(field_index < fields.size());

                    CHECK((**c_it).first_field() == fields[field_index]);
                    c_it++;
                    field_index++;
                }
            }

            AND_THEN("A new CommandList object can be constructed "
                     "with the move constructor")
            {
                CommandList cmd_lst_moved(std::move(cmd_lst));

                int field_index = 0;

                CommandList::const_iterator c_it = cmd_lst_moved.cbegin();
                CommandList::const_iterator c_it_end = cmd_lst_moved.cend();

                // Ensure the cmd_lst has been moved to cmd_lst_moved
                while (c_it != c_it_end) {
                    REQUIRE(field_index < fields.size());

                    CHECK((**c_it).first_field() == fields[field_index]);
                    c_it++;
                    field_index++;
                }

            }

            AND_THEN("A new CommandList object can be constructed "
                     "with the move assignment operator")
            {
                CommandList cmd_lst_moved;
                cmd_lst_moved.add_command<MultiKeyCommand>();

                cmd_lst_moved = std::move(cmd_lst);

                int field_index = 0;

                CommandList::const_iterator c_it = cmd_lst_moved.cbegin();
                CommandList::const_iterator c_it_end = cmd_lst_moved.cend();

                // Ensure the cmd_lst has been moved into cmd_lst_moved
                while (c_it != c_it_end) {
                    REQUIRE(field_index < fields.size());

                    CHECK((**c_it).first_field() == fields[field_index]);
                    c_it++;
                    field_index++;
                }
            }

            AND_THEN("A new CommandList object can be constructed"
                     "with the copy constructor")
            {
                CommandList* cmd_lst_cpy = new CommandList(cmd_lst);

                std::vector<Command*>::const_iterator lst_it =
                    cmd_lst.cbegin();
                std::vector<Command*>::const_iterator lst_it_end =
                    cmd_lst.cend();
                std::vector<Command*>::const_iterator lst_it_cpy =
                    cmd_lst_cpy->cbegin();
                std::vector<Command*>::const_iterator lst_it_end_cpy =
                    cmd_lst_cpy->cend();

                // Ensure the two CommandLists are the same after copying
                while (lst_it != lst_it_end) {
                    if (lst_it_cpy == lst_it_end_cpy)
                        REQUIRE(false);

                    Command::const_iterator cmd_it =
                        (**lst_it).cbegin();
                    Command::const_iterator cmd_it_end =
                        (**lst_it).cend();
                    Command::const_iterator cmd_it_cpy =
                        (**lst_it_cpy).cbegin();
                    Command::const_iterator cmd_it_end_cpy =
                        (**lst_it_cpy).cend();

                    // Ensure each field in the current Command is the same
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

                // Ensure that the state of the original
                // CommandList object is preserved
                int cmd_count = 0;
                lst_it = cmd_lst.cbegin();

                while((lst_it != lst_it_end) && (cmd_count < fields.size())) {
                    CHECK((*lst_it)->first_field() == fields[cmd_count]);
                    lst_it++;
                    cmd_count++;
                }

                CHECK(cmd_count == fields.size());
            }

            AND_THEN("The CommandList object can be copied "
                     "with the assignment operator")
            {
                CommandList* cmd_lst_cpy = new CommandList;
                Command* tmp_cmd = cmd_lst_cpy->add_command<SingleKeyCommand>();
                tmp_cmd->add_field("tmp_field", true);

                *cmd_lst_cpy = cmd_lst;

                std::vector<Command*>::const_iterator lst_it =
                    cmd_lst.cbegin();
                std::vector<Command*>::const_iterator lst_it_end =
                    cmd_lst.cend();
                std::vector<Command*>::const_iterator lst_it_cpy =
                    cmd_lst_cpy->cbegin();
                std::vector<Command*>::const_iterator lst_it_end_cpy =
                    cmd_lst_cpy->cend();

                // Ensure the two CommandLists are the same after copying
                while (lst_it != lst_it_end) {
                    REQUIRE(lst_it_cpy != lst_it_end_cpy);

                    Command::const_iterator cmd_it =
                        (**lst_it).cbegin();
                    Command::const_iterator cmd_it_end =
                        (**lst_it).cend();
                    Command::const_iterator cmd_it_cpy =
                        (**lst_it_cpy).cbegin();
                    Command::const_iterator cmd_it_end_cpy =
                        (**lst_it_cpy).cend();

                    // Ensure each field in the current Commands are the same
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

                // Ensure that the state of the original
                // CommandList object is preserved
                int cmd_count = 0;
                lst_it = cmd_lst.cbegin();

                while((lst_it != lst_it_end) && (cmd_count < fields.size())) {
                    CHECK((*lst_it)->first_field() == fields[cmd_count]);
                    lst_it++;
                    cmd_count++;
                }

                CHECK(cmd_count == fields.size());
            }

            THEN("Fields can be added to the commands")
            {
                CommandList::iterator it = cmd_lst.begin();
                CommandList::iterator it_end = cmd_lst.end();

                int field_index = 0;

                // Verify that the fields have been
                // correctly added to each Command
                while (it != it_end) {
                    REQUIRE(field_index < fields.size());

                    CHECK((**it).first_field() == fields[field_index]);
                    it++;
                    field_index++;
                }
            }
        }
    }
}

SCENARIO("Testing CommandList object on heap", "[CommandList]")
{

    GIVEN("A CommandList object on the heap with three Commands")
    {
        CommandList* cmd_lst = new CommandList;

        SingleKeyCommand* sk_cmd = cmd_lst->add_command<SingleKeyCommand>();
        MultiKeyCommand* mk_cmd = cmd_lst->add_command<MultiKeyCommand>();
        CompoundCommand* c_cmd = cmd_lst->add_command<CompoundCommand>();

        std::vector<std::string> fields = {"TAG", "BLOB", "TAG"};
        sk_cmd->add_field(fields[0], true);
        mk_cmd->add_field(fields[1], true);
        c_cmd->add_field(fields[2], true);

        WHEN("A new CommandList is created usig the assignment "
             "operator and then the original is deleted")
        {
            CommandList* cmd_lst_cpy = new CommandList;
            SingleKeyCommand* tmp_cmd = cmd_lst_cpy->add_command<SingleKeyCommand>();
            tmp_cmd->add_field("tmp_field", true);

            *cmd_lst_cpy = *cmd_lst;

            delete cmd_lst;

            THEN("The state of the new CommandList is preserved")
            {
                std::vector<Command*>::const_iterator it =
                    cmd_lst_cpy->cbegin();
                std::vector<Command*>::const_iterator it_end =
                    cmd_lst_cpy->cend();

                int cmd_count = 0;

                // Ensure that the state of the original
                // CommandList object is preserved
                while((it != it_end) && (cmd_count < fields.size())) {
                    ((*it)->first_field() == fields[cmd_count]);
                    it++;
                    cmd_count++;
                }
                CHECK(cmd_count == fields.size());
            }
        }
    }
}