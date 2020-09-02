#ifndef SMARTSIM_COMMANDLIST_H
#define SMARTSIM_COMMANDLIST_H

#include "stdlib.h"
#include <vector>
#include "command.h"

///@file
///\brief The CommandList class for constructing data transfer commands
class CommandList;

class CommandList
{
    private:

        std::vector<Command*> _commands;

    public:

        typedef std::vector<Command*>::iterator iterator;
        typedef std::vector<Command*>::const_iterator const_iterator;

        //! CommandList constructor
        CommandList();

        //! CommandList destructor
        ~CommandList();

        //! Returns a pointer to a new command
        Command* add_command();

        //! Returns an iterator pointing to the first command
        iterator begin();
        //! Returns an const_iterator pointing to the first command
        const_iterator cbegin();
        //! Returns an iterator pointing to the past-the-end command
        iterator end();
        //! Returns an const_iterator pointing to the past-the-end command
        const_iterator cend();

};
#endif //SMARTSIM_COMMANDLIST_H