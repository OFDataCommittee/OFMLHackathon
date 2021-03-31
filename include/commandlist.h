#ifndef SMARTREDIS_COMMANDLIST_H
#define SMARTREDIS_COMMANDLIST_H

#include "stdlib.h"
#include <vector>
#include "command.h"

namespace SmartRedis {

class CommandList;

//@file
/*!
*   \brief The CommandList class constructs multiple Client
*          Command.
*   \details CommandList handles the dynamic allocation of a new
*            message in the list and provides iterators
*            for iterating over Command.
*/
class CommandList
{
    public:

        /*!
        *   \brief Default CommandList constructor
        */
        CommandList() = default;

        /*!
        *   \brief Default CommandList destructor
        */
        ~CommandList();

        /*!
        *   \brief Dynamically allocate a new Command
        *          and return a pointer to the new Command
        *   \returns Pointer to a new Command
        */
        Command* add_command();

        /*!
        *   \typedef An iterator type for iterating
        *            over all Commands
        */
        typedef std::vector<Command*>::iterator iterator;

        /*!
        *   \typedef A const iterator type for iterating
        *            over all Commands
        */
        typedef std::vector<Command*>::const_iterator const_iterator;

        /*!
        *   \brief Returns an iterator pointing to the
        *          first Command
        *   \returns CommandList iterator to the first Command
        */
        iterator begin();

        /*!
        *   \brief Returns a const iterator pointing to the
        *          first Command
        *   \returns Const CommandList iterator to the first Command
        */
        const_iterator cbegin();

        /*!
        *   \brief Returns an iterator pointing to the
        *          past-the-end Command
        *   \returns CommandList iterator to the past-the-end Command
        */
        iterator end();

        /*!
        *   \brief Returns a const iterator pointing to the
        *          past-the-end Command
        *   \returns Const CommandList iterator to the past-the-end Command
        */
        const_iterator cend();

    private:

        /*!
        *   \brief A vector container a pointer to all Command
        */
        std::vector<Command*> _commands;


};

} //namespace SmartRedis

#endif //SMARTREDIS_COMMANDLIST_H
