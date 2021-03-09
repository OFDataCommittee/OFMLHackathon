/* Defines the entity types
for c-clients to use as a
type specifier
*/

#ifndef SMARTSIM_CENTITYTYPE_H
#define SMARTSIM_CENTITYTYPE_H

typedef enum{
    c_tensor = 1,
    c_model  = 2,
    c_dataset= 3,
}CEntityType;

#endif //SMARTSIM_CENTITYTYPE_H
