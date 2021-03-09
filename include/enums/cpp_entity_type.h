/* The EntityType enum specifies
the type of the entity.  This
enum should be used by users.
*/

#ifndef SMARTSIM_ENTITYTYPE_H
#define SMARTSIM_ENTITYTYPE_H

#include "enums/c_entity_type.h"

namespace SILC {

enum class EntityType{
    tensor = 1,
    model  = 2,
    dataset= 3};

//! Helper method to convert between CEntityType and EntityType
inline EntityType convert_entity_type(CEntityType type) {
    EntityType t;
    switch(type) {
        case CEntityType::c_tensor :
            t = EntityType::tensor;
            break;
        case CEntityType::c_model :
            t = EntityType::model;
            break;
        case CEntityType::c_dataset :
            t = EntityType::dataset;
            break;
        default :
            throw std::runtime_error("Error converting CEntityType "\
                                     "to EntityType.");
    }
    return t;
}

//! Helper method to convert between CTensorType and TensorType
inline CEntityType convert_entity_type(EntityType type) {
    CEntityType t;
    switch(type) {
        case EntityType::tensor :
            t = CEntityType::c_tensor;
            break;
        case EntityType::model :
            t = CEntityType::c_model;
            break;
        case EntityType::dataset :
            t = CEntityType::c_dataset;
            break;
        default :
            throw std::runtime_error("Error converting TensorType "\
                                     "to CTensorType.");
    }
    return t;

}

} //namespace SILC

#endif //SMARTSIM_ENTITYTYPE_H
