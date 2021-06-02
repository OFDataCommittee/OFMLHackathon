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

#ifndef SMARTREDIS_METADATAFIELD_H
#define SMARTREDIS_METADATAFIELD_H

#include <vector>
#include <string>
#include "enums/cpp_metadata_type.h"

namespace SmartRedis {

class MetadataField;

/*!
*   \brief  Abstract base class for interfacing with
*           different metadata field types.
*/
class MetadataField {

    public:

    /*!
    *   \brief MetadataField constructor
    *   \param name The name used to reference the metadata field
    *   \param MetaDataType The metadata type for this field
    */
    MetadataField(const std::string& name, MetaDataType type);

    /*!
    *   \brief Default MetadataField destructor
    */
    virtual ~MetadataField() = default;

    /*!
    *   \brief Retrieve a serialized version of the field
    *   \returns A std::string containing the serialized field
    */
    virtual std::string serialize() = 0;

    /*!
    *   \brief Retrieve the MetadataField name
    *   \returns MetadataField name
    */
    std::string name();

    /*!
    *   \brief Retrieve the Metadatafield type
    *   \returns MetadataField type
    */
    MetaDataType type();

    /*!
    *   \brief Retrieve the number of values in the field
    *   \returns The number of values
    */
    virtual size_t size() = 0;

    /*!
    *   \brief Clear the values in the field
    */
    virtual void clear() = 0;

    protected:

    /*!
    *   \brief The name of the field.
    */
    std::string _name;

    /*!
    *   \brief The field type.
    */
    MetaDataType _type;

};


}  //namespace SmartRedis



#endif //SMARTREDIS_METADATAFIELD_H