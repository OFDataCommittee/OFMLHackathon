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

#ifndef SMARTREDIS_SCALARFIELD_H
#define SMARTREDIS_SCALARFIELD_H

#include "metadatafield.h"
#include <iostream>
namespace SmartRedis {

/*!
*   \brief  The ScalarField class implements
*   MetadataField class methods needed for
*   storage and transfer of metadata scalar
*   fields (e.g. double, float, int64, etc.)
*/
template <class T>
class ScalarField : public MetadataField {

    public:

        /*!
        *   \brief ScalarField constructor
        *   \param name The name used to reference the scalar field
        *   \param MetaDataType The metadata type for this field
        */
        ScalarField(const std::string& name, MetaDataType type);

        /*!
        *   \brief ScalarField constructor with initial values
        *   \param name The name used to reference the scalar field
        *   \param MetaDataType The metadata type for this field
        *   \param vals Initial values to be copied into the ScalarField
        */
        ScalarField(const std::string& name,
                    MetaDataType type,
                    const std::vector<T>& vals);

        /*!
        *   \brief ScalarField constructor with initial values
        *   \param name The name used to reference the scalar field
        *   \param MetaDataType The metadata type for this field
        *   \param vals Initial values to be placed in the Scalarfield
        *               via move semantics
        */
        ScalarField(const std::string& name,
                    MetaDataType type,
                    std::vector<T>&& vals);

        /*!
        *   \brief Default ScalarField copy constructor
        *   \param scalar_field The scalar field to be copied.
        */
        ScalarField(const ScalarField<T>& scalar_field) = default;

        /*!
        *   \brief Default ScalarField move constructor
        *   \param scalar_field The scalar field to be moved for construction.
        */
        ScalarField(ScalarField<T>&& scalar_field) = default;

        /*!
        *   \brief Default ScalarField copy assignment operator
        *   \param scalar_field The scalar field to be copied.
        */
        ScalarField<T>& operator=(const ScalarField<T>& scalar_field)
            = default;

        /*!
        *   \brief Default ScalarField move assignment operator
        *   \param scalar_field The scalar field to be moved.
        */
        ScalarField<T>& operator=(ScalarField<T>&& scalar_field)
            = default;

        /*!
        *   \brief Default MetadataField destructor
        */
        virtual ~ScalarField() = default;

        /*!
        *   \brief Serialize the ScalarField for
        *          transmission and storage.
        *   \returns A string of the serialized field
        */
        virtual std::string serialize();

        /*!
        *   \brief Add a value to the field
        *   \param value A c-ptr to the value to append
        */
        void append(const void* value);

        /*!
        *   \brief Retrieve the number of values in the field
        *   \returns The number of values
        */
        virtual size_t size();

        /*!
        *   \brief Clear the values in the field
        */
        virtual void clear();

        /*!
        *   \brief Returns a c-ptr to the underlying data.  This
        *          pointer is not valid if any other operation
        *          such as append() is performed.
        *   \returns A c-ptr to the underlying data
        */
        void* data();

        /*!
        *   \brief Returns a constant reference to the internal
        *          std::vector<T> object.
        *   \returns const reference to std::vector<T>
        *            of values
        */
        const std::vector<T>& immutable_values();

    private:

        /*!
        *   \brief The ScalarField values
        */
        std::vector<T> _vals;

};

#include "scalarfield.tcc"

} //namespace SmartRedis

#endif //SMARTREDIS_SCALARFIELD_H