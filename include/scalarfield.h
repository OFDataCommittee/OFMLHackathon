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
        *   \brief MetadataField constructor that
        *          takes in a serialized string to populate values.
        *   \param name The name used to reference the metadata
        *               field
        *   \param serial_string The serialized string containing
        *                        values
        */
        ScalarField(const std::string& name,
                    const std::string_view& serial_string);

        /*!
        *   \brief Default ScalarField copy constructor
        *   \param The scalar field to be copied.
        */
        ScalarField(const ScalarField<T>& scalar_field) = default;

        /*!
        *   \brief Default ScalarField move constructor
        *   \param The scalar field to be moved for construction.
        */
        ScalarField(ScalarField<T>&& scalar_field) = default;

        /*!
        *   \brief Default ScalarField copy assignment operator
        *   \param The scalar field to be copied.
        */
        ScalarField<T>& operator=(const ScalarField<T>& scalar_field)
            = default;

        /*!
        *   \brief Default ScalarField move assignment operator
        *   \param The scalar field to be moved.
        */
        ScalarField<T>& operator=(ScalarField<T>&& scalar_field)
            = default;

        /*!
        *   \brief Default MetadataField destructor
        */
        ~ScalarField() = default;

        /*!
        *   \brief Serialize the ScalarField for
        *          transmission and storage.  The serialized
        *          buffer returned by the ScalarField class
        *          contains the number of field values
        *          followed by the field values.
        *   \param A prefix to attach to the serialized data
        *   \param A suffix to attach to the serialized data
        *   \returns A string of the serialized field
        */
        virtual std::string serialize(const std::string& prefix,
                                      const std::string& suffix);

        /*!
        *   \brief Add a value to the field
        */
        void append(T value);

    private:

        /*!
        *   \brief Unpack the data contained in the buffer string.
        *          The buffer string should point to the location
        *          after any prefix that was added to the buffer.
        *   \param The buffer containing ScalarField data.
        */
        void _unpack(void* buf);

        /*!
        *   \brief The ScalarField values
        */
        std::vector<T> _vals;

};

#include "scalarfield.tcc"

} //namespace SmartRedis

#endif //SMARTREDIS_SCALARFIELD_H