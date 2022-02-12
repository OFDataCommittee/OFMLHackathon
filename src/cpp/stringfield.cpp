/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

#include "stringfield.h"

using namespace SmartRedis;

// StringField constructor
StringField::StringField(const std::string& name)
 : MetadataField(name, SRMetadataTypeString)
{
    // NOP
}

// StringField constructor with initial values to be copied
StringField::StringField(const std::string& name,
                         const std::vector<std::string>& vals)
 : MetadataField(name, SRMetadataTypeString)
{
    _vals = vals;
}

// StringField constructor with initial values to be copied
StringField::StringField(const std::string& name,
                         std::vector<std::string>&& vals)
 : MetadataField(name, SRMetadataTypeString)
{
    _vals = std::move(vals);
}

// Serialize the StringField for transmission and storage.
std::string StringField::serialize()
{
    return MetadataBuffer::generate_string_buf(_vals);
}

// Add a string to the field
void StringField::append(const std::string& value)
{
    _vals.push_back(value);
}

// Retrieve the number of values in the field
size_t StringField::size()
{
    return _vals.size();
}

// Clear the values in the field
void StringField::clear()
{
    _vals.clear();
}

// Retrieve a copy of the underlying field string values.
std::vector<std::string> StringField::values()
{
    return std::vector<std::string>(_vals);
}

// Returns a constant reference to the internal std::vectorstd::string object.
const std::vector<std::string>& StringField::immutable_values()
{
    return _vals;
}
