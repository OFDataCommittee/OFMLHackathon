/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

#include "../../../third-party/catch/single_include/catch2/catch.hpp"
#include "metadata.h"
#include "srexception.h"
#include "logger.h"

unsigned long get_time_offset();

// helper function for checking if the MetaData object was copied correctly
void check_metadata_copied_correctly(MetaData metadata, MetaData metadata_cpy)
{
    double* dbl_data;
    float* flt_data;
    int64_t* int64_data;
    uint64_t* uint64_data;
    int32_t* int32_data;
    uint32_t* uint32_data;
    size_t length;
    SRMetaDataType type;

    double* dbl_data_cpy;
    float* flt_data_cpy;
    int64_t* int64_data_cpy;
    uint64_t* uint64_data_cpy;
    int32_t* int32_data_cpy;
    uint32_t* uint32_data_cpy;
    size_t length_cpy;
    SRMetaDataType type_cpy;
    char** str_data;
    size_t n_strings;
    size_t* lengths;
    char** str_data_cpy;
    size_t n_strings_cpy;
    size_t* lengths_cpy;

    metadata.get_scalar_values("dbl_scalar",
                              (void*&)dbl_data,
                              length, type);
    metadata_cpy.get_scalar_values("dbl_scalar",
                                  (void*&)dbl_data_cpy,
                                  length_cpy, type_cpy);
    CHECK(*dbl_data == *dbl_data_cpy);
    CHECK(length == length_cpy);
    CHECK(type == type_cpy);

    metadata.get_scalar_values("flt_scalar",
                              (void*&)flt_data,
                               length, type);
    metadata_cpy.get_scalar_values("flt_scalar",
                                  (void*&)flt_data_cpy,
                                  length_cpy, type_cpy);
    CHECK(*flt_data == *flt_data_cpy);
    CHECK(length == length_cpy);
    CHECK(type == type_cpy);

    metadata.get_scalar_values("int64_scalar",
                              (void*&)int64_data,
                              length, type);
    metadata_cpy.get_scalar_values("int64_scalar",
                                  (void*&)int64_data_cpy,
                                  length_cpy, type_cpy);
    CHECK(*int64_data == *int64_data_cpy);
    CHECK(length == length_cpy);
    CHECK(type == type_cpy);

    metadata.get_scalar_values("uint64_scalar",
                              (void*&)uint64_data,
                              length, type);
    metadata_cpy.get_scalar_values("uint64_scalar",
                                  (void*&)uint64_data_cpy,
                                  length_cpy, type_cpy);
    CHECK(*uint64_data == *uint64_data_cpy);
    CHECK(length == length_cpy);
    CHECK(type == type_cpy);

    metadata.get_scalar_values("int32_scalar",
                              (void*&)int32_data,
                              length, type);
    metadata_cpy.get_scalar_values("int32_scalar",
                                  (void*&)int32_data_cpy,
                                  length_cpy, type_cpy);
    CHECK(*int32_data == *int32_data_cpy);
    CHECK(length == length_cpy);
    CHECK(type == type_cpy);

    metadata.get_scalar_values("uint32_scalar",
                              (void*&)uint32_data,
                              length, type);
    metadata_cpy.get_scalar_values("uint32_scalar",
                                  (void*&)uint32_data_cpy,
                                  length_cpy, type_cpy);
    CHECK(*uint32_data == *uint32_data_cpy);
    CHECK(length == length_cpy);
    CHECK(type == type_cpy);

    metadata.get_string_values("str_field", str_data,
                                n_strings, lengths);
    metadata_cpy.get_string_values("str_field", str_data_cpy,
                                    n_strings_cpy, lengths_cpy);
    CHECK(std::strcmp(*str_data, *str_data_cpy) == 0);
    CHECK(n_strings == n_strings_cpy);
    CHECK(*lengths == *lengths_cpy);
}

SCENARIO("Test MetaData", "[MetaData]")
{
    std::cout << std::to_string(get_time_offset()) << ": Test MetaData" << std::endl;
    std::string context("test_metadata");
    log_data(context, LLDebug, "***Beginning Metadata testing***");
    GIVEN("A MetaData object")
    {
        MetaData metadata;

        WHEN("Valid scalars are added to the MetaData object")
        {
            std::vector<std::string> keys = {"dbl_scalar", "flt_scalar",
                                             "int64_scalar", "uint64_scalar",
                                             "int32_scalar", "uint32_scalar"};

            double dbl_val = std::numeric_limits<double>::max();
            float flt_val = std::numeric_limits<float>::max();
            int64_t int64_val = std::numeric_limits<int64_t>::max();
            uint64_t uint64_val = std::numeric_limits<uint64_t>::max();
            int32_t int32_val = std::numeric_limits<int32_t>::max();
            uint32_t uint32_val = std::numeric_limits<uint32_t>::max();

            metadata.add_scalar(keys[0], &dbl_val,
                                 SRMetadataTypeDouble);
            metadata.add_scalar(keys[1], &flt_val,
                                 SRMetadataTypeFloat);
            metadata.add_scalar(keys[2], &int64_val,
                                 SRMetadataTypeInt64);
            metadata.add_scalar(keys[3], &uint64_val,
                                 SRMetadataTypeUint64);
            metadata.add_scalar(keys[4], &int32_val,
                                 SRMetadataTypeInt32);
            metadata.add_scalar(keys[5], &uint32_val,
                                 SRMetadataTypeUint32);

            double* dbl_data;
            float* flt_data;
            int64_t* int64_data;
            uint64_t* uint64_data;
            int32_t* int32_data;
            uint32_t* uint32_data;
            size_t length;
            SRMetaDataType type;

            THEN("The scalers can be retrieved correctly")
            {
                metadata.get_scalar_values(keys[0], (void*&)dbl_data,
                                            length, type);
                CHECK(*dbl_data == dbl_val);
                CHECK(length == 1);
                CHECK(type == SRMetadataTypeDouble);
                metadata.get_scalar_values(keys[1], (void*&)flt_data,
                                            length, type);
                CHECK(*flt_data == flt_val);
                CHECK(length == 1);
                CHECK(type == SRMetadataTypeFloat);
                metadata.get_scalar_values(keys[2], (void*&)int64_data,
                                            length, type);
                CHECK(*int64_data == int64_val);
                CHECK(length == 1);
                CHECK(type == SRMetadataTypeInt64);
                metadata.get_scalar_values(keys[3], (void*&)uint64_data,
                                            length, type);
                CHECK(*uint64_data == uint64_val);
                CHECK(length == 1);
                CHECK(type == SRMetadataTypeUint64);
                metadata.get_scalar_values(keys[4], (void*&)int32_data,
                                            length, type);
                CHECK(*int32_data == int32_val);
                CHECK(length == 1);
                CHECK(type == SRMetadataTypeInt32);
                metadata.get_scalar_values(keys[5], (void*&)uint32_data,
                                            length, type);
                CHECK(*uint32_data == uint32_val);
                CHECK(length == 1);
                CHECK(type == SRMetadataTypeUint32);
            }

            AND_THEN("The scalers can be retrieved incorrectly")
            {
                uint32_t* data;
                size_t length;
                SRMetaDataType type;

                INFO("Cannot retrieve a scalar value that does not exist");
                CHECK_THROWS_AS(
                    metadata.get_scalar_values("DNE", (void*&)data,
                                                length, type),
                    RuntimeException);

                INFO("Cannot retrieve a scalar through "
                     "get_string_values method");
                CHECK_THROWS_AS(
                    metadata.get_string_values("uint32_scalar"),
                    RuntimeException);
            }

            AND_THEN("The MetaData object can be copied "
                     "via the copy constructor")
            {
                std::string str_val = "100";
                metadata.add_string("str_field", str_val);

                MetaData metadata_cpy(metadata);

                check_metadata_copied_correctly(metadata, metadata_cpy);
            }

            AND_THEN("The MetaData object can be copied "
                     "via the assignment operator")
            {
                std::string str_val = "100";
                metadata.add_string("str_field", str_val);

                MetaData metadata_cpy;
                metadata_cpy = metadata;

                check_metadata_copied_correctly(metadata, metadata_cpy);
            }

            AND_THEN("The MetaData object can be moved")
            {
                MetaData metadata_2;
                metadata_2 = std::move(metadata);

                metadata_2.get_scalar_values(keys[0], (void*&)dbl_data,
                                            length, type);
                CHECK(*dbl_data == dbl_val);
                CHECK(length == 1);
                CHECK(type == SRMetadataTypeDouble);
                metadata_2.get_scalar_values(keys[1], (void*&)flt_data,
                                            length, type);
                CHECK(*flt_data == flt_val);
                CHECK(length == 1);
                CHECK(type == SRMetadataTypeFloat);
                metadata_2.get_scalar_values(keys[2], (void*&)int64_data,
                                            length, type);
                CHECK(*int64_data == int64_val);
                CHECK(length == 1);
                CHECK(type == SRMetadataTypeInt64);
                metadata_2.get_scalar_values(keys[3], (void*&)uint64_data,
                                            length, type);
                CHECK(*uint64_data == uint64_val);
                CHECK(length == 1);
                CHECK(type == SRMetadataTypeUint64);
                metadata_2.get_scalar_values(keys[4], (void*&)int32_data,
                                            length, type);
                CHECK(*int32_data == int32_val);
                CHECK(length == 1);
                CHECK(type == SRMetadataTypeInt32);
                metadata_2.get_scalar_values(keys[5], (void*&)uint32_data,
                                            length, type);
                CHECK(*uint32_data == uint32_val);
                CHECK(length == 1);
                CHECK(type == SRMetadataTypeUint32);
            }
        }

        AND_WHEN("Invalid scalars are added to the MetaData object")
        {
            std::string str_val = "invalid";

            THEN("Runtime errors are thrown")
            {
                INFO("Cannot add a string with add_scalar method");
                CHECK_THROWS_AS(
                    metadata.add_scalar("str_scalar", &str_val,
                                         SRMetadataTypeString),
                    RuntimeException);
                INFO("The existing metadata field has a different "
                     "type from SRMetadataTypeDouble");
                CHECK_THROWS_AS(
                    metadata.add_scalar("str_scalar", &str_val,
                                         SRMetadataTypeDouble),
                    RuntimeException);
            }
        }

        AND_WHEN("Strings are added to the MetaData object")
        {
            std::string str_val = "100";
            metadata.add_string("str_field", str_val);

            THEN("The string can be retrieved correctly")
            {
                char** str_data;
                size_t n_strings;
                size_t* lengths;
                metadata.get_string_values("str_field", str_data,
                                            n_strings, lengths);
                CHECK(*str_data == str_val);
                CHECK(n_strings == 1);
                CHECK(*lengths == 3);

                std::vector<std::string> string_vals;
                string_vals = metadata.get_string_values("str_field");
                CHECK(string_vals.size() == 1);
                CHECK(string_vals[0] == "100");
            }

            AND_THEN("The string can be retrieved incorrectly")
            {
                std::string* str_data;
                size_t length;
                SRMetaDataType type;
                INFO("A string field cannot be retrieved through "
                     "the get_scalar_values method");
                CHECK_THROWS_AS(
                    metadata.get_scalar_values("str_field", (void*&)str_data,
                                                length, type),
                    RuntimeException);
                INFO("Cannot retrieve a string value that does not exist");
                CHECK_THROWS_AS(
                    metadata.get_string_values("DNE"),
                    RuntimeException
                );

            }

            AND_THEN("The string field can be cleared "
                     "from the MetaData object")
            {
                metadata.clear_field("str_field");
                INFO("The field does not exist once clear_field "
                     "is called on the MetaData object");
                CHECK_THROWS_AS(
                    metadata.get_string_values("str_field"),
                    RuntimeException);
            }
        }
    }
    log_data(context, LLDebug, "***End DBNode testing***");
}