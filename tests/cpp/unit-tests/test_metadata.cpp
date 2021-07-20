#include "../../../third-party/catch/catch.hpp"
#include "metadata.h"

SCENARIO("Test MetaData", "[MetaData]")
{
    GIVEN("A MetaData object")
    {
        MetaData metadata;
        WHEN("Valid scalars are added to the MetaData object")
        {
            double dbl_val = std::numeric_limits<double>::max();
            float flt_val = std::numeric_limits<float>::max();
            int64_t int64_val = std::numeric_limits<int64_t>::max();
            uint64_t uint64_val = std::numeric_limits<uint64_t>::max();
            int32_t int32_val = std::numeric_limits<int32_t>::max();
            uint32_t uint32_val = std::numeric_limits<uint32_t>::max();
            metadata.add_scalar("dbl_scalar", &dbl_val, MetaDataType::dbl);
            metadata.add_scalar("flt_scalar", &flt_val, MetaDataType::flt);
            metadata.add_scalar("int64_scalar", &int64_val, MetaDataType::int64);
            metadata.add_scalar("uint64_scalar", &uint64_val, MetaDataType::uint64);
            metadata.add_scalar("int32_scalar", &int32_val, MetaDataType::int32);
            metadata.add_scalar("uint32_scalar", &uint32_val, MetaDataType::uint32);

            double* dbl_data;
            float* flt_data;
            int64_t* int64_data;
            uint64_t* uint64_data;
            int32_t* int32_data;
            uint32_t* uint32_data;
            size_t length;
            MetaDataType type;
            THEN("The scalers can be retrieved correctly")
            {
                metadata.get_scalar_values("dbl_scalar", (void*&)dbl_data, length, type);
                CHECK(*dbl_data == dbl_val);
                CHECK(length == 1);
                CHECK(type == MetaDataType::dbl);
                metadata.get_scalar_values("flt_scalar", (void*&)flt_data, length, type);
                CHECK(*flt_data == flt_val);
                CHECK(length == 1);
                CHECK(type == MetaDataType::flt);
                metadata.get_scalar_values("int64_scalar", (void*&)int64_data, length, type);
                CHECK(*int64_data == int64_val);
                CHECK(length == 1);
                CHECK(type == MetaDataType::int64);
                metadata.get_scalar_values("uint64_scalar", (void*&)uint64_data, length, type);
                CHECK(*uint64_data == uint64_val);
                CHECK(length == 1);
                CHECK(type == MetaDataType::uint64);
                metadata.get_scalar_values("int32_scalar", (void*&)int32_data, length, type);
                CHECK(*int32_data == int32_val);
                CHECK(length == 1);
                CHECK(type == MetaDataType::int32);
                metadata.get_scalar_values("uint32_scalar", (void*&)uint32_data, length, type);
                CHECK(*uint32_data == uint32_val);
                CHECK(length == 1);
                CHECK(type == MetaDataType::uint32);
            }
            AND_THEN("The scalers can be retrieved incorrectly")
            {
                uint32_t* data;
                size_t length;
                MetaDataType type;
                INFO("Cannot retrieve a scalar value that does not exist");
                CHECK_THROWS_AS(
                    metadata.get_scalar_values("DNE", (void*&)data, length, type),
                    std::runtime_error
                );
                INFO("Cannot retrieve a scalar through get_string_values method");
                CHECK_THROWS_AS(
                    metadata.get_string_values("uint32_scalar"),
                    std::runtime_error
                );
            }
            AND_THEN("The MetaData object can be copied via the copy constructor")
            {
                double* dbl_data_cpy;
                float* flt_data_cpy;
                int64_t* int64_data_cpy;
                uint64_t* uint64_data_cpy;
                int32_t* int32_data_cpy;
                uint32_t* uint32_data_cpy;
                size_t length_cpy;
                MetaDataType type_cpy;
                char** str_data;
                size_t n_strings;
                size_t* lengths;
                char** str_data_cpy;
                size_t n_strings_cpy;
                size_t* lengths_cpy;

                std::string str_val = "100";
                metadata.add_string("str_field", str_val);
                MetaData metadata_cpy(metadata);

                metadata.get_scalar_values("dbl_scalar", (void*&)dbl_data, length, type);
                metadata_cpy.get_scalar_values("dbl_scalar", (void*&)dbl_data_cpy, length_cpy, type_cpy);
                CHECK(*dbl_data == *dbl_data_cpy);
                CHECK(length == length_cpy);
                CHECK(type == type_cpy);
                metadata.get_scalar_values("flt_scalar", (void*&)flt_data, length, type);
                metadata_cpy.get_scalar_values("flt_scalar", (void*&)flt_data_cpy, length_cpy, type_cpy);
                CHECK(*flt_data == *flt_data_cpy);
                CHECK(length == length_cpy);
                CHECK(type == type_cpy);
                metadata.get_scalar_values("int64_scalar", (void*&)int64_data, length, type);
                metadata_cpy.get_scalar_values("int64_scalar", (void*&)int64_data_cpy, length_cpy, type_cpy);
                CHECK(*int64_data == *int64_data_cpy);
                CHECK(length == length_cpy);
                CHECK(type == type_cpy);
                metadata.get_scalar_values("uint64_scalar", (void*&)uint64_data, length, type);
                metadata_cpy.get_scalar_values("uint64_scalar", (void*&)uint64_data_cpy, length_cpy, type_cpy);
                CHECK(*uint64_data == *uint64_data_cpy);
                CHECK(length == length_cpy);
                CHECK(type == type_cpy);
                metadata.get_scalar_values("int32_scalar", (void*&)int32_data, length, type);
                metadata_cpy.get_scalar_values("int32_scalar", (void*&)int32_data_cpy, length_cpy, type_cpy);
                CHECK(*int32_data == *int32_data_cpy);
                CHECK(length == length_cpy);
                CHECK(type == type_cpy);
                metadata.get_scalar_values("uint32_scalar", (void*&)uint32_data, length, type);
                metadata_cpy.get_scalar_values("uint32_scalar", (void*&)uint32_data_cpy, length_cpy, type_cpy);
                CHECK(*uint32_data == *uint32_data_cpy);
                CHECK(length == length_cpy);
                CHECK(type == type_cpy);
                metadata.get_string_values("str_field", str_data, n_strings, lengths);
                metadata_cpy.get_string_values("str_field", str_data_cpy, n_strings_cpy, lengths_cpy);
                CHECK(std::strcmp(*str_data, *str_data_cpy) == 0);
                CHECK(n_strings == n_strings_cpy);
                CHECK(*lengths == *lengths_cpy);
            }
            AND_THEN("The MetaData object can be copied via the assignment operator")
            {
                MetaData medadata_2;
                medadata_2 = metadata;
                // TODO: Ensure it was copied correctly
            }
            AND_THEN("The MetaData object can be moved")
            {
                MetaData metadata_2;
                metadata_2 = std::move(metadata);
                // TODO: Ensure it was moved correctly
            }
        }
        AND_WHEN("Invalid scalars are added to the MetaData object")
        {
            std::string str_val = "invalid";
            THEN("Runtime errors are thrown")
            {
                INFO("Cannot add a string with add_scalar method");
                CHECK_THROWS_AS(
                    metadata.add_scalar("str_scalar", &str_val, MetaDataType::string),
                    std::runtime_error
                );
                INFO("The existing metadata field has a different type from MetaDataType::dbl");
                CHECK_THROWS_AS(
                    metadata.add_scalar("str_scalar", &str_val, MetaDataType::dbl),
                    std::runtime_error
                );
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
                metadata.get_string_values("str_field", str_data, n_strings, lengths);
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
                MetaDataType type;
                INFO("A string field cannot be retrieved through the get_scalar_values method");
                CHECK_THROWS_AS(
                    metadata.get_scalar_values("str_field", (void*&)str_data, length, type),
                    std::runtime_error
                );
                INFO("Cannot retrieve a string value that does not exist");
                CHECK_THROWS_AS(
                    metadata.get_string_values("DNE"),
                    std::runtime_error
                );

            }
            AND_THEN("The string field can be cleared from the MetaData object")
            {
                metadata.clear_field("str_field");
                INFO("The field does not exist once clear_field is called on the MetaData object");
                CHECK_THROWS_AS(
                    metadata.get_string_values("str_field"),
                    std::runtime_error
                );
            }
        }
        AND_WHEN("Valid serialized fields are added to the MetaData object")
        {
            std::string name = "serialized_field";
            char buf[4] = "buf";
            size_t buf_size = std::strlen(buf);
            THEN("The serialized fields can be retrieved")
            {
            }
            THEN("Cannot add a serialized field with a name that already exists")
            {
                INFO("Cannot add a serialized field that already exists");
            }
        }
    }
}