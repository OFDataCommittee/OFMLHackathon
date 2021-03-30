#ifndef SILC_DATASET_TEST_UTILS_H
#define SILC_DATASET_TEST_UTILS_H

#include <limits>
#include "dataset.h"

namespace DATASET_TEST_UTILS {

const static double dbl_meta_1 =
    std::numeric_limits<double>::max();
const static double dbl_meta_2 =
    std::numeric_limits<double>::min();
const static double dbl_meta_3 = M_PI;

const static float flt_meta_1 =
    std::numeric_limits<float>::max();
const static float flt_meta_2 =
    std::numeric_limits<float>::min();
const static float flt_meta_3 = M_PI;

const static int64_t i64_meta_1 =
    std::numeric_limits<int64_t>::max();
const static int64_t i64_meta_2 =
    std::numeric_limits<int64_t>::min();
const static int64_t i64_meta_3 = (int64_t)M_PI;

const static int32_t i32_meta_1 =
    std::numeric_limits<int32_t>::max();
const static int32_t i32_meta_2 =
    std::numeric_limits<int32_t>::min();
const static int32_t i32_meta_3 = (int32_t)M_PI;

const static uint64_t ui64_meta_1 =
    std::numeric_limits<uint64_t>::max();
const static uint64_t ui64_meta_2 =
    std::numeric_limits<uint64_t>::min();
const static uint64_t ui64_meta_3 = (uint64_t)M_PI;

const static uint32_t ui32_meta_1 =
    std::numeric_limits<uint32_t>::max();
const static uint32_t ui32_meta_2 =
    std::numeric_limits<uint32_t>::min();
const static uint32_t ui32_meta_3 = (uint32_t)M_PI;

const static std::string str_meta_1 =
    std::string("test_meta_string_1");
const static std::string str_meta_2 =
    std::string("test_meta_string_2");
const static std::string str_meta_3 =
    std::string("test_meta_string_3");

/*!
*   \brief This fills a DataSet with metadata
*   that has been constructed to test that the
*   metadata is working correctly.
*   \param dataset The DataSet to fill with metadata
*/
inline void fill_dataset_with_metadata(SmartRedis::DataSet& dataset)
{
    //Add metadata fields to the DataSet.  _meta_1 and _meta_2
    //values added to _field_1 and _meta_3 is added to _field_2.
    dataset.add_meta_scalar("dbl_field_1",
                            &dbl_meta_1,
                            SmartRedis::MetaDataType::dbl);
    dataset.add_meta_scalar("dbl_field_1",
                            &dbl_meta_2,
                            SmartRedis::MetaDataType::dbl);
    dataset.add_meta_scalar("dbl_field_2",
                            &dbl_meta_3,
                            SmartRedis::MetaDataType::dbl);

    dataset.add_meta_scalar("flt_field_1",
                            &flt_meta_1,
                            SmartRedis::MetaDataType::flt);
    dataset.add_meta_scalar("flt_field_1",
                            &flt_meta_2,
                            SmartRedis::MetaDataType::flt);
    dataset.add_meta_scalar("flt_field_2",
                            &flt_meta_3,
                            SmartRedis::MetaDataType::flt);

    dataset.add_meta_scalar("i64_field_1",
                            &i64_meta_1,
                            SmartRedis::MetaDataType::int64);
    dataset.add_meta_scalar("i64_field_1",
                            &i64_meta_2,
                            SmartRedis::MetaDataType::int64);
    dataset.add_meta_scalar("i64_field_2",
                            &i64_meta_3,
                            SmartRedis::MetaDataType::int64);

    dataset.add_meta_scalar("i32_field_1",
                            &i32_meta_1,
                            SmartRedis::MetaDataType::int32);
    dataset.add_meta_scalar("i32_field_1",
                            &i32_meta_2,
                            SmartRedis::MetaDataType::int32);
    dataset.add_meta_scalar("i32_field_2",
                            &i32_meta_3,
                            SmartRedis::MetaDataType::int32);

    dataset.add_meta_scalar("ui64_field_1",
                            &ui64_meta_1,
                            SmartRedis::MetaDataType::uint64);
    dataset.add_meta_scalar("ui64_field_1",
                            &ui64_meta_2,
                            SmartRedis::MetaDataType::uint64);
    dataset.add_meta_scalar("ui64_field_2",
                            &ui64_meta_3,
                            SmartRedis::MetaDataType::uint64);

    dataset.add_meta_scalar("ui32_field_1",
                            &ui32_meta_1,
                            SmartRedis::MetaDataType::uint32);
    dataset.add_meta_scalar("ui32_field_1",
                            &ui32_meta_2,
                            SmartRedis::MetaDataType::uint32);
    dataset.add_meta_scalar("ui32_field_2",
                            &ui32_meta_3,
                            SmartRedis::MetaDataType::uint32);

    dataset.add_meta_string("str_field_1",
                            str_meta_1);
    dataset.add_meta_string("str_field_1",
                            str_meta_2);
    dataset.add_meta_string("str_field_2",
                            str_meta_3);
    return;
}

/*!
*   \brief Check that the provided tensor
*          matches the tensor in the DataSet
*   \param dataset The DataSet containing the tensor
*                  to check
*   \param tensor_name The name of the tensor in the
*                      DataSet
*   \param type The TensorType that corresponds to
*               to the template type T
*   \param vals A pointer to the values to compare
*               to the DataSet tensor.  It is assumed
*               that this is a nested memory space.
*   \param dims The dimensions of the provided vals
*               tensor.  This must be length of 3
*   \tparam T the data type of the vals tensor
*   \throw std::runtime_error if the values do not match
*/
template <typename T>
void check_nested_3D_tensor(SmartRedis::DataSet& dataset,
                            std::string tensor_name,
                            SmartRedis::TensorType type,
                            T*** vals,
                            std::vector<size_t> dims
                            )
{

    T*** t_unpack = allocate_3D_array<T>(dims[0], dims[1], dims[2]);

    dataset.unpack_tensor(tensor_name, t_unpack, dims, type,
                          SmartRedis::MemoryLayout::nested);

    //Check that the tensor values are correct
    if(is_equal_3D_array(vals, t_unpack, dims[0], dims[1], dims[2])) {
        std::cout<<"Correctly fetched " + tensor_name +
                    " with unpack tensor"<<std::endl;
    }
    else {
        throw std::runtime_error(tensor_name +
                                 " did not match the send and "\
                                 "receive values while unpacking");
    }

    free_3D_array(t_unpack, dims[0], dims[1]);

    T*** t_get = allocate_3D_array<T>(dims[0], dims[1], dims[2]);

    std::vector<size_t> get_dims;
    SmartRedis::TensorType get_type;

    dataset.get_tensor(tensor_name, (void*&)t_get,
                       get_dims, get_type,
                       SmartRedis::MemoryLayout::nested);

    if(get_type!=type)
        throw std::runtime_error("Retrieved type for " +
                                 tensor_name +
                                 " does not match known type");

    if(get_dims!=dims)
        throw std::runtime_error("Retrieved dims for " +
                                 tensor_name +
                                 " does not match known dimensions");

    if(is_equal_3D_array(vals, (T***)t_get, dims[0], dims[1], dims[2]))
        std::cout<<"Correctly fetched and allocated "<<tensor_name<<std::endl;
    else
        throw std::runtime_error(tensor_name +
                                 " could not be retrieved "\
                                 "correctly with get_tensor.");

}

/*!
*   \brief Check that the metadata field value
*          matches the provided values.  This method
*          is only meant to be used with scalars,
*          not strings.
*   \param dataset The DataSet containing the fields
*                  to check
*   \param field_name The name of the field to check
*   \param type The MetaDataType that corresponds to
*               to the template type T
*   \param vals A std::vector containing values to compare
*               to the DataSet field values.
*   \tparam T the data type of the vals tensor
*   \throw std::runtime_error if the values do not match
*/
template <typename T>
void check_meta_field(SmartRedis::DataSet& dataset,
                      std::string field_name,
                      SmartRedis::MetaDataType type,
                      std::vector<T> vals)
{
    T* retrieved_vals;
    size_t retrieved_length;
    SmartRedis::MetaDataType retrieved_type;

    dataset.get_meta_scalars(field_name,
                            (void*&)retrieved_vals,
                            retrieved_length,
                            retrieved_type);

    if(retrieved_type!=type)
        throw std::runtime_error("The retrieved type "\
                                "does not match expected "\
                                "value of for field "\
                                + field_name);

    if(retrieved_length!=vals.size())
        throw std::runtime_error("The number of values in field " +
                                field_name + " does not match "\
                                "expected value of " +
                                std::to_string(vals.size()));

    for(size_t i=0; i<vals.size(); i++) {
        T retrieved_val = ((T*)retrieved_vals)[i];
        if((retrieved_val)!=vals[i]) {
        throw std::runtime_error("The " + std::to_string(i)+
                                " value of field " +
                                field_name + " does not match "\
                                "expected value of " +
                                std::to_string(vals[i]) + " . A "\
                                "value of " +
                                std::to_string(retrieved_val) +
                                " was retrieved.");
        }
    }

    std::cout<<"Correct fetched metadata field "
             <<field_name<<std::endl;

    return;
}

/*!
*   \brief This function checks that the tensor
*          names in the DataSet match the known
*          tensor names.  Note that the tensor
*          names are order dependent.
*   \param dataset The DataSet to check
*   \param tensor_names The known tensor names
*   \throw std::runtime_error if the tensor names
*          do not match.
*/
void check_tensor_names(SmartRedis::DataSet& dataset,
                        std::vector<std::string> tensor_names)
{
    std::vector<std::string> names = dataset.get_tensor_names();

    if(names.size()!=tensor_names.size())
        throw std::runtime_error("The corrent number of tensor names "\
                                 "are not contained in the DataSet.");

    for(size_t i=0; i<tensor_names.size(); i++) {
        if(tensor_names[i].compare(names[i])!=0)
            throw std::runtime_error("The provided tensor name " +
                                     tensor_names[i] +
                                     "does not match the retrieved "\
                                     "value of " + names[i]);
    }
    std::cout<<"Correctly fetched metadata tensor names."<<std::endl;
}


/*!
*   \brief This fills a DataSet with metadata
*   that has been constructed to test that the
*   metadata is working correctly.
*   \param dataset The DataSet to fill with metadata
*   \throw std::runtime_error if the metadata field
*          does not match
*/
void check_dataset_metadata(SmartRedis::DataSet& dataset)
{
    //Check that the metadata values are correct for dbl
    check_meta_field<double>(dataset,
                            "dbl_field_1",
                            SmartRedis::MetaDataType::dbl,
                            {dbl_meta_1, dbl_meta_2});

    check_meta_field<double>(dataset,
                            "dbl_field_2",
                            SmartRedis::MetaDataType::dbl,
                            {dbl_meta_3});

    //Check that the metadata values are correct for flt

    check_meta_field<float>(dataset,
                            "flt_field_1",
                            SmartRedis::MetaDataType::flt,
                            {flt_meta_1, flt_meta_2});

    check_meta_field<float>(dataset,
                            "flt_field_2",
                            SmartRedis::MetaDataType::flt,
                            {flt_meta_3});

    //Check that the metadata values are correct for i64

    check_meta_field<int64_t>(dataset,
                                "i64_field_1",
                                SmartRedis::MetaDataType::int64,
                                {i64_meta_1, i64_meta_2});

    check_meta_field<int64_t>(dataset,
                                "i64_field_2",
                                SmartRedis::MetaDataType::int64,
                                {i64_meta_3});

    //Check that the metadata values are correct for i32

    check_meta_field<int32_t>(dataset,
                                "i32_field_1",
                                SmartRedis::MetaDataType::int32,
                                {i32_meta_1, i32_meta_2});

    check_meta_field<int32_t>(dataset,
                                "i32_field_2",
                                SmartRedis::MetaDataType::int32,
                                {i32_meta_3});

    //Check that the metadata values are correct for ui64

    check_meta_field<uint64_t>(dataset,
                                "ui64_field_1",
                                SmartRedis::MetaDataType::uint64,
                                {ui64_meta_1, ui64_meta_2});

    check_meta_field<uint64_t>(dataset,
                                "ui64_field_2",
                                SmartRedis::MetaDataType::uint64,
                                {ui64_meta_3});

    //Check that the metadata values are correct for ui32
    check_meta_field<uint32_t>(dataset,
                                "ui32_field_1",
                                SmartRedis::MetaDataType::uint32,
                                {ui32_meta_1, ui32_meta_2});

    check_meta_field<uint32_t>(dataset,
                                "ui32_field_2",
                                SmartRedis::MetaDataType::uint32,
                                {ui32_meta_3});

    std::vector<std::string> str_meta_field_1 =
    dataset.get_meta_strings("str_field_1");

    if(str_meta_field_1.size()!=2)
        throw std::runtime_error("The number of entries in "\
                                "str_meta_field_1 is incorrect.");
    if(str_meta_1.compare(str_meta_field_1[0])!=0)
        throw std::runtime_error("The retrieved value  "\
                                "for str_meta_1 is incorrect.");
    if(str_meta_2.compare(str_meta_field_1[1])!=0)
        throw std::runtime_error("The retrieved value "\
                                "for str_meta_2 is incorrect.");

    std::vector<std::string> str_meta_field_2 =
        dataset.get_meta_strings("str_field_2");

    if(str_meta_field_2.size()!=1)
        throw std::runtime_error("The number of entries in "\
                                "str_meta_field_2 is incorrect.");
    if(str_meta_3.compare(str_meta_field_2[0])!=0)
        throw std::runtime_error("The retrieved value for "\
                                "str_meta_3 is incorrect.");

    std::cout<<"Correctly fetched string type metadata"<<std::endl;
}

};  //Namespace DATASET_TEST_UTILS

#endif //SILC_DATASET_TEST_UTILS_H