#include "client.h"
#include "dataset.h"
#include "client_test_utils.h"
#include "dataset_test_utils.h"

template <typename T_send, typename T_recv>
void put_get_3D_array(
		    void (*fill_array)(T_send***, int, int, int),
		    std::vector<size_t> dims,
        SmartRedis::TensorType type,
        std::string key_suffix,
        std::string dataset_name)
{
    T_send*** t_send_1 =
        allocate_3D_array<T_send>(dims[0], dims[1], dims[2]);
    fill_array(t_send_1, dims[0], dims[1], dims[2]);

    T_send*** t_send_2 =
        allocate_3D_array<T_send>(dims[0], dims[1], dims[2]);
    fill_array(t_send_2, dims[0], dims[1], dims[2]);

    T_send*** t_send_3 =
        allocate_3D_array<T_send>(dims[0], dims[1], dims[2]);
    fill_array(t_send_3, dims[0], dims[1], dims[2]);

    //Create Client and DataSets
    SmartRedis::Client client(use_cluster());
    SmartRedis::DataSet* dataset = new SmartRedis::DataSet(dataset_name);

    //Add tensors to the DataSet
    std::string t_name_1 = "tensor_1";
    std::string t_name_2 = "tensor_2";
    std::string t_name_3 = "tensor_3";

    dataset->add_tensor(t_name_1, t_send_1,
                        dims, type, SmartRedis::MemoryLayout::nested);
    dataset->add_tensor(t_name_2, t_send_2,
                        dims, type, SmartRedis::MemoryLayout::nested);
    dataset->add_tensor(t_name_3, t_send_3,
                        dims, type, SmartRedis::MemoryLayout::nested);

    //Add metadata fields to the DataSet.  _meta_1 and _meta_2
    //Add only a portion of the metadata values so that we can test
    //that a user can add metadata after the object has been copied

    dataset->add_meta_scalar("dbl_field_1",
                            &DATASET_TEST_UTILS::dbl_meta_1,
                            SmartRedis::MetaDataType::dbl);
    dataset->add_meta_scalar("dbl_field_1",
                            &DATASET_TEST_UTILS::dbl_meta_2,
                            SmartRedis::MetaDataType::dbl);
    dataset->add_meta_scalar("dbl_field_2",
                            &DATASET_TEST_UTILS::dbl_meta_3,
                            SmartRedis::MetaDataType::dbl);

    dataset->add_meta_scalar("flt_field_1",
                            &DATASET_TEST_UTILS::flt_meta_1,
                            SmartRedis::MetaDataType::flt);
    dataset->add_meta_scalar("flt_field_1",
                            &DATASET_TEST_UTILS::flt_meta_2,
                            SmartRedis::MetaDataType::flt);
    dataset->add_meta_scalar("flt_field_2",
                            &DATASET_TEST_UTILS::flt_meta_3,
                            SmartRedis::MetaDataType::flt);

    dataset->add_meta_scalar("i64_field_1",
                            &DATASET_TEST_UTILS::i64_meta_1,
                            SmartRedis::MetaDataType::int64);
    dataset->add_meta_scalar("i64_field_1",
                            &DATASET_TEST_UTILS::i64_meta_2,
                            SmartRedis::MetaDataType::int64);
    dataset->add_meta_scalar("i64_field_2",
                            &DATASET_TEST_UTILS::i64_meta_3,
                            SmartRedis::MetaDataType::int64);

    dataset->add_meta_scalar("i32_field_1",
                            &DATASET_TEST_UTILS::i32_meta_1,
                            SmartRedis::MetaDataType::int32);
    dataset->add_meta_scalar("i32_field_1",
                            &DATASET_TEST_UTILS::i32_meta_2,
                            SmartRedis::MetaDataType::int32);
    dataset->add_meta_scalar("i32_field_2",
                            &DATASET_TEST_UTILS::i32_meta_3,
                            SmartRedis::MetaDataType::int32);

    dataset->add_meta_scalar("ui64_field_1",
                            &DATASET_TEST_UTILS::ui64_meta_1,
                            SmartRedis::MetaDataType::uint64);
    dataset->add_meta_scalar("ui64_field_1",
                            &DATASET_TEST_UTILS::ui64_meta_2,
                            SmartRedis::MetaDataType::uint64);
    dataset->add_meta_scalar("ui64_field_2",
                            &DATASET_TEST_UTILS::ui64_meta_3,
                            SmartRedis::MetaDataType::uint64);

    dataset->add_meta_scalar("ui32_field_1",
                            &DATASET_TEST_UTILS::ui32_meta_1,
                            SmartRedis::MetaDataType::uint32);

    //Move the DataSet half way through metadata additions to
    //test that we can continue adding new fields to the old fields
    SmartRedis::DataSet moved_dataset = std::move(*dataset);

    moved_dataset.add_meta_scalar("ui32_field_1",
                            &DATASET_TEST_UTILS::ui32_meta_2,
                            SmartRedis::MetaDataType::uint32);
    moved_dataset.add_meta_scalar("ui32_field_2",
                            &DATASET_TEST_UTILS::ui32_meta_3,
                            SmartRedis::MetaDataType::uint32);

    moved_dataset.add_meta_string("str_field_1",
                            DATASET_TEST_UTILS::str_meta_1);
    moved_dataset.add_meta_string("str_field_1",
                            DATASET_TEST_UTILS::str_meta_2);
    moved_dataset.add_meta_string("str_field_2",
                            DATASET_TEST_UTILS::str_meta_3);

    delete dataset;
    client.put_dataset(moved_dataset);

    if(!client.tensor_exists(dataset_name))
        throw std::runtime_error("The moved assigned DataSet "\
                                 "confirmation key is not set.");

    SmartRedis::DataSet full_dataset = client.get_dataset(dataset_name);

    DATASET_TEST_UTILS::check_tensor_names(full_dataset,
                                    {t_name_1, t_name_2, t_name_3});

    //Check that the tensors are the same
    DATASET_TEST_UTILS::check_nested_3D_tensor(full_dataset,
                                               t_name_1,
                                               type, t_send_1, dims);
    DATASET_TEST_UTILS::check_nested_3D_tensor(full_dataset,
                                               t_name_2,
                                               type, t_send_2, dims);
    DATASET_TEST_UTILS::check_nested_3D_tensor(full_dataset,
                                               t_name_3,
                                               type, t_send_3, dims);

    //Check that the metadata values are correct for the metadata
    DATASET_TEST_UTILS::check_dataset_metadata(full_dataset);

    return;
}

int main(int argc, char* argv[]) {

    //Declare the dimensions for the 3D arrays
    std::vector<size_t> dims{5,4,17};

    std::string dataset_name;

    dataset_name = "3D_dbl_dataset_move_assign";
    put_get_3D_array<double,double>(
                    &set_3D_array_floating_point_values<double>,
                    dims, SmartRedis::TensorType::dbl,
                    "_dbl", dataset_name);

    dataset_name = "3D_flt_dataset_move_assign";
    put_get_3D_array<float,float>(
                    &set_3D_array_floating_point_values<float>,
                    dims, SmartRedis::TensorType::flt,
                    "_flt", dataset_name);

    dataset_name = "3D_i64_dataset_move_assign";
    put_get_3D_array<int64_t,int64_t>(
                        &set_3D_array_integral_values<int64_t>,
                        dims, SmartRedis::TensorType::int64,
                        "_i64", dataset_name);

    dataset_name = "3D_i32_dataset_move_assign";
    put_get_3D_array<int32_t,int32_t>(
                        &set_3D_array_integral_values<int32_t>,
                        dims, SmartRedis::TensorType::int32,
                        "_i32", dataset_name);

    dataset_name = "3D_i16_dataset_move_assign";
    put_get_3D_array<int16_t,int16_t>(
                        &set_3D_array_integral_values<int16_t>,
                        dims, SmartRedis::TensorType::int16,
                        "_i16", dataset_name);

    dataset_name = "3D_i8_dataset_move_assign";
    put_get_3D_array<int8_t,int8_t>(
                        &set_3D_array_integral_values<int8_t>,
                        dims, SmartRedis::TensorType::int8,
                        "_i8", dataset_name);

    dataset_name = "3D_ui16_dataset_move_assign";
    put_get_3D_array<uint16_t,uint16_t>(
                        &set_3D_array_integral_values<uint16_t>,
                        dims, SmartRedis::TensorType::uint16,
                        "_ui16", dataset_name);

    dataset_name = "3D_ui8_dataset_move_assign";
    put_get_3D_array<uint8_t,uint8_t>(
                        &set_3D_array_integral_values<uint8_t>,
                        dims, SmartRedis::TensorType::uint8,
                        "_ui8", dataset_name);

    std::cout<<"Finished DataSet move assignment operator test."<<std::endl;

  return 0;
}
