#include "client.h"
#include "dataset.h"
#include "client_test_utils.h"

template <typename T>
void check_meta_field(SILC::DataSet& dataset,
                      std::string field_name,
                      SILC::MetaDataType type,
                      std::vector<T> vals)
{
  /* This function will check the meta data field
  that is supposed to have the provided values
  and type.
  */

  T* retrieved_vals;
  size_t retrieved_length;
  SILC::MetaDataType retrieved_type;

  dataset.get_meta_scalars(field_name,
                          (void*&)retrieved_vals,
                          retrieved_length,
                          retrieved_type);

  if(retrieved_type!=type)
    throw std::runtime_error("The retrieved type "\
                             "does not match "\
                             "expected value of for field "\
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

template <typename T_send, typename T_recv>
void put_get_3D_array(
		    void (*fill_array)(T_send***, int, int, int),
		    std::vector<size_t> dims,
        SILC::TensorType type,
        std::string key_suffix,
        std::string dataset_name)
{
  /* This function will put and get a 3D arrays and metadata
  in the database.
  */

  T_send*** t_send_1 = allocate_3D_array<T_send>(dims[0], dims[1], dims[2]);
  fill_array(t_send_1, dims[0], dims[1], dims[2]);

  T_send*** t_send_2 = allocate_3D_array<T_send>(dims[0], dims[1], dims[2]);
  fill_array(t_send_2, dims[0], dims[1], dims[2]);

  T_send*** t_send_3 = allocate_3D_array<T_send>(dims[0], dims[1], dims[2]);
  fill_array(t_send_3, dims[0], dims[1], dims[2]);

  //Declare meta data values
  double dbl_meta_1 = (1.0*rand())/RAND_MAX;
  double dbl_meta_2 = (1.0*rand())/RAND_MAX;
  double dbl_meta_3 = (1.0*rand())/RAND_MAX;
  float flt_meta_1 = (1.0*rand())/RAND_MAX;
  float flt_meta_2 = (1.0*rand())/RAND_MAX;
  float flt_meta_3 = (1.0*rand())/RAND_MAX;
  int64_t i64_meta_1 = rand();
  int64_t i64_meta_2 = rand();
  int64_t i64_meta_3 = rand();
  int32_t i32_meta_1 = rand();
  int32_t i32_meta_2 = rand();
  int32_t i32_meta_3 = rand();
  uint64_t ui64_meta_1 = rand();
  uint64_t ui64_meta_2 = rand();
  uint64_t ui64_meta_3 = rand();
  uint32_t ui32_meta_1 = rand();
  uint32_t ui32_meta_2 = rand();
  uint32_t ui32_meta_3 = rand();
  std::string str_meta_1 = std::string("test_meta_string_1");
  std::string str_meta_2 = std::string("test_meta_string_2");
  std::string str_meta_3 = std::string("test_meta_string_3");

  //Create Client and DataSets
  SILC::Client client(use_cluster());
  SILC::DataSet* MyDataSet = new SILC::DataSet(dataset_name);

  //Add tensors to the DataSet
  MyDataSet->add_tensor("tensor_1", t_send_1,
                       dims, type, SILC::MemoryLayout::nested);
  MyDataSet->add_tensor("tensor_2", t_send_2,
                       dims, type, SILC::MemoryLayout::nested);
  MyDataSet->add_tensor("tensor_3", t_send_3,
                       dims, type, SILC::MemoryLayout::nested);

  //Add metadata fields to the DataSet.  _meta_1 and _meta_2
  //values added to _field_1 and _meta_3 is added to _field_2.
  MyDataSet->add_meta_scalar("dbl_field_1", &dbl_meta_1, SILC::MetaDataType::dbl);
  MyDataSet->add_meta_scalar("dbl_field_1", &dbl_meta_2, SILC::MetaDataType::dbl);
  MyDataSet->add_meta_scalar("dbl_field_2", &dbl_meta_3, SILC::MetaDataType::dbl);

  MyDataSet->add_meta_scalar("flt_field_1", &flt_meta_1, SILC::MetaDataType::flt);
  MyDataSet->add_meta_scalar("flt_field_1", &flt_meta_2, SILC::MetaDataType::flt);
  MyDataSet->add_meta_scalar("flt_field_2", &flt_meta_3, SILC::MetaDataType::flt);

  MyDataSet->add_meta_scalar("i64_field_1", &i64_meta_1, SILC::MetaDataType::int64);
  MyDataSet->add_meta_scalar("i64_field_1", &i64_meta_2, SILC::MetaDataType::int64);
  MyDataSet->add_meta_scalar("i64_field_2", &i64_meta_3, SILC::MetaDataType::int64);

  MyDataSet->add_meta_scalar("i32_field_1", &i32_meta_1, SILC::MetaDataType::int32);
  MyDataSet->add_meta_scalar("i32_field_1", &i32_meta_2, SILC::MetaDataType::int32);
  MyDataSet->add_meta_scalar("i32_field_2", &i32_meta_3, SILC::MetaDataType::int32);

  MyDataSet->add_meta_scalar("ui64_field_1", &ui64_meta_1, SILC::MetaDataType::uint64);
  MyDataSet->add_meta_scalar("ui64_field_1", &ui64_meta_2, SILC::MetaDataType::uint64);
  MyDataSet->add_meta_scalar("ui64_field_2", &ui64_meta_3, SILC::MetaDataType::uint64);

  MyDataSet->add_meta_scalar("ui32_field_1", &ui32_meta_1, SILC::MetaDataType::uint32);

  //Copy the DataSet half way through metadata additions to
  //test that we can continue adding new fields to the old fields
  SILC::DataSet CopiedDataSet = *MyDataSet;

  CopiedDataSet.add_meta_scalar("ui32_field_1", &ui32_meta_2, SILC::MetaDataType::uint32);
  CopiedDataSet.add_meta_scalar("ui32_field_2", &ui32_meta_3, SILC::MetaDataType::uint32);

  CopiedDataSet.add_meta_string("str_field_1", str_meta_1);
  CopiedDataSet.add_meta_string("str_field_1", str_meta_2);
  CopiedDataSet.add_meta_string("str_field_2", str_meta_3);

  client.put_dataset(*MyDataSet);

  //Check that the meta fields we added to CopiedDataSet do not show up
  //in MyDataSet if we put MyDataSet in the repo.
  //Check that the metadata values are correct for ui32
  //TODO we should add a method to DataSet to check if a field exists
  //and confirm that the other fields have not been added.  Currently
  //if we get a field that does not exist we will get a std::runtime_error.
  //Currently the test that there is only one entry in ui32_field_1
  //is sufficient.
  SILC::DataSet RetrievedDataSet1 = client.get_dataset(dataset_name);
  check_meta_field<uint32_t>(RetrievedDataSet1,
                             "ui32_field_1",
                             SILC::MetaDataType::uint32,
                             {ui32_meta_1});

  delete MyDataSet;
  client.put_dataset(CopiedDataSet);

  //Retrieving a dataset
  T_recv*** t_recv_1 = allocate_3D_array<T_recv>(dims[0], dims[1], dims[2]);
  T_recv*** t_recv_2 = allocate_3D_array<T_recv>(dims[0], dims[1], dims[2]);
  T_recv*** t_recv_3 = allocate_3D_array<T_recv>(dims[0], dims[1], dims[2]);

  SILC::DataSet RetrievedDataSet = client.get_dataset(dataset_name);
  RetrievedDataSet.unpack_tensor("tensor_1", t_recv_1,
                                 dims, type, SILC::MemoryLayout::nested);
  RetrievedDataSet.unpack_tensor("tensor_2", t_recv_2,
                                 dims, type, SILC::MemoryLayout::nested);
  RetrievedDataSet.unpack_tensor("tensor_3", t_recv_3,
                                 dims, type, SILC::MemoryLayout::nested);

  /*
  for(int i = 0; i < dims[0]; i++)
    for(int j = 0; j < dims[1]; j++)
      for(int k = 0; k < dims[2]; k++)
        std::cout<<"t_recv_1: "<<t_recv_1[i][j][k]<<std::endl;
  */

  //Check metadata .tensors value for consistency
  char** tensor_ids;
  size_t n_strings;
  size_t* lengths;
  RetrievedDataSet.get_meta_strings(".tensors",
                                    tensor_ids,
                                    n_strings,
                                    lengths);

  for(size_t q=0; q<n_strings; q++) {
    if(std::strlen(tensor_ids[q])!=lengths[q])
      throw std::runtime_error("The tensor id length does not "\
                               "the retrieved length.");
  }

  if(n_strings!=3)
    throw std::runtime_error("The .tensors metadata field does not "\
                             "contain the correct number of entries.");
  if(strcmp(tensor_ids[0],"tensor_1")!=0)
    throw std::runtime_error("The .tensors[0] metadata entry is "\
                             "incorrect");
  if(strcmp(tensor_ids[1],"tensor_2")!=0)
    throw std::runtime_error("The .tensors[1] metadata entry is "\
                             "incorrect");
  if(strcmp(tensor_ids[2],"tensor_3")!=0)
    throw std::runtime_error("The .tensors[2] metadata entry is "\
                             "incorrect");
  std::cout<<"Correctly fetched metadata .tensors."<<std::endl;

  //Check that the tensor values are correct
  if(is_equal_3D_array(t_send_1, t_recv_1, dims[0], dims[1], dims[2]))
    std::cout<<"Correctly fetched tensor_1 with unpack tensor"<<std::endl;
  else
    throw std::runtime_error("tensor_1 did not match the send "\
                             "and receive values");
  if(is_equal_3D_array(t_send_2, t_recv_2, dims[0], dims[1], dims[2]))
    std::cout<<"Correctly fetched tensor_2 with unpack tensor"<<std::endl;
  else
    throw std::runtime_error("tensor_2 did not match the send "\
                             "and receive values");
  if(is_equal_3D_array(t_send_2, t_recv_2, dims[0], dims[1], dims[2]))
    std::cout<<"Correctly fetched tensor_3 with unpack tensor"<<std::endl;
  else
    throw std::runtime_error("tensor_3 did not match the send "\
                             "and receive values");

  // Retrieve tensors where the DataSet handles memory allocation
  void* t_get_1;
  SILC::TensorType t_get_1_type;
  std::vector<size_t> t_get_1_dims;
  RetrievedDataSet.get_tensor("tensor_1", t_get_1,
                              t_get_1_dims, t_get_1_type,
                              SILC::MemoryLayout::nested);
  void* t_get_2;
  SILC::TensorType t_get_2_type;
  std::vector<size_t> t_get_2_dims;
  RetrievedDataSet.get_tensor("tensor_2", t_get_2,
                              t_get_2_dims, t_get_2_type,
                              SILC::MemoryLayout::nested);
  void* t_get_3;
  SILC::TensorType t_get_3_type;
  std::vector<size_t> t_get_3_dims;
  RetrievedDataSet.get_tensor("tensor_3", t_get_3,
                              t_get_3_dims, t_get_3_type,
                              SILC::MemoryLayout::nested);

  if(t_get_1_type!=type)
    throw std::runtime_error("Retrieved type for tensor_1 "\
                             "does not match known type");

  if(t_get_2_type!=type)
    throw std::runtime_error("Retrieved type for tensor_2 "\
                             "does not match known type");

  if(t_get_3_type!=type)
    throw std::runtime_error("Retrieved type for tensor_3 "\
                             "does not match known type");

  if(t_get_1_dims!=dims)
    throw std::runtime_error("Retrieved dims for tensor_1 "\
                             "do not match the known dims.");

  if(t_get_2_dims!=dims)
    throw std::runtime_error("Retrieved dims for tensor_2 "\
                             "do not match the known dims.");

  if(t_get_3_dims!=dims)
    throw std::runtime_error("Retrieved dims for tensor_3 "\
                             "do not match the known dims.");

  if(is_equal_3D_array(t_send_1, (T_recv***)t_get_1, dims[0], dims[1], dims[2]))
    std::cout<<"Correctly fetched and allocated t_get_1."<<std::endl;
  else
    throw std::runtime_error("tensor_1 could not be retrieved correctly "\
                             "with get_tensor.");

  if(is_equal_3D_array(t_send_2, (T_recv***)t_get_2, dims[0], dims[1], dims[2]))
    std::cout<<"Correctly fetched and allocated t_get_2."<<std::endl;
  else
    throw std::runtime_error("tensor_2 could not be retrieved correctly "\
                             "with get_tensor.");

  if(is_equal_3D_array(t_send_3, (T_recv***)t_get_3, dims[0], dims[1], dims[2]))
    std::cout<<"Correctly fetched and allocated t_get_3."<<std::endl;
  else
    throw std::runtime_error("tensor_3 could not be retrieved correctly "\
                             "with get_tensor.");

  //Check that the metadata values are correct for dbl
  check_meta_field<double>(RetrievedDataSet,
                           "dbl_field_1",
                           SILC::MetaDataType::dbl,
                           {dbl_meta_1, dbl_meta_2});

  check_meta_field<double>(RetrievedDataSet,
                           "dbl_field_2",
                           SILC::MetaDataType::dbl,
                           {dbl_meta_3});

  //Check that the metadata values are correct for flt

  check_meta_field<float>(RetrievedDataSet,
                          "flt_field_1",
                          SILC::MetaDataType::flt,
                          {flt_meta_1, flt_meta_2});

  check_meta_field<float>(RetrievedDataSet,
                         "flt_field_2",
                         SILC::MetaDataType::flt,
                         {flt_meta_3});

  //Check that the metadata values are correct for i64

  check_meta_field<int64_t>(RetrievedDataSet,
                            "i64_field_1",
                            SILC::MetaDataType::int64,
                            {i64_meta_1, i64_meta_2});

  check_meta_field<int64_t>(RetrievedDataSet,
                            "i64_field_2",
                            SILC::MetaDataType::int64,
                            {i64_meta_3});

  //Check that the metadata values are correct for i32

  check_meta_field<int32_t>(RetrievedDataSet,
                            "i32_field_1",
                            SILC::MetaDataType::int32,
                            {i32_meta_1, i32_meta_2});

  check_meta_field<int32_t>(RetrievedDataSet,
                            "i32_field_2",
                            SILC::MetaDataType::int32,
                            {i32_meta_3});

  //Check that the metadata values are correct for ui64

  check_meta_field<uint64_t>(RetrievedDataSet,
                             "ui64_field_1",
                             SILC::MetaDataType::uint64,
                             {ui64_meta_1, ui64_meta_2});

  check_meta_field<uint64_t>(RetrievedDataSet,
                             "ui64_field_2",
                             SILC::MetaDataType::uint64,
                             {ui64_meta_3});

  //Check that the metadata values are correct for ui32
  check_meta_field<uint32_t>(RetrievedDataSet,
                             "ui32_field_1",
                             SILC::MetaDataType::uint32,
                             {ui32_meta_1, ui32_meta_2});

  check_meta_field<uint32_t>(RetrievedDataSet,
                             "ui32_field_2",
                             SILC::MetaDataType::uint32,
                             {ui32_meta_3});

  //Check that the metadata values are correct for str

  std::vector<std::string> str_meta_field_1 =
    RetrievedDataSet.get_meta_strings("str_field_1");

  if(str_meta_field_1.size()!=2)
    throw std::runtime_error("The number of entries in str_meta_field_1 "\
                             "is incorrect.");
  if(str_meta_1.compare(str_meta_field_1[0])!=0)
    throw std::runtime_error("The retrieved value for str_meta_1 "\
                             "is incorrect.");
  if(str_meta_2.compare(str_meta_field_1[1])!=0)
    throw std::runtime_error("The retrieved value for str_meta_2 "\
                             "is incorrect.");

  //Check that the metadata values are correct for str_field_2
  std::vector<std::string> str_meta_field_2 =
    RetrievedDataSet.get_meta_strings("str_field_2");

  if(str_meta_field_2.size()!=1)
    throw std::runtime_error("The number of entries in str_meta_field_2 "\
                             "is incorrect.");
  if(str_meta_3.compare(str_meta_field_2[0])!=0)
    throw std::runtime_error("The retrieved value for str_meta_3 "\
                             "is incorrect.");

  std::cout<<"Correctly fetched string type metadata"<<std::endl;

  //Free out tensor memory info
  free_3D_array(t_send_1, dims[0], dims[1]);
  free_3D_array(t_send_2, dims[0], dims[1]);
  free_3D_array(t_send_3, dims[0], dims[1]);
  free_3D_array(t_recv_1, dims[0], dims[1]);
  free_3D_array(t_recv_2, dims[0], dims[1]);
  free_3D_array(t_recv_3, dims[0], dims[1]);

  return;
}

int main(int argc, char* argv[]) {

  //Declare the dimensions for the 3D arrays
  std::vector<size_t> dims{5,4,17};

  std::string dataset_name;

  dataset_name = "3D_dbl_dataset_rank";
  put_get_3D_array<double,double>(
				  &set_3D_array_floating_point_values<double>,
				  dims, SILC::TensorType::dbl, "_dbl", dataset_name);

  dataset_name = "3D_flt_dataset_rank";
  put_get_3D_array<float,float>(
	  			&set_3D_array_floating_point_values<float>,
  				dims, SILC::TensorType::flt, "_flt", dataset_name);

  dataset_name = "3D_i64_dataset_rank";
  put_get_3D_array<int64_t,int64_t>(
				    &set_3D_array_integral_values<int64_t>,
				    dims, SILC::TensorType::int64, "_i64", dataset_name);

  dataset_name = "3D_i32_dataset_rank";
  put_get_3D_array<int32_t,int32_t>(
				    &set_3D_array_integral_values<int32_t>,
				    dims, SILC::TensorType::int32, "_i32", dataset_name);

  dataset_name = "3D_i16_dataset_rank";
  put_get_3D_array<int16_t,int16_t>(
				      &set_3D_array_integral_values<int16_t>,
				      dims, SILC::TensorType::int16, "_i16", dataset_name);

  dataset_name = "3D_i8_dataset_rank";
  put_get_3D_array<int8_t,int8_t>(
				      &set_3D_array_integral_values<int8_t>,
				      dims, SILC::TensorType::int8, "_i8", dataset_name);

  dataset_name = "3D_ui16_dataset_rank";
  put_get_3D_array<uint16_t,uint16_t>(
				      &set_3D_array_integral_values<uint16_t>,
				      dims, SILC::TensorType::uint16, "_ui16", dataset_name);

  dataset_name = "3D_ui8_dataset_rank";
  put_get_3D_array<uint8_t,uint8_t>(
				      &set_3D_array_integral_values<uint8_t>,
				      dims, SILC::TensorType::uint8, "_ui8", dataset_name);

  std::cout<<"Finished DataSet copy assignment test."<<std::endl;
  return 0;
}
