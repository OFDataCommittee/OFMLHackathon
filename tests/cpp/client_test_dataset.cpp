#include "client.h"
#include "dataset.h"
#include "client_test_utils.h"
#include <mpi.h>

template <typename T>
void check_meta_field(DataSet& dataset,
                      std::string field_name,
                      std::string type,
                      std::vector<T> vals)
{
  /* This function will check the meta data field
  that is supposed to have the provided values
  and type.
  */

  T* retrieved_vals;
  size_t retrieved_length;
  std::string retrieved_type;

  dataset.get_meta(field_name,
                   retrieved_type,
                   (void*&)retrieved_vals,
                   retrieved_length);

  if(retrieved_type.compare(type.c_str())!=0)
    throw std::runtime_error("The retrieved type of " +
                             retrieved_type +
                             "does not match "\
                             "expected value of " + type +
                             "for field " + field_name);

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
        std::string type,
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

  //Create Client and DataSet
  SmartSimClient client(true);
  DataSet MyDataSet(dataset_name);

  //Add tensors to the DataSet
  MyDataSet.add_tensor("tensor_1", type, t_send_1,
                       dims, MemoryLayout::nested);
  MyDataSet.add_tensor("tensor_2", type, t_send_2,
                       dims, MemoryLayout::nested);
  MyDataSet.add_tensor("tensor_3", type, t_send_3,
                       dims, MemoryLayout::nested);

  //Add metadata fields to the DataSet.  _meta_1 and _meta_2
  //values added to _field_1 and _meta_3 is added to _field_2.
  MyDataSet.add_meta("dbl_field_1", "DOUBLE", &dbl_meta_1);
  MyDataSet.add_meta("dbl_field_1", "DOUBLE", &dbl_meta_2);
  MyDataSet.add_meta("dbl_field_2", "DOUBLE", &dbl_meta_3);

  MyDataSet.add_meta("flt_field_1", "FLOAT", &flt_meta_1);
  MyDataSet.add_meta("flt_field_1", "FLOAT", &flt_meta_2);
  MyDataSet.add_meta("flt_field_2", "FLOAT", &flt_meta_3);

  MyDataSet.add_meta("i64_field_1", "INT64", &i64_meta_1);
  MyDataSet.add_meta("i64_field_1", "INT64", &i64_meta_2);
  MyDataSet.add_meta("i64_field_2", "INT64", &i64_meta_3);

  MyDataSet.add_meta("i32_field_1", "INT32", &i32_meta_1);
  MyDataSet.add_meta("i32_field_1", "INT32", &i32_meta_2);
  MyDataSet.add_meta("i32_field_2", "INT32", &i32_meta_3);

  MyDataSet.add_meta("ui64_field_1", "UINT64", &ui64_meta_1);
  MyDataSet.add_meta("ui64_field_1", "UINT64", &ui64_meta_2);
  MyDataSet.add_meta("ui64_field_2", "UINT64", &ui64_meta_3);

  MyDataSet.add_meta("ui32_field_1", "UINT32", &ui32_meta_1);
  MyDataSet.add_meta("ui32_field_1", "UINT32", &ui32_meta_2);
  MyDataSet.add_meta("ui32_field_2", "UINT32", &ui32_meta_3);

  MyDataSet.add_meta("str_field_1", "STRING", str_meta_1.c_str());
  MyDataSet.add_meta("str_field_1", "STRING", str_meta_2.c_str());
  MyDataSet.add_meta("str_field_2", "STRING", str_meta_3.c_str());

  //Put the DataSet into the database
  client.put_dataset(MyDataSet);

  //Retrieving a dataset
  T_recv*** t_recv_1 = allocate_3D_array<T_recv>(dims[0], dims[1], dims[2]);
  T_recv*** t_recv_2 = allocate_3D_array<T_recv>(dims[0], dims[1], dims[2]);
  T_recv*** t_recv_3 = allocate_3D_array<T_recv>(dims[0], dims[1], dims[2]);

  DataSet RetrievedDataSet = client.get_dataset(dataset_name);
  RetrievedDataSet.unpack_tensor("tensor_1", type, t_recv_1,
                                 dims, MemoryLayout::nested);
  RetrievedDataSet.unpack_tensor("tensor_2", type, t_recv_2,
                                 dims, MemoryLayout::nested);
  RetrievedDataSet.unpack_tensor("tensor_3", type, t_recv_3,
                                 dims, MemoryLayout::nested);

  /*
  for(int i = 0; i < dims[0]; i++)
    for(int j = 0; j < dims[1]; j++)
      for(int k = 0; k < dims[2]; k++)
        std::cout<<"t_recv_1: "<<t_recv_1[i][j][k]<<std::endl;
  */

  //Check metadata .tensors value for consistency
  char** tensor_ids;
  size_t n_strings;
  std::string tensor_meta_type;
  RetrievedDataSet.get_meta(".tensors",
                            tensor_meta_type,
                            (void*&)tensor_ids,
                            n_strings);

  if(tensor_meta_type.compare("STRING")!=0)
    throw std::runtime_error("The .tensor metadata field has the "\
                             "wrong type.");

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
  std::string t_get_1_type;
  std::vector<size_t> t_get_1_dims;
  RetrievedDataSet.get_tensor("tensor_1", t_get_1_type,
                              t_get_1, t_get_1_dims,
                              MemoryLayout::nested);
  void* t_get_2;
  std::string t_get_2_type;
  std::vector<size_t> t_get_2_dims;
  RetrievedDataSet.get_tensor("tensor_2", t_get_2_type,
                              t_get_2, t_get_2_dims,
                              MemoryLayout::nested);
  void* t_get_3;
  std::string t_get_3_type;
  std::vector<size_t> t_get_3_dims;
  RetrievedDataSet.get_tensor("tensor_3", t_get_3_type,
                              t_get_3, t_get_3_dims,
                              MemoryLayout::nested);

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
                           "DOUBLE",
                           {dbl_meta_1, dbl_meta_2});

  check_meta_field<double>(RetrievedDataSet,
                           "dbl_field_2",
                           "DOUBLE",
                           {dbl_meta_3});

  //Check that the metadata values are correct for flt

  check_meta_field<float>(RetrievedDataSet,
                          "flt_field_1",
                          "FLOAT",
                          {flt_meta_1, flt_meta_2});

  check_meta_field<float>(RetrievedDataSet,
                         "flt_field_2",
                         "FLOAT",
                         {flt_meta_3});

  //Check that the metadata values are correct for i64

  check_meta_field<int64_t>(RetrievedDataSet,
                            "i64_field_1",
                            "INT64",
                            {i64_meta_1, i64_meta_2});

  check_meta_field<int64_t>(RetrievedDataSet,
                            "i64_field_2",
                            "INT64",
                            {i64_meta_3});

  //Check that the metadata values are correct for i32

  check_meta_field<int32_t>(RetrievedDataSet,
                            "i32_field_1",
                            "INT32",
                            {i32_meta_1, i32_meta_2});

  check_meta_field<int32_t>(RetrievedDataSet,
                            "i32_field_2",
                            "INT32",
                            {i32_meta_3});

  //Check that the metadata values are correct for ui64

  check_meta_field<uint64_t>(RetrievedDataSet,
                             "ui64_field_1",
                             "UINT64",
                             {ui64_meta_1, ui64_meta_2});

  check_meta_field<uint64_t>(RetrievedDataSet,
                             "ui64_field_2",
                             "UINT64",
                             {ui64_meta_3});

  //Check that the metadata values are correct for ui32
  check_meta_field<uint32_t>(RetrievedDataSet,
                             "ui32_field_1",
                             "UINT32",
                             {ui32_meta_1, ui32_meta_2});

  check_meta_field<uint32_t>(RetrievedDataSet,
                             "ui32_field_2",
                             "UINT32",
                             {ui32_meta_3});

  //Check that the metadata values are correct for str

  char** str_meta_field_1;
  size_t n_str_meta_field_1;
  std::string str_field_1_type;
  RetrievedDataSet.get_meta("str_field_1", str_field_1_type,
                            (void*&)str_meta_field_1, n_str_meta_field_1);
  if(n_str_meta_field_1!=2)
    throw std::runtime_error("The number of entries in str_meta_field_1 "\
                             "is incorrect.");
  if(str_meta_1.compare(std::string(str_meta_field_1[0]))!=0)
    throw std::runtime_error("The retrieved value for str_meta_1 "\
                             "is incorrect.");
  if(str_meta_2.compare(std::string(str_meta_field_1[1]))!=0)
    throw std::runtime_error("The retrieved value for str_meta_2 "\
                             "is incorrect.");

  if(str_field_1_type.compare("STRING")!=0)
    throw std::runtime_error("The retrieved type of " +
                             str_field_1_type +
                             "does not match "\
                             "expected value of STRING "\
                             "for field str_field_1");

  //Check that the metadata values are correct for str_field_2
  char** str_meta_field_2;
  size_t n_str_meta_field_2;
  std::string str_field_2_type;
  RetrievedDataSet.get_meta("str_field_2", str_field_2_type,
                            (void*&)str_meta_field_2, n_str_meta_field_2);

  if(str_field_2_type.compare("STRING")!=0)
    throw std::runtime_error("The retrieved type of " +
                             str_field_2_type +
                             "does not match "\
                             "expected value of STRING "\
                             "for field str_field_2");

  if(n_str_meta_field_2!=1)
    throw std::runtime_error("The number of entries in str_meta_field_2 "\
                             "is incorrect.");
  if(str_meta_3.compare(std::string(str_meta_field_2[0]))!=0)
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

  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //Declare the dimensions for the 3D arrays
  std::vector<size_t> dims{5,4,17};


  //Double 3D array
  put_get_3D_array<double, double>(&set_3D_array_floating_point_values<double>,
                                  dims, "DOUBLE", "_double", "3D_dbl_dataset");

  //Float 3D array
  put_get_3D_array<float, float>(&set_3D_array_floating_point_values<float>,
                                 dims, "FLOAT", "_float", "3D_flt_dataset");

  //int64_t 3D array
  put_get_3D_array<int64_t, int64_t>(&set_3D_array_integral_values<int64_t>,
                                 dims, "INT64", "_int64", "3D_i64_dataset");

  //int32_t 3D array
  put_get_3D_array<int32_t, int32_t>(&set_3D_array_integral_values<int32_t>,
                                 dims, "INT32", "_int32", "3D_i32_dataset");

  //int16_t 3D array
  put_get_3D_array<int16_t, int16_t>(&set_3D_array_integral_values<int16_t>,
                                 dims, "INT16", "_int16", "3D_i16_dataset");

  //int8_t 3D array
  put_get_3D_array<int8_t, int8_t>(&set_3D_array_integral_values<int8_t>,
                                 dims, "INT8", "_int8", "3D_i8_dataset");

  //uint16_t 3D array
  put_get_3D_array<uint16_t, uint16_t>(&set_3D_array_integral_values<uint16_t>,
                                 dims, "UINT16", "_uint16", "3D_ui16_dataset");

  //uint8_t 3D array
  put_get_3D_array<uint8_t, uint8_t>(&set_3D_array_integral_values<uint8_t>,
                                 dims, "UINT8", "_uint8", "3D_ui8_dataset");

  MPI_Finalize();

  std::cout<<"Rank "<<rank<<" finished 3D put and "\
                            " get tests."<<std::endl;

  return 0;
}
