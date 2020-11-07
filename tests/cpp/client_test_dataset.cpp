#include "client.h"
#include "dataset.h"
#include "client_test_utils.h"
#include <mpi.h>

template <typename T_send, typename T_recv>
void put_get_3D_array(
		    void (*fill_array)(T_send***, int, int, int),
		    std::vector<int> dims,
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
  MyDataSet.add_tensor("tensor_1", type, t_send_1, dims);
  MyDataSet.add_tensor("tensor_2", type, t_send_2, dims);
  MyDataSet.add_tensor("tensor_3", type, t_send_3, dims);

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
  RetrievedDataSet.unpack_tensor("tensor_1", type, t_recv_1, dims);
  RetrievedDataSet.unpack_tensor("tensor_2", type, t_recv_2, dims);
  RetrievedDataSet.unpack_tensor("tensor_3", type, t_recv_3, dims);

  /*
  for(int i = 0; i < dims[0]; i++)
    for(int j = 0; j < dims[1]; j++)
      for(int k = 0; k < dims[2]; k++)
        std::cout<<"t_recv_1: "<<t_recv_1[i][j][k]<<std::endl;
  */

  //Check metadata .tensors value for consistency
  char** tensor_ids;
  int n_strings;
  RetrievedDataSet.get_meta(".tensors", "STRING",
                            (void*&)tensor_ids, n_strings);
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
  RetrievedDataSet.get_tensor("tensor_1", type, t_get_1, dims);
  void* t_get_2;
  RetrievedDataSet.get_tensor("tensor_2", type, t_get_2, dims);
  void* t_get_3;
  RetrievedDataSet.get_tensor("tensor_3", type, t_get_3, dims);

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

  //Check that the metadata values are correct for dbl_field_1
  double* dbl_meta_field_1;
  int n_dbl_meta_field_1;
  RetrievedDataSet.get_meta("dbl_field_1", "DOUBLE",
                            (void*&)dbl_meta_field_1, n_dbl_meta_field_1);
  if(n_dbl_meta_field_1!=2)
    throw std::runtime_error("The number of entries in dbl_meta_field_1 "\
                             "is incorrect.");
  if(dbl_meta_field_1[0]!=dbl_meta_1)
    throw std::runtime_error("The retrieved value for dbl_meta_1 "\
                             "is incorrect.");
  if(dbl_meta_field_1[1]!=dbl_meta_2)
    throw std::runtime_error("The retrieved value for dbl_meta_2 "\
                             "is incorrect.");

  //Check that the metadata values are correct for dbl_field_2
  double* dbl_meta_field_2;
  int n_dbl_meta_field_2;
  RetrievedDataSet.get_meta("dbl_field_2", "DOUBLE",
                            (void*&)dbl_meta_field_2, n_dbl_meta_field_2);
  if(n_dbl_meta_field_2!=1)
    throw std::runtime_error("The number of entries in dbl_meta_field_2 "\
                             "is incorrect.");
  if(dbl_meta_field_2[0]!=dbl_meta_3)
    throw std::runtime_error("The retrieved value for dbl_meta_3 "\
                             "is incorrect.");

  std::cout<<"Correctly fetched double type metadata"<<std::endl;

  //Check that the metadata values are correct for flt_field_1
  float* flt_meta_field_1;
  int n_flt_meta_field_1;
  RetrievedDataSet.get_meta("flt_field_1", "FLOAT",
                            (void*&)flt_meta_field_1, n_flt_meta_field_1);
  if(n_flt_meta_field_1!=2)
    throw std::runtime_error("The number of entries in flt_meta_field_1 "\
                             "is incorrect.");
  if(flt_meta_field_1[0]!=flt_meta_1)
    throw std::runtime_error("The retrieved value for flt_meta_1 "\
                             "is incorrect.");
  if(flt_meta_field_1[1]!=flt_meta_2)
    throw std::runtime_error("The retrieved value for flt_meta_2 "\
                             "is incorrect.");

  //Check that the metadata values are correct for flt_field_2
  float* flt_meta_field_2;
  int n_flt_meta_field_2;
  RetrievedDataSet.get_meta("flt_field_2", "FLOAT",
                            (void*&)flt_meta_field_2, n_flt_meta_field_2);
  if(n_flt_meta_field_2!=1)
    throw std::runtime_error("The number of entries in flt_meta_field_2 "\
                             "is incorrect.");
  if(flt_meta_field_2[0]!=flt_meta_3)
    throw std::runtime_error("The retrieved value for flt_meta_3 "\
                             "is incorrect.");

  std::cout<<"Correctly fetched float type metadata"<<std::endl;

  //Check that the metadata values are correct for i64_field_1
  int64_t* i64_meta_field_1;
  int n_i64_meta_field_1;
  RetrievedDataSet.get_meta("i64_field_1", "INT64",
                            (void*&)i64_meta_field_1, n_i64_meta_field_1);
  if(n_i64_meta_field_1!=2)
    throw std::runtime_error("The number of entries in i64_meta_field_1 "\
                             "is incorrect.");
  if(i64_meta_field_1[0]!=i64_meta_1)
    throw std::runtime_error("The retrieved value for i64_meta_1 "\
                             "is incorrect.");
  if(i64_meta_field_1[1]!=i64_meta_2)
    throw std::runtime_error("The retrieved value for i64_meta_2 "\
                             "is incorrect.");

  //Check that the metadata values are correct for i64_field_2
  int64_t* i64_meta_field_2;
  int n_i64_meta_field_2;
  RetrievedDataSet.get_meta("i64_field_2", "INT64",
                            (void*&)i64_meta_field_2, n_i64_meta_field_2);
  if(n_i64_meta_field_2!=1)
    throw std::runtime_error("The number of entries in i64_meta_field_2 "\
                             "is incorrect.");
  if(i64_meta_field_2[0]!=i64_meta_3)
    throw std::runtime_error("The retrieved value for i64_meta_3 "\
                             "is incorrect.");

  std::cout<<"Correctly fetched i64 type metadata"<<std::endl;

  //Check that the metadata values are correct for i32_field_1
  int32_t* i32_meta_field_1;
  int n_i32_meta_field_1;
  RetrievedDataSet.get_meta("i32_field_1", "INT32",
                            (void*&)i32_meta_field_1, n_i32_meta_field_1);
  if(n_i32_meta_field_1!=2)
    throw std::runtime_error("The number of entries in i32_meta_field_1 "\
                             "is incorrect.");
  if(i32_meta_field_1[0]!=i32_meta_1)
    throw std::runtime_error("The retrieved value for i32_meta_1 "\
                             "is incorrect.");
  if(i32_meta_field_1[1]!=i32_meta_2)
    throw std::runtime_error("The retrieved value for i32_meta_2 "\
                             "is incorrect.");

  //Check that the metadata values are correct for i32_field_2
  int32_t* i32_meta_field_2;
  int n_i32_meta_field_2;
  RetrievedDataSet.get_meta("i32_field_2", "INT32",
                            (void*&)i32_meta_field_2, n_i32_meta_field_2);
  if(n_i32_meta_field_2!=1)
    throw std::runtime_error("The number of entries in i32_meta_field_2 "\
                             "is incorrect.");
  if(i32_meta_field_2[0]!=i32_meta_3)
    throw std::runtime_error("The retrieved value for i32_meta_3 "\
                             "is incorrect.");

  std::cout<<"Correctly fetched i32 type metadata"<<std::endl;

  //Check that the metadata values are correct for ui64_field_1
  uint64_t* ui64_meta_field_1;
  int n_ui64_meta_field_1;
  RetrievedDataSet.get_meta("ui64_field_1", "UINT64",
                            (void*&)ui64_meta_field_1, n_ui64_meta_field_1);
  if(n_ui64_meta_field_1!=2)
    throw std::runtime_error("The number of entries in ui64_meta_field_1 "\
                             "is incorrect.");
  if(ui64_meta_field_1[0]!=ui64_meta_1)
    throw std::runtime_error("The retrieved value for ui64_meta_1 "\
                             "is incorrect.");
  if(ui64_meta_field_1[1]!=ui64_meta_2)
    throw std::runtime_error("The retrieved value for ui64_meta_1 "\
                             "is incorrect.");

  //Check that the metadata values are correct for ui64_field_2
  uint64_t* ui64_meta_field_2;
  int n_ui64_meta_field_2;
  RetrievedDataSet.get_meta("ui64_field_2", "UINT64",
                            (void*&)ui64_meta_field_2, n_ui64_meta_field_2);
  if(n_ui64_meta_field_2!=1)
    throw std::runtime_error("The number of entries in ui64_meta_field_2 "\
                             "is incorrect.");
  if(ui64_meta_field_2[0]!=ui64_meta_3)
    throw std::runtime_error("The retrieved value for ui64_meta_3 "\
                             "is incorrect.");

  std::cout<<"Correctly fetched ui64 type metadata"<<std::endl;

  //Check that the metadata values are correct for ui32_field_1
  uint32_t* ui32_meta_field_1;
  int n_ui32_meta_field_1;
  RetrievedDataSet.get_meta("ui32_field_1", "UINT32",
                            (void*&)ui32_meta_field_1, n_ui32_meta_field_1);
  if(n_ui32_meta_field_1!=2)
    throw std::runtime_error("The number of entries in ui32_meta_field_1 "\
                             "is incorrect.");
  if(ui32_meta_field_1[0]!=ui32_meta_1)
    throw std::runtime_error("The retrieved value for ui32_meta_1 "\
                             "is incorrect.");
  if(ui32_meta_field_1[1]!=ui32_meta_2)
    throw std::runtime_error("The retrieved value for ui32_meta_2 "\
                             "is incorrect.");

  //Check that the metadata values are correct for ui32_field_2
  uint32_t* ui32_meta_field_2;
  int n_ui32_meta_field_2;
  RetrievedDataSet.get_meta("ui32_field_2", "UINT32",
                            (void*&)ui32_meta_field_2, n_ui32_meta_field_2);
  if(n_ui32_meta_field_2!=1)
    throw std::runtime_error("The number of entries in ui32_meta_field_2 "\
                             "is incorrect.");
  if(ui32_meta_field_2[0]!=ui32_meta_3)
    throw std::runtime_error("The retrieved value for ui32_meta_3 "\
                             "is incorrect.");

  std::cout<<"Correctly fetched ui32 type metadata"<<std::endl;

  //Check that the metadata values are correct for str_field_1
  char** str_meta_field_1;
  int n_str_meta_field_1;
  RetrievedDataSet.get_meta("str_field_1", "STRING",
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

  //Check that the metadata values are correct for str_field_2
  char** str_meta_field_2;
  int n_str_meta_field_2;
  RetrievedDataSet.get_meta("str_field_2", "STRING",
                            (void*&)str_meta_field_2, n_str_meta_field_2);
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
  std::vector<int> dims{5,4,17};


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
