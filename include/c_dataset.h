#ifndef SMARTSIM_C_DATASET_H
#define SMARTSIM_C_DATASET_H
///@file
///\brief C-wrappers for the C++ DataSet class
#include "dataset.h"
#include "enums/c_memory_layout.h"
#include "enums/c_tensor_type.h"
#include "enums/c_metadata_type.h"

#ifdef __cplusplus
extern "C" {
#endif

//! DataSet c-interface constructor
void* CDataSet(const char* name /*!< The name of the DataSet*/,
               const size_t name_length /*!< The length of the DataSet name c-string, excluding null terminating character*/
              );

//! Add a tensor to the DataSet
void add_tensor(void* dataset /*!< A c_ptr to the dataset object */,
                const char* name /*!< The name of the tensor*/,
                const size_t name_length /*!< The length of the tensor name c-string, excluding null terminating character*/,
                void* data /*!< A c_ptr to the data of the tensor*/,
                const size_t* dims /*!< Length along each dimension of the tensor*/,
                const size_t n_dims /*!< The number of dimensions of the tensor*/,
                CTensorType type /*!< The data type of the tensor */,
                CMemoryLayout mem_layout /*!< The MemoryLayout enum describing the layout of source data*/
                );

//! Add metadata field value (non-string) to the DataSet.  Default behavior is to append existing fields.
void add_meta_scalar(void* dataset /*!< A c_ptr to the dataset object */,
                     const char* name /*!< The name of the metadata field*/,
                     const size_t name_length /*!< The length of the metadata name c-string, excluding null terminating character*/,
                     const void* data /*!< A c_ptr to the metadata value*/,
                     CMetaDataType type /*!< The data type of the metadata */
                     );

//! Add metadata string field value to the DataSet.  Default behavior is to append existing fields.
void add_meta_string(void* dataset /*!< A c_ptr to the dataset object */,
                     const char* name /*!< The name of the metadata field*/,
                     const size_t name_length /*!< The length of the metadata name c-string, excluding null terminating character*/,
                     const char* data /*!< A c_ptr to the metadata value*/,
                     const size_t data_length /*!< The length of the metadata string value*/
                     );


//! Get tensor data and return an allocated multi-dimensional array
void get_dataset_tensor(void* dataset /*!< A c_ptr to the dataset object */,
                        const char* name /*!< The name used to reference the tensor*/,
                        const size_t name_length /*!< The length of the tensor name c-string, excluding null terminating character*/,
                        void** data /*!< A c_ptr to the tensor data */,
                        size_t** dims /*!< Length along each dimension of the tensor*/,
                        size_t* n_dims /*!< The number of dimensions of the tensor*/,
                        CTensorType* type /*!< The data type of the tensor*/,
                        const CMemoryLayout mem_layout /*!< The MemoryLayout enum describing the layout of source data*/
                        );

//! Get tensor data and fill an already allocated array
void unpack_dataset_tensor(void* dataset /*!< A c_ptr to the dataset object */,
                           const char* name /*!< The name used to reference the tensor*/,
                           const size_t name_length /*!< The length of the tensor name c-string, excluding null terminating character*/,
                           void* data /*!< A c_ptr to the data of the tensor*/,
                           const size_t* dims /*!< Length along each dimension of the tensor in the memory space*/,
                           const size_t n_dims /*!< The number of dimensions of the tensor in the memory space*/,
                           const CTensorType type /*!< The data type of the tensor*/,
                           const CMemoryLayout mem_layout /*!< The MemoryLayout enum describing the layout of source data*/
                           );

//! Get metadata (non-string) field from the DataSet
void get_meta_scalars(void* dataset /*!< A c_ptr to the dataset object */,
                      const char* name /*!< The name used to reference the metadata*/,
                      const size_t name_length /*!< The length of the metadata field name c-string, excluding null terminating character*/,
                      void** data /*!< A c_ptr reference to the metadata*/,
                      size_t* length /*!< The length of the metadata c_ptr*/,
                      CMetaDataType* type /*!< The data type of the metadata*/
                      );

//! Get metadata (non-string) field from the DataSet
void get_meta_strings(void* dataset /*!< A c_ptr to the dataset object */,
                      const char* name /*!< The name used to reference the metadata*/,
                      const size_t name_length /*!< The length of the metadata field name c-string, excluding null terminating character*/,
                      char*** data /*!< A c_ptr to the char** pointer that will be redirected to the string values*/,
                      size_t* n_strings /*!< The number of strings returned to the user*/,
                      size_t** lengths /*!< An array of string lengths returned to the user*/
                      );

#ifdef __cplusplus
}
#endif
#endif // SMARTSIM_C_DATASET_H
