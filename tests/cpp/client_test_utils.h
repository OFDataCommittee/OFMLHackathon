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

#ifndef SMARTREDIS_TEST_UTILS_H
#define SMARTREDIS_TEST_UTILS_H

#include <typeinfo>
#include <random>

#include "rediscluster.h"

using namespace SmartRedis;

class RedisClusterTestObject : public RedisCluster
{
    public:
        RedisClusterTestObject() : RedisCluster() {};

        std::string get_crc16_prefix(uint64_t hash_slot) {
            return _get_crc16_prefix(hash_slot);
        }
};

inline void to_lower(char* s) {
    /* This will turn each character in the
    c-str into the lowercase value.
    This assumes the c-str is null terminated.
    */
    if(!s)
        return;

    while((*s)!=0) {
        if( *s>='A' && *s<='Z')
            *s = *s - 'A' + 'a';
        s++;
    }
    return;
}

inline bool use_cluster()
{
    /* This function determines if a cluster
    configuration should be used in the test
    when creating a Client.
    */
    char* smartredis_test_cluster = std::getenv("SMARTREDIS_TEST_CLUSTER");
    to_lower(smartredis_test_cluster);

    if(smartredis_test_cluster) {
        if(std::strcmp(smartredis_test_cluster, "true")==0)
            return true;
    }
    return false;
}

template <typename T>
T** allocate_2D_array(int dim_1, int dim_2)
{
  /* This function allocates a 2D array and
     and returns a pointer to that 2D array.
  */
  T **array = (T **)malloc(dim_1*sizeof(T *));
  for (int i=0; i<dim_1; i++)
    array[i] = (T *)malloc(dim_2*sizeof(T));

  return array;
}

template <typename T>
T*** allocate_3D_array(int dim_1, int dim_2, int dim_3)
{
  /* This function allocates a 3D array and returns
     a pointer to that 3D array.
  */
  T*** array = (T***)malloc(dim_1*sizeof(T**));
  for (int i=0; i<dim_1; i++) {
    array[i] = (T**)malloc(dim_2*sizeof(T*));
    for(int j=0; j<dim_2; j++){
      array[i][j] = (T*)malloc(dim_3 * sizeof(T));
    }
  }
  return array;
}

template <typename T>
T**** allocate_4D_array(int dim_1, int dim_2,
                        int dim_3, int dim_4)
{
  /* This function allocates a 4D array and returns
  a pointer to that 4D array.  This is not coded
  recursively to avoid propagating bugs.
  */
  T**** array = (T****)malloc(dim_1*sizeof(T***));
  for(int i=0; i<dim_1; i++) {
    array[i] = (T***)malloc(dim_2*sizeof(T**));
    for(int j=0; j<dim_2; j++) {
      array[i][j] = (T**)malloc(dim_3*sizeof(T*));
      for(int k=0; k<dim_4; k++) {
        array[i][j][k] = (T*)malloc(dim_4 * sizeof(T));
      }
    }
  }
  return array;
}

template <typename T>
void free_1D_array(T* array)
{
  /* This function frees memory associated with
     pointer.
  */
  free(array);
}

template <typename T>
void free_2D_array(T** array, int dim_1)
{
  /*  This function frees memory of dynamically
      allocated 2D array.
  */
  for(int i=0; i<dim_1; i++)
       free(array[i]);
  free(array);
}

template <typename T>
void free_3D_array(T*** array, int dim_1, int dim_2)
{
  /* This function frees memory of dynamically
     allocated 3D array.
  */
  for(int i=0; i<dim_1; i++)
    free_2D_array(array[i], dim_2);
  free(array);
}

template <typename T>
void free_4D_array(T**** array, int dim_1,
                   int dim_2, int dim_3)
{
  for(int i=0; i<dim_1; i++)
    free_3D_array(array[i], dim_2, dim_3);
  return;
}

template <typename T, typename U>
bool is_equal_1D_array(T* a, U* b, int dim_1)
{
  /* This function compares two arrays to
     make sure their values are identical.
  */
  for(int i=0; i<dim_1; i++)
      if(!(a[i] == b[i]))
        return false;
  return true;
}

template <typename T, typename U>
bool is_equal_2D_array(T** a, U** b, int dim_1, int dim_2)
{
  /* This function compares two 2D arrays to
     check if they are identical.
  */
  for(int i=0; i<dim_1; i++)
    for(int j=0; j<dim_2; j++)
      if(!(a[i][j] == b[i][j]))
        return false;
  return true;
}

template <typename T, typename U>
bool is_equal_3D_array(T*** a, U*** b, int dim_1, int dim_2, int dim_3)
{
  /* This function compares two 3D arrays to
     check if they are identical.
  */
  for(int i=0; i<dim_1; i++)
    for(int j=0; j<dim_2; j++)
      for(int k=0; k<dim_3; k++)
      if(!(a[i][j][k] == b[i][j][k]))
        return false;
  return true;
}

template <typename T>
void set_1D_array_floating_point_values(T* a, int dim_1)
{
  /* This function fills a 1D array with random
     floating point values.
  */
  std::default_random_engine generator(rand());
  std::uniform_real_distribution<T> distribution;
  for(int i=0; i<dim_1; i++)
    //a[i] = distribution(generator);
    a[i] = 2.0*rand()/RAND_MAX - 1.0;
}

template <typename T>
void set_2D_array_floating_point_values(T** a, int dim_1, int dim_2)
{
  /* This function fills a 2D array with random
     floating point values.
  */
  for(int i = 0; i < dim_1; i++) {
    set_1D_array_floating_point_values<T>(a[i], dim_2);
  }
}

template <typename T>
void set_3D_array_floating_point_values(T*** a, int dim_1, int dim_2, int dim_3)
{
  /* This function fills a 3D array with random floating
     point values.
  */
  for(int i = 0; i < dim_1; i++)
    set_2D_array_floating_point_values<T>(a[i], dim_2, dim_3);
}

template <typename T>
void set_1D_array_integral_values(T* a, int dim_1)
{
  /* This function fills a 1D array with random
     integral values.
  */
  std::default_random_engine generator(rand());
  T t_min = std::numeric_limits<T>::min();
  T t_max = std::numeric_limits<T>::max();
  std::uniform_int_distribution<T> distribution(t_min, t_max);
  for(int i=0; i<dim_1; i++)
    a[i] = distribution(generator);
}

template <typename T>
void set_2D_array_integral_values(T** a, int dim_1, int dim_2)
{
  /* This function fills a 2D array with random
     integral values.
  */
  for(int i = 0; i < dim_1; i++) {
    set_1D_array_integral_values<T>(a[i], dim_2);
  }

}

template <typename T>
void set_3D_array_integral_values(T*** a, int dim_1, int dim_2, int dim_3)
{
  /* This function fills a 3D array with random
     integral values.
  */
  for(int i = 0; i < dim_1; i++)
    set_2D_array_integral_values<T>(a[i], dim_2, dim_3);
}

template <typename T>
T get_integral_scalar()
{
  /* This function returns a random integral
     scalar value.
  */
  std::default_random_engine generator;
  std::uniform_int_distribution<T> distribution;
  return distribution(generator);
}

template <typename T>
T get_floating_point_scalar()
{
  /* This function returns a random floating
     point value.
  */
  std::default_random_engine generator;
  std::uniform_real_distribution<T> distribution;
  return distribution(generator);
}

inline std::string get_prefix()
{
        // get prefix, if it exists. Assumes Client._use_tensor_prefix
    // defaults to true, which it does at time of implementation
    std::string prefix = "";
    char* sskeyin = std::getenv("SSKEYIN");
    std::string sskeyin_str = "";

    if (sskeyin != NULL) {
        // get the first value
        char* a = &sskeyin[0];
        char* b = a;
        char parse_char = ',';
        while (*b) {
            if(*b == parse_char) {
                if (a != b) {
                    sskeyin_str = std::string(a, b - a);
                    break;
                }
                a = ++b;
            }
            else
                b++;
        }
        if (sskeyin_str.length() == 0)
            sskeyin_str = std::string(sskeyin);
        prefix = sskeyin_str + ".";
    }
    return prefix;
}

#endif //SMARTREDIS_TEST_UTILS_H
