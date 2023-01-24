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

#ifndef SMARTREDIS_CTEST_INT32_UTILS_H
#define SMARTREDIS_CTEST_INT32_UTILS_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

void to_lower(char* s, int maxchars) {
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
        if (--maxchars == 0)
            break;
    }
    return;
}

bool use_cluster()
{
    /* This function determines if a cluster
    configuration should be used in the test
    when creating a Client.
    */
    char* smartredis_test_cluster = getenv("SMARTREDIS_TEST_CLUSTER");
    to_lower(smartredis_test_cluster, 256);

    if(smartredis_test_cluster) {
        if(strcmp(smartredis_test_cluster, "true")==0)
            return true;
    }
    return false;
}

void test_result(int result, char *test){
    if (result) {
        fprintf(stdout, "SUCCESS: %c", *test);
        return;
    }
    else {
        fprintf(stderr, "FAILED: %c", *test);
        exit(-1);
    }
}

unsigned safe_rand(){
    unsigned random = (rand() % 254) + 1;
    return random;
}

float rand_float(){
    float random = ((float) safe_rand())/safe_rand();
    return random;
}

double rand_double(){
    double random = ((double) safe_rand())/safe_rand();
    return random;
}

bool is_equal_1D_tensor(void* a, void* b, size_t n_bytes)
{
    return !(memcmp(a, b, n_bytes));
}

bool is_equal_1D_tensor_dbl(double* a, double* b,
                           size_t dim_1)
{
    return is_equal_1D_tensor(a,b, dim_1*sizeof(double));
}

bool is_equal_1D_tensor_flt(float* a, float* b,
                            size_t dim_1)
{
    return is_equal_1D_tensor(a,b, dim_1*sizeof(float));
}

bool is_equal_1D_tensor_i8(int8_t* a, int8_t* b,
                           size_t dim_1)
{
    return is_equal_1D_tensor(a,b, dim_1*sizeof(int8_t));
}

bool is_equal_1D_tensor_i16(int16_t* a, int16_t* b,
                           size_t dim_1)
{
    return is_equal_1D_tensor(a,b, dim_1*sizeof(int16_t));
}

bool is_equal_1D_tensor_i32(int32_t* a, int32_t* b,
                           size_t dim_1)
{
    return is_equal_1D_tensor(a,b, dim_1*sizeof(int32_t));
}

bool is_equal_1D_tensor_i64(int64_t* a, int64_t* b,
                           size_t dim_1)
{
    return is_equal_1D_tensor(a,b, dim_1*sizeof(int64_t));
}

bool is_equal_1D_tensor_ui8(uint8_t* a, uint8_t* b,
                           size_t dim_1)
{
    return is_equal_1D_tensor(a,b, dim_1*sizeof(uint8_t));
}

bool is_equal_1D_tensor_ui16(uint16_t* a, uint16_t* b,
                           size_t dim_1)
{
    return is_equal_1D_tensor(a,b, dim_1*sizeof(uint16_t));
}

bool is_equal_2D_tensors_dbl(double**a, double** b,
                             int dim_1, int dim_2)
{
    /* Compares two 2D tensors for equality
    */
   for(int i=0; i<dim_1; i++) {
       size_t n_bytes = dim_2 * sizeof(double);
       if(!is_equal_1D_tensor(a[i], b[i], n_bytes))
        return false;
   }
   return true;
}

bool is_equal_3D_tensors_dbl(double*** a, double*** b,
                      int dim_1, int dim_2, int dim_3)
{
    /* Compares two 3D tensors for equality
    */
    for(int i=0; i<dim_1; i++) {
        if (!is_equal_2D_tensors_dbl(a[i], b[i], dim_2, dim_3))
            return false;
    }
    return true;
}

#endif //SMARTREDIS_CTEST_INT32_UTILS_H