! BSD 2-Clause License
!
! Copyright (c) 2021-2023, Hewlett Packard Enterprise
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this
!    list of conditions and the following disclaimer.
!
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

program main

  use iso_c_binding
  use, intrinsic :: iso_fortran_env, only: stderr => error_unit
  use, intrinsic :: iso_fortran_env, only: stdout => output_unit
  use smartredis_client,  only : client_type
  use smartredis_dataset, only : dataset_type
  use test_utils,         only : irand, use_cluster

  implicit none

#include "enum_fortran.inc"

  integer, parameter :: dim1 = 10
  integer, parameter :: dim2 = 20
  integer, parameter :: dim3 = 30

  real(kind=c_float),      dimension(dim1, dim2, dim3) :: recv_array_real_32
  real(kind=c_double),     dimension(dim1, dim2, dim3) :: recv_array_real_64
  integer(kind=c_int8_t),  dimension(dim1, dim2, dim3) :: recv_array_integer_8
  integer(kind=c_int16_t), dimension(dim1, dim2, dim3) :: recv_array_integer_16
  integer(kind=c_int32_t), dimension(dim1, dim2, dim3) :: recv_array_integer_32
  integer(kind=c_int64_t), dimension(dim1, dim2, dim3) :: recv_array_integer_64

  real(kind=c_float),      dimension(dim1, dim2, dim3) :: true_array_real_32
  real(kind=c_double),     dimension(dim1, dim2, dim3) :: true_array_real_64
  integer(kind=c_int8_t),  dimension(dim1, dim2, dim3) :: true_array_integer_8
  integer(kind=c_int16_t), dimension(dim1, dim2, dim3) :: true_array_integer_16
  integer(kind=c_int32_t), dimension(dim1, dim2, dim3) :: true_array_integer_32
  integer(kind=c_int64_t), dimension(dim1, dim2, dim3) :: true_array_integer_64

  character(len=16) :: str_meta_dbl = 'meta_dbl'
  character(len=16) :: str_meta_flt = 'meta_flt'
  character(len=16) :: str_meta_int32 = 'meta_int32'
  character(len=16) :: str_meta_int64 = 'meta_int64'

  real(kind=c_float),  dimension(dim1) :: meta_flt_vec
  real(kind=c_float), dimension(:), pointer :: meta_flt_recv
  real(kind=c_double), dimension(dim1) :: meta_dbl_vec
  real(kind=c_double), dimension(:), pointer :: meta_dbl_recv
  integer(kind=c_int32_t), dimension(dim1) :: meta_int32_vec
  integer(kind=c_int32_t), dimension(:), pointer :: meta_int32_recv
  integer(kind=c_int64_t), dimension(dim1) :: meta_int64_vec
  integer(kind=c_int64_t), dimension(:), pointer :: meta_int64_recv

  integer, dimension(10) :: dims
  integer :: ndims

  integer :: i, j, k
  type(dataset_type) :: dataset
  type(client_type) :: client

  integer :: err_code
  integer :: result
  integer :: ttype
  integer :: mdtype
  logical :: exists
  character(kind=c_char, len=:), allocatable :: dumpstr

  call random_number(true_array_real_32)
  call random_number(true_array_real_64)
  do k=1,dim3; do j=1,dim2 ; do i=1,dim1
    true_array_integer_8(i,j,k) = irand()
    true_array_integer_16(i,j,k) = irand()
    true_array_integer_32(i,j,k) = irand()
    true_array_integer_64(i,j,k) = irand()
  enddo; enddo; enddo

  call random_number(recv_array_real_32)
  call random_number(recv_array_real_64)
  do k=1,dim3; do j=1,dim2; do i=1,dim1
    recv_array_integer_8(i,j,k) = irand()
    recv_array_integer_16(i,j,k) = irand()
    recv_array_integer_32(i,j,k) = irand()
    recv_array_integer_64(i,j,k) = irand()
  enddo; enddo; enddo

  result = dataset%initialize( "test_dataset" )
  if (result .ne. SRNoError) error stop

  ! Test adding, validating type, and retrieving a tensor of every supported type
  result = dataset%add_tensor("true_array_real_32", true_array_real_32, shape(true_array_real_32))
  if (result .ne. SRNoError) error stop
  result = dataset%get_tensor_type("true_array_real_32", ttype)
  if (result .ne. SRNoError) error stop
  if (ttype .ne. tensor_flt) error stop
  ndims = size(dims)
  result = dataset%get_tensor_dims("true_array_real_32", dims, ndims)
  if (result .ne. SRNoError) error stop
  if (3 .ne. ndims) error stop 'Wrong number of dimensions for true_array_real_32'
  if (dim1 .ne. dims(1)) error stop 'Wrong size dim 1 for true_array_real_32'
  if (dim2 .ne. dims(2)) error stop 'Wrong size dim 2 for true_array_real_32'
  if (dim3 .ne. dims(3)) error stop 'Wrong size dim 3 for true_array_real_32'
  result = dataset%unpack_dataset_tensor("true_array_real_32", recv_array_real_32, shape(recv_array_real_32))
  if (result .ne. SRNoError) error stop
  if (.not. all(true_array_real_32 == recv_array_real_32)) error stop 'true_array_real_32: FAILED'

  result = dataset%add_tensor("true_array_real_64", true_array_real_64, shape(true_array_real_64))
  if (result .ne. SRNoError) error stop
  result = dataset%get_tensor_type("true_array_real_64", ttype)
  if (result .ne. SRNoError) error stop
  if (ttype .ne. tensor_dbl) error stop
  ndims = size(dims)
  result = dataset%get_tensor_dims("true_array_real_64", dims, ndims)
  if (result .ne. SRNoError) error stop
  if (3 .ne. ndims) error stop 'Wrong number of dimensions for true_array_real_64'
  if (dim1 .ne. dims(1)) error stop 'Wrong size dim 1 for true_array_real_64'
  if (dim2 .ne. dims(2)) error stop 'Wrong size dim 2 for true_array_real_64'
  if (dim3 .ne. dims(3)) error stop 'Wrong size dim 3 for true_array_real_64'
  result = dataset%unpack_dataset_tensor("true_array_real_64", recv_array_real_64, shape(recv_array_real_64))
  if (result .ne. SRNoError) error stop
  if (.not. all(true_array_real_64 == recv_array_real_64)) error stop 'true_array_real_64: FAILED'

  result = dataset%add_tensor("true_array_integer_8", true_array_integer_8, shape(true_array_integer_8))
  if (result .ne. SRNoError) error stop
  result = dataset%get_tensor_type("true_array_integer_8", ttype)
  if (result .ne. SRNoError) error stop
  if (ttype .ne. tensor_int8) error stop
  result = dataset%unpack_dataset_tensor("true_array_integer_8", recv_array_integer_8, shape(recv_array_integer_8))
  if (result .ne. SRNoError) error stop
  if (.not. all(true_array_integer_8 == recv_array_integer_8)) error stop 'true_array_integer_8: FAILED'

  result = dataset%add_tensor("true_array_integer_16", true_array_integer_16, shape(true_array_integer_16))
  if (result .ne. SRNoError) error stop
  result = dataset%get_tensor_type("true_array_integer_16", ttype)
  if (result .ne. SRNoError) error stop
  if (ttype .ne. tensor_int16) error stop
  result = dataset%unpack_dataset_tensor("true_array_integer_16", recv_array_integer_16, shape(recv_array_integer_16))
  if (result .ne. SRNoError) error stop
  if (.not. all(true_array_integer_16 == recv_array_integer_16)) error stop 'true_array_integer_16: FAILED'

  result = dataset%add_tensor("true_array_integer_32", true_array_integer_32, shape(true_array_integer_32))
  if (result .ne. SRNoError) error stop
  result = dataset%get_tensor_type("true_array_integer_32", ttype)
  if (result .ne. SRNoError) error stop
  if (ttype .ne. tensor_int32) error stop
  result = dataset%unpack_dataset_tensor("true_array_integer_32", recv_array_integer_32, shape(recv_array_integer_32))
  if (result .ne. SRNoError) error stop
  if (.not. all(true_array_integer_32 == recv_array_integer_32)) error stop 'true_array_integer_32: FAILED'

  result = dataset%add_tensor("true_array_integer_64", true_array_integer_64, shape(true_array_integer_64))
  if (result .ne. SRNoError) error stop
  result = dataset%get_tensor_type("true_array_integer_64", ttype)
  if (result .ne. SRNoError) error stop
  if (ttype .ne. tensor_int64) error stop
  result = dataset%unpack_dataset_tensor("true_array_integer_64", recv_array_integer_64, shape(recv_array_integer_64))
  if (result .ne. SRNoError) error stop
  if (.not. all(true_array_integer_64 == recv_array_integer_64)) error stop 'true_array_integer_64: FAILED'

  ! Test adding scalar metadata of all supported types to the dataset
  call random_number(meta_dbl_vec)
  call random_number(meta_flt_vec)
  meta_int32_vec(:) = nint(255.*meta_dbl_vec)
  meta_int64_vec(:) = nint(255.*meta_flt_vec)

  do i=1,dim1
    result = dataset%add_meta_scalar(str_meta_dbl, meta_dbl_vec(i))
    if (result .ne. SRNoError) error stop
    result = dataset%get_metadata_field_type(str_meta_dbl, mdtype)
    if (result .ne. SRNoError) error stop
    if (mdtype .ne. meta_dbl) error stop
    result = dataset%add_meta_scalar(str_meta_flt, meta_flt_vec(i))
    if (result .ne. SRNoError) error stop
    result = dataset%get_metadata_field_type(str_meta_flt, mdtype)
    if (result .ne. SRNoError) error stop
    if (mdtype .ne. meta_flt) error stop
    result = dataset%add_meta_scalar(str_meta_int32, meta_int32_vec(i))
    if (result .ne. SRNoError) error stop
    result = dataset%get_metadata_field_type(str_meta_int32, mdtype)
    if (result .ne. SRNoError) error stop
    if (mdtype .ne. meta_int32) error stop
    result = dataset%add_meta_scalar(str_meta_int64, meta_int64_vec(i))
    if (result .ne. SRNoError) error stop
    result = dataset%get_metadata_field_type(str_meta_int64, mdtype)
    if (result .ne. SRNoError) error stop
    if (mdtype .ne. meta_int64) error stop
  enddo

  result = dataset%get_meta_scalars(str_meta_dbl, meta_dbl_recv)
  if (result .ne. SRNoError) error stop
  if (.not. all(meta_dbl_recv == meta_dbl_vec)) error stop 'meta_dbl: FAILED'
  result = dataset%get_meta_scalars(str_meta_flt, meta_flt_recv)
  if (result .ne. SRNoError) error stop
  if (.not. all(meta_flt_recv == meta_flt_vec)) error stop 'meta_flt: FAILED'
  result = dataset%get_meta_scalars(str_meta_int32, meta_int32_recv)
  if (result .ne. SRNoError) error stop
  if (.not. all(meta_int32_recv == meta_int32_vec)) error stop 'meta_int32: FAILED'
  result = dataset%get_meta_scalars(str_meta_int64, meta_int64_recv)
  if (result .ne. SRNoError) error stop
  if (.not. all(meta_int64_recv == meta_int64_vec)) error stop 'meta_int64: FAILED'

  ! Test dataset serialization
  dumpstr = dataset%to_string()
  if (dumpstr(1:7) .ne. "DataSet") error stop
  call dataset%print_dataset()
  call dataset%print_dataset(stdout)

  ! test dataset_existence
  result = client%initialize("client_test_dataset")
  if (result .ne. SRNoError) error stop
  result = client%dataset_exists("nonexistent", exists)
  if (result .ne. SRNoError) error stop
  if (exists) error stop 'non-existent dataset: FAILED'
  result = client%poll_dataset("nonexistent", 50, 5, exists)
  if (result .ne. SRNoError) error stop
  if (exists) error stop 'non-existent dataset: FAILED'
  result = client%put_dataset(dataset)
  if (result .ne. SRNoError) error stop
  result = client%dataset_exists("test_dataset", exists)
  if (result .ne. SRNoError) error stop
  if (.not. exists) error stop 'existent dataset: FAILED'
  result = client%poll_dataset("test_dataset", 50, 5, exists)
  if (result .ne. SRNoError) error stop
  if (.not. exists) error stop 'existent dataset: FAILED'


  write(*,*) "Fortran Dataset: passed"

end program main
