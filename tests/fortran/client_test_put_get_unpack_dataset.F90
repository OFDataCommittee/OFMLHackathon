! BSD 2-Clause License
!
! Copyright (c) 2021-2022, Hewlett Packard Enterprise
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
  use smartredis_client,  only : client_type
  use smartredis_dataset, only : dataset_type
  use test_utils,   only : use_cluster, irand

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

  integer :: i, j, k
  type(client_type)  :: client
  type(dataset_type) :: send_dataset, recv_dataset
  integer(kind=enum_kind) :: result
  logical(kind=c_bool) :: exists

  integer :: err_code

  result = client%initialize(use_cluster())
  if (result .ne. SRNoError) stop

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

  result = send_dataset%initialize( "test_dataset" )
  if (result .ne. SRNoError) stop

  result = send_dataset%add_tensor("true_array_real_32", true_array_real_32, shape(true_array_real_32))
  if (result .ne. SRNoError) stop
  result = send_dataset%add_tensor("true_array_real_64", true_array_real_64, shape(true_array_real_64))
  if (result .ne. SRNoError) stop
  result = send_dataset%add_tensor("true_array_integer_8",  true_array_integer_8, shape(true_array_integer_8))
  if (result .ne. SRNoError) stop
  result = send_dataset%add_tensor("true_array_integer_16", true_array_integer_16, shape(true_array_integer_16))
  if (result .ne. SRNoError) stop
  result = send_dataset%add_tensor("true_array_integer_32", true_array_integer_32, shape(true_array_integer_32))
  if (result .ne. SRNoError) stop
  result = send_dataset%add_tensor("true_array_integer_64", true_array_integer_64, shape(true_array_integer_64))
  if (result .ne. SRNoError) stop

  result = client%put_dataset(send_dataset)
  if (result .ne. SRNoError) stop
  result = client%get_dataset("test_dataset", recv_dataset)
  if (result .ne. SRNoError) stop

  result = recv_dataset%unpack_dataset_tensor("true_array_real_32", recv_array_real_32, shape(recv_array_real_32))
  if (result .ne. SRNoError) stop
  if (.not. all(true_array_real_32 == recv_array_real_32)) stop 'true_array_real_32: FAILED'
  result = recv_dataset%unpack_dataset_tensor("true_array_real_64", recv_array_real_64, shape(recv_array_real_64))
  if (result .ne. SRNoError) stop
  if (.not. all(true_array_real_64 == recv_array_real_64)) stop 'true_array_real_64: FAILED'
  result = recv_dataset%unpack_dataset_tensor("true_array_integer_8", recv_array_integer_8, shape(recv_array_integer_8))
  if (result .ne. SRNoError) stop
  if (.not. all(true_array_integer_8 == recv_array_integer_8)) stop 'true_array_integer_8: FAILED'
  result = recv_dataset%unpack_dataset_tensor("true_array_integer_16", recv_array_integer_16, shape(recv_array_integer_16))
  if (result .ne. SRNoError) stop
  if (.not. all(true_array_integer_16 == recv_array_integer_16)) stop 'true_array_integer_16: FAILED'
  result = recv_dataset%unpack_dataset_tensor("true_array_integer_32", recv_array_integer_32, shape(recv_array_integer_32))
  if (result .ne. SRNoError) stop
  if (.not. all(true_array_integer_32 == recv_array_integer_32)) stop 'true_array_integer_32: FAILED'
  result = recv_dataset%unpack_dataset_tensor("true_array_integer_64", recv_array_integer_64, shape(recv_array_integer_64))
  if (result .ne. SRNoError) stop
  if (.not. all(true_array_integer_64 == recv_array_integer_64)) stop 'true_array_integer_64: FAILED'

  print *, "Fortran Client put/get/unpack dataset: passed"

end program
