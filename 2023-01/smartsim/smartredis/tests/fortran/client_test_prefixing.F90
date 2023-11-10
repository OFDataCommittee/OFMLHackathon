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

use smartredis_client, only : client_type
use smartredis_dataset, only : dataset_type
use test_utils,  only : setenv, use_cluster
use iso_fortran_env, only : STDERR => error_unit
use iso_c_binding, only : c_ptr, c_bool, c_null_ptr, c_char, c_int
use iso_c_binding, only : c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_float, c_double, c_size_t

implicit none

#include "enum_fortran.inc"

character(len=255) :: prefix
character(len=255) :: tensor_key, dataset_key, dataset_tensor_key

real, dimension(10) :: tensor
type(client_type) :: client
type(dataset_type) :: dataset
integer :: result
logical :: exists

prefix = "prefix_test"
call setenv("SSKEYIN", prefix)
call setenv("SSKEYOUT", prefix)

result = client%initialize("client_test_prefixing")
if (result .ne. SRNoError) error stop
result = client%use_tensor_ensemble_prefix(.true.)
if (result .ne. SRNoError) error stop
result = client%use_dataset_ensemble_prefix(.true.)
if (result .ne. SRNoError) error stop

! Put tensor and dataset into the database. Then check to make
! sure that keys were prefixed correctly

tensor_key = "test_tensor"
result = client%put_tensor(tensor_key, tensor, shape(tensor))
if (result .ne. SRNoError) error stop
result = client%tensor_exists(tensor_key, exists)
if (result .ne. SRNoError) error stop
if (.not. exists) then
  write(STDERR,*) 'Tensor does not exist: ', tensor_key
  error stop
endif
result = client%key_exists(trim(prefix)//"."//trim(tensor_key), exists)
if (result .ne. SRNoError) error stop
if (.not. exists) then
  write(STDERR,*) 'Key does not exist: ', trim(prefix)//"."//tensor_key
  error stop
endif

dataset_key = "test_dataset"
result = dataset%initialize(dataset_key)
if (result .ne. SRNoError) error stop
dataset_tensor_key = "dataset_tensor"
result = dataset%add_tensor(tensor_key, tensor, shape(tensor))
if (result .ne. SRNoError) error stop
result = client%put_dataset(dataset)
if (result .ne. SRNoError) error stop
result = client%dataset_exists(dataset_key, exists)
if (result .ne. SRNoError) error stop
if (.not. exists) then
  write(STDERR,*) 'Dataset does not exist: ', dataset_key
  error stop
endif
result = client%key_exists(trim(prefix)//".{"//trim(dataset_key)//"}.meta", exists)
if (result .ne. SRNoError) error stop
if (.not. exists) then
  write(STDERR,*) 'Key does not exist: ', trim(prefix)//".{"//trim(dataset_key)//"}.meta"
  error stop
endif


end program main
