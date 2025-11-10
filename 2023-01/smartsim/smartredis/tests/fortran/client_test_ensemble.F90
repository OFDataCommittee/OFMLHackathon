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
use test_utils,  only : setenv, use_cluster
use iso_fortran_env, only : STDERR => error_unit
use iso_c_binding, only : c_ptr, c_bool, c_null_ptr, c_char, c_int
use iso_c_binding, only : c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_float, c_double, c_size_t

implicit none

#include "enum_fortran.inc"

character(len=255) :: ensemble_keyout, ensemble_keyin
character(len=255) :: tensor_key, model_key, script_key
character(len=255) :: script_file, model_file

real, dimension(10) :: tensor
type(client_type) :: client
integer :: result
logical :: exists

ensemble_keyout = "producer_0"
call setenv("SSKEYIN", "producer_0,producer_1")
call setenv("SSKEYOUT", ensemble_keyout)

result = client%initialize("client_test_ensemble")
if (result .ne. SRNoError) error stop
result = client%use_model_ensemble_prefix(.true.)
if (result .ne. SRNoError) error stop

! Put tensor, script, and model into the database. Then check to make
! sure that keys were prefixed correctly

tensor_key = "ensemble_tensor"
result = client%put_tensor(tensor_key, tensor, shape(tensor))
if (result .ne. SRNoError) error stop
result = client%tensor_exists(tensor_key, exists)
if (result .ne. SRNoError) error stop
if (.not. exists) then
  write(STDERR,*) 'Tensor does not exist: ', tensor_key
  error stop
endif
result = client%key_exists(trim(ensemble_keyout)//"."//trim(tensor_key), exists)
if (result .ne. SRNoError) error stop
if (.not. exists) then
  write(STDERR,*) 'Key does not exist: ', trim(ensemble_keyout)//"."//tensor_key
  error stop
endif

script_key = "ensemble_script"
script_file = "../cpp/mnist_data/data_processing_script.txt"
result = client%set_script_from_file(script_key, "CPU", script_file)
if (result .ne. SRNoError) error stop
result = client%model_exists(script_key, exists)
if (result .ne. SRNoError) error stop
if (.not. exists) then
  write(STDERR,*) 'Script does not exist: ', script_key
  error stop
endif

model_key = "ensemble_model"
model_file = "../cpp/mnist_data/mnist_cnn.pt"
result = client%set_model_from_file(model_key, model_file, "TORCH", "CPU")
if (result .ne. SRNoError) error stop 'set_model_from_file failed'
result = client%model_exists(model_key, exists)
if (result .ne. SRNoError) error stop
if (.not. exists) then
  write(STDERR,*) 'Model does not exist: ', model_key
  error stop
endif

result = client%destructor()
if (result .ne. SRNoError) error stop

! Now set the consumer tests
ensemble_keyout = "producer_1"
call setenv("SSKEYIN", "producer_1,producer_0")
call setenv("SSKEYOUT", ensemble_keyout)

result = client%initialize("client_test_ensemble")
if (result .ne. SRNoError) error stop
result = client%use_model_ensemble_prefix(.true.)
if (result .ne. SRNoError) error stop

! Check that the keys associated with producer_1 do not exist
result = client%tensor_exists(tensor_key, exists)
if (result .ne. SRNoError) error stop
if (exists) then
  write(STDERR,*) "Tensor should not exist: ", tensor_key
  error stop
endif
result = client%key_exists(trim(ensemble_keyout)//"."//trim(tensor_key), exists)
if (result .ne. SRNoError) error stop
if (exists) then
  write(STDERR,*) "Key should not exist: "//trim(ensemble_keyout)//"."//tensor_key
  error stop
endif
result = client%set_data_source("producer_0")
if (result .ne. SRNoError) error stop
result = client%tensor_exists(tensor_key, exists)
if (result .ne. SRNoError) error stop
if (.not. exists) then
  write(STDERR,*) "Tensor does not exist: ", tensor_key
  error stop
endif
result = client%model_exists(model_key, exists)
if (result .ne. SRNoError) error stop
if (.not. exists) then
  write(STDERR,*) "Model does not exist: "//model_key
  error stop
endif
result = client%model_exists(script_key, exists)
if (result .ne. SRNoError) error stop
if (.not. exists) then
  write(STDERR,*) "Script does not exist: "//script_key
  error stop
endif

result = client%delete_model(model_key)
if (result .ne. SRNoError) error stop
result = client%model_exists(model_key, exists)
if (result .ne. SRNoError) error stop
if (exists) then
  write(STDERR,*) "Model exists after deletion: "//model_key
  error stop
endif
result = client%delete_script(script_key)
if (result .ne. SRNoError) error stop
result = client%model_exists(script_key, exists)
if (result .ne. SRNoError) error stop
if (exists) then
  write(STDERR,*) "Script exists after deletion: "//script_key
  error stop
endif


end program main
