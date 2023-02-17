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

program mnist_test

  use iso_c_binding
  use smartredis_client, only : client_type
  use test_utils,  only : use_cluster

  implicit none

#include "enum_fortran.inc"

  character(len=*), parameter :: model_key = "mnist_model"
  character(len=*), parameter :: model_file = "../../cpp/mnist_data/mnist_cnn.pt"
  character(len=*), parameter :: script_key = "mnist_script"
  character(len=*), parameter :: script_file = "../../cpp/mnist_data/data_processing_script.txt"
  integer, parameter :: first_gpu = 0
  integer, parameter :: num_gpus = 1
  integer, parameter :: offset = 0

  type(client_type) :: client
  integer :: err_code
  character(len=2) :: key_suffix
  integer :: sr_return_code

  sr_return_code = client%initialize(use_cluster(), &
    "client_test_mnist_multigpu")
  if (sr_return_code .ne. SRNoError) error stop

  sr_return_code = client%set_model_from_file_multigpu(model_key, model_file, "TORCH", first_gpu, num_gpus)
  if (sr_return_code .ne. SRNoError) error stop
  sr_return_code = client%set_script_from_file_multigpu(script_key, script_file, first_gpu, num_gpus)
  if (sr_return_code .ne. SRNoError) error stop

  call run_mnist_multigpu(client, model_key, script_key, offset, first_gpu, num_gpus)

contains

subroutine run_mnist_multigpu(client, model_name, script_name, offset, first_gpu, num_gpus)
  type(client_type), intent(in) :: client
  character(len=*),  intent(in) :: model_name
  character(len=*),  intent(in) :: script_name
  integer,           intent(in) :: offset
  integer,           intent(in) :: first_gpu
  integer,           intent(in) :: num_gpus

  integer, parameter :: mnist_dim1 = 28
  integer, parameter :: mnist_dim2 = 28
  integer, parameter :: result_dim1 = 10
  real, dimension(1,1,mnist_dim1,mnist_dim2) :: array
  real, dimension(1,result_dim1) :: result

  character(len=255) :: in_key
  character(len=255) :: script_out_key
  character(len=255) :: out_key

  character(len=255), dimension(1) :: inputs
  character(len=255), dimension(1) :: outputs
  integer :: call_result

  ! Construct the keys used for the specifiying inputs and outputs
  in_key = "mnist_input"
  script_out_key = "mnist_processed_input"
  out_key = "mnist_processed_input"

  ! Generate some fake data for inference
  call random_number(array)
  call_result = client%put_tensor(in_key, array, shape(array))
  if (call_result .ne. SRNoError) error stop

  ! Prepare the script inputs and outputs
  inputs(1) = in_key
  outputs(1) = script_out_key
  call_result = client%run_script_multigpu(script_name, "pre_process", inputs, outputs, offset, first_gpu, num_gpus)
  if (call_result .ne. SRNoError) error stop
  inputs(1) = script_out_key
  outputs(1) = out_key
  call_result = client%run_model_multigpu(model_name, inputs, outputs, offset, first_gpu, num_gpus)
  if (call_result .ne. SRNoError) error stop
  result(:,:) = 0.
  call_result = client%unpack_tensor(out_key, result, shape(result))
  if (call_result .ne. SRNoError) error stop
  call_result = client%delete_model_multigpu(model_name, first_gpu, num_gpus)
  if (call_result .ne. SRNoError) error stop
  call_result = client%delete_script_multigpu(script_name, first_gpu, num_gpus)
  if (call_result .ne. SRNoError) error stop

  print *, "Result: ", result
  print *, "Fortran test mnist multigpu: passed"

end subroutine run_mnist_multigpu

end program mnist_test
