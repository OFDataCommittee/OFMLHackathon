!! BSD 2-Clause License
!!
!! Copyright (c) 2021-2022, Hewlett Packard Enterprise
!! All rights reserved.
!!
!! Redistribution and use in source and binary forms, with or without
!! modification, are permitted provided that the following conditions are met:
!!
!! 1. Redistributions of source code must retain the above copyright notice, this
!!    list of conditions and the following disclaimer.
!!
!! 2. Redistributions in binary form must reproduce the above copyright notice,
!!    this list of conditions and the following disclaimer in the documentation
!!    and/or other materials provided with the distribution.
!!
!! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
!! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
!! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
!! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
!! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
!! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
!! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
!! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
!! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
!! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!!

program main

  use iso_c_binding
  use smartredis_client,  only : client_type
  use smartredis_dataset, only : dataset_type
  use test_utils,         only : irand, use_cluster

  implicit none

#include "enum_fortran.inc"

  integer, parameter :: NUM_DATASETS = 5
  integer, parameter :: TENSOR_LENGTH = 10
  character(len=7), parameter :: list_name = "example"
  integer :: i, retrieved_length
  real, dimension(TENSOR_LENGTH,NUM_DATASETS) :: true_vectors
  real, dimension(TENSOR_LENGTH) :: vector, retrieved_vector
  type(dataset_type), dimension(NUM_DATASETS) :: datasets
  type(dataset_type), dimension(:), allocatable :: retrieved_datasets
  type(client_type) :: client
  character(len=12) :: dataset_name
  integer :: result

  result = client%initialize(use_cluster())
  if (result .ne. SRNoError) error stop

  call random_number(true_vectors)

  result = client%delete_list(list_name)

  do i=1,NUM_DATASETS
    write(dataset_name,'(A,I0)') "dataset_",i
    vector(:) = true_vectors(:,i)
    result = datasets(i)%initialize(dataset_name)
    result = datasets(i)%add_tensor('tensor',vector,shape(vector))
    result = client%put_dataset(datasets(i))
    if (result /= SRNoError) then
      write(*,*) "Could not put dataset: ", dataset_name, result
      error stop
    endif
    result = client%append_to_list(list_name, datasets(i))
    if (result /= SRNoError) then
      write(*,*) "Error when appending to list"
      error stop
    endif
  enddo

  result = client%get_list_length( list_name, retrieved_length )
  if (retrieved_length/=num_datasets) then
    write(*,*) "Retrieved number of datasets does not equal the expected value: ", retrieved_length, num_datasets
    error stop
  endif

  result = client%get_datasets_from_list( list_name, retrieved_datasets, retrieved_length)

  ! Check to make sure that the dataset contents are as expected
  do i=1,NUM_DATASETS
    result = datasets(i)%unpack_dataset_tensor("tensor", retrieved_vector, SHAPE(retrieved_vector))
    if (.not. all(retrieved_vector == true_vectors(:,i))) then
      write(*,*) "Retrieved tensor in dataset not equal to original"
      error stop
    endif
  enddo

  write(*,*) "Fortran dataset aggregation: passed"

end program main