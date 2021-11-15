! BSD 2-Clause License
!
! Copyright (c) 2021, Hewlett Packard Enterprise
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

!> Tests a variety of client
program main

  use iso_c_binding
  use smartredis_client,  only : client_type
  use smartredis_dataset, only : dataset_type
  use test_utils,   only : use_cluster
  use iso_fortran_env, only : STDERR => error_unit

  implicit none

#include "enums/enum_fortran.inc"

  type(client_type)  :: client
  type(dataset_type) :: send_dataset

  real, dimension(10,10,10) :: array

  integer :: err_code
  integer(kind=enum_kind) :: result
  logical(kind=c_bool) :: exists

  result = client%initialize(use_cluster())
  if (result .ne. sr_ok) stop

  print *, "Putting tensor"
  result = client%put_tensor( "test_initial", array, shape(array) )
  if (result .ne. sr_ok) stop

  print *, "Renaming tensor"
  result = client%rename_tensor( "test_initial", "test_rename" )
  if (result .ne. sr_ok) stop
  result = client%key_exists("test_rename", exists)
  if (result .ne. sr_ok) stop
  if (.not. exists) stop 'Renamed tensor does not exist'

  result = client%copy_tensor("test_rename", "test_copy")
  if (result .ne. sr_ok) stop

  result = client%key_exists("test_copy", exists)
  if (result .ne. sr_ok) stop
  if (.not. exists) stop 'Copied tensor does not exist'

  result = client%delete_tensor("test_copy")
  if (result .ne. sr_ok) stop
  result = client%key_exists("test_copy", exists)
  if (result .ne. sr_ok) stop
  if (exists) stop 'Copied tensor incorrectly exists'

  print *, "Fortran Client misc tensor: passed"

end program
