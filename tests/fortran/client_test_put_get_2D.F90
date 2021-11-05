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

program main

  use smartredis_client, only : client_type
  use test_utils,  only : irand, use_cluster

  implicit none

  integer, parameter :: dim1 = 10
  integer, parameter :: dim2 = 20

  real(kind=4),    dimension(dim1,dim2) :: recv_array_real_32
  real(kind=8),    dimension(dim1,dim2) :: recv_array_real_64
  integer(kind=1), dimension(dim1,dim2) :: recv_array_integer_8
  integer(kind=2), dimension(dim1,dim2) :: recv_array_integer_16
  integer(kind=4), dimension(dim1,dim2) :: recv_array_integer_32
  integer(kind=8), dimension(dim1,dim2) :: recv_array_integer_64

  real(kind=4),    dimension(dim1,dim2) :: true_array_real_32
  real(kind=8),    dimension(dim1,dim2) :: true_array_real_64
  integer(kind=1), dimension(dim1,dim2) :: true_array_integer_8
  integer(kind=2), dimension(dim1,dim2) :: true_array_integer_16
  integer(kind=4), dimension(dim1,dim2) :: true_array_integer_32
  integer(kind=8), dimension(dim1,dim2) :: true_array_integer_64

  integer :: i, j
  type(client_type) :: client

  integer :: err_code

  call random_number(true_array_real_32)
  call random_number(true_array_real_64)
  do j=1,dim2; do i=1,dim1
    true_array_integer_8(i,j) = irand()
    true_array_integer_16(i,j) = irand()
    true_array_integer_32(i,j) = irand()
    true_array_integer_64(i,j) = irand()
  enddo; enddo

  call random_number(recv_array_real_32)
  call random_number(recv_array_real_64)
  do j=1,dim2; do i=1,dim1
    recv_array_integer_8(i,j) = irand()
    recv_array_integer_16(i,j) = irand()
    recv_array_integer_32(i,j) = irand()
    recv_array_integer_64(i,j) = irand()
  enddo; enddo

  call client%initialize(use_cluster())

  call client%put_tensor("true_array_real_32", true_array_real_32, shape(true_array_real_32))
  call client%unpack_tensor("true_array_real_32", recv_array_real_32, shape(recv_array_real_32))
  if (.not. all(true_array_real_32 == recv_array_real_32)) stop 'true_array_real_32: FAILED'

  call client%put_tensor("true_array_real_64", true_array_real_64, shape(true_array_real_64))
  call client%unpack_tensor("true_array_real_64", recv_array_real_64, shape(recv_array_real_64))
  if (.not. all(true_array_real_64 == recv_array_real_64)) stop 'true_array_real_64: FAILED'

  call client%put_tensor("true_array_integer_8", true_array_integer_8, shape(true_array_integer_8))
  call client%unpack_tensor("true_array_integer_8", recv_array_integer_8, shape(recv_array_integer_8))
  if (.not. all(true_array_integer_8 == recv_array_integer_8)) stop 'true_array_integer_8: FAILED'

  call client%put_tensor("true_array_integer_16", true_array_integer_16, shape(true_array_integer_16))
  call client%unpack_tensor("true_array_integer_16", recv_array_integer_16, shape(recv_array_integer_16))
  if (.not. all(true_array_integer_16 == recv_array_integer_16)) stop 'true_array_integer_16: FAILED'

  call client%put_tensor("true_array_integer_32", true_array_integer_32, shape(true_array_integer_32))
  call client%unpack_tensor("true_array_integer_32", recv_array_integer_32, shape(recv_array_integer_32))
  if (.not. all(true_array_integer_32 == recv_array_integer_32)) stop 'true_array_integer_32: FAILED'

  call client%put_tensor("true_array_integer_64", true_array_integer_64, shape(true_array_integer_64))
  call client%unpack_tensor("true_array_integer_64", recv_array_integer_64, shape(recv_array_integer_64))
  if (.not. all(true_array_integer_64 == recv_array_integer_64)) stop 'true_array_integer_64: FAILED'

  write(*,*) "2D put/get: passed"

end program main
