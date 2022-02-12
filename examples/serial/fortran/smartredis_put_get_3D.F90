program main

  use iso_c_binding
  use smartredis_client, only : client_type

  implicit none

#include "enum_fortran.inc"

  integer, parameter :: dim1 = 10
  integer, parameter :: dim2 = 20
  integer, parameter :: dim3 = 30

  real(kind=8),    dimension(dim1, dim2, dim3) :: recv_array_real_64
  real(kind=c_double),    dimension(dim1, dim2, dim3) :: send_array_real_64

  integer :: i, j, k
  type(client_type) :: client
  integer(kind=enum_kind) :: result

  call random_number(send_array_real_64)

  ! Initialize a client
  result = client%initialize(.true.) ! Change .false. to .true. if not using a clustered database
  if (result .ne. SRNoError) stop 'client%initialize failed'

  ! Send a tensor to the database via the client and verify that we can retrieve it
  result = client%put_tensor("send_array", send_array_real_64, shape(send_array_real_64))
  if (result .ne. SRNoError) stop 'client%put_tensor failed'
  result = client%unpack_tensor("send_array", recv_array_real_64, shape(recv_array_real_64))
  if (result .ne. SRNoError) stop 'client%unpack_tensor failed'

end program main
