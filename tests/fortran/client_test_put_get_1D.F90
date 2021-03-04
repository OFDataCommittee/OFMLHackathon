program main

  use silc_client, only : client_type
  use test_utils,  only : irand, use_cluster

  implicit none

  integer, parameter :: dim1 = 10

  real(kind=4),    dimension(dim1) :: recv_array_real_32
  real(kind=8),    dimension(dim1) :: recv_array_real_64
  integer(kind=1), dimension(dim1) :: recv_array_integer_8
  integer(kind=2), dimension(dim1) :: recv_array_integer_16
  integer(kind=4), dimension(dim1) :: recv_array_integer_32
  integer(kind=8), dimension(dim1) :: recv_array_integer_64

  real(kind=4),    dimension(dim1) :: true_array_real_32
  real(kind=8),    dimension(dim1) :: true_array_real_64
  integer(kind=1), dimension(dim1) :: true_array_integer_8
  integer(kind=2), dimension(dim1) :: true_array_integer_16
  integer(kind=4), dimension(dim1) :: true_array_integer_32
  integer(kind=8), dimension(dim1) :: true_array_integer_64

  integer :: i
  type(client_type) :: client

  integer :: err_code

  call random_number(true_array_real_32)
  call random_number(true_array_real_64)
  do i=1,dim1
    true_array_integer_8(i) = irand()
    true_array_integer_16(i) = irand()
    true_array_integer_32(i) = irand()
    true_array_integer_64(i) = irand()
  enddo

  call random_number(recv_array_real_32)
  call random_number(recv_array_real_64)
  do i=1,dim1
    recv_array_integer_8(i) = irand()
    recv_array_integer_16(i) = irand()
    recv_array_integer_32(i) = irand()
    recv_array_integer_64(i) = irand()
  enddo

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

  write(*,*) "1D put/get: passed"

end program main
