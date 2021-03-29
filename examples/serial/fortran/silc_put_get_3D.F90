program main

  use iso_c_binding
  use silc_client, only : client_type

  implicit none

  integer, parameter :: dim1 = 10
  integer, parameter :: dim2 = 20
  integer, parameter :: dim3 = 30

  real(kind=8),    dimension(dim1, dim2, dim3) :: recv_array_real_64
  real(kind=c_double),    dimension(dim1, dim2, dim3) :: send_array_real_64

  integer :: i, j, k
  type(client_type) :: client

  integer :: err_code, pe_id

  call random_number(send_array_real_64)

  call client%initialize(.false.) ! Change .true. to false if using a clustered database

  call client%put_tensor("send_array", send_array_real_64, shape(send_array_real_64))
  call client%unpack_tensor("send_array", recv_array_real_64, shape(recv_array_real_64))

end program main
