program main

  use iso_c_binding
  use silc_client, only : client_type
  use silc_dataset, only : dataset_type

  implicit none

  integer, parameter :: dim1 = 10
  integer, parameter :: dim2 = 20
  integer, parameter :: dim3 = 30

  real(kind=c_float),      dimension(dim1, dim2, dim3) :: recv_array_real_32
  real(kind=c_float),      dimension(dim1, dim2, dim3) :: send_array_real_32

  integer :: i, j, k
  type(client_type)  :: client
  type(dataset_type) :: send_dataset, recv_dataset

  integer :: err_code
  
  call client%initialize(.false.)

  call random_number(send_array_real_32)

  call send_dataset%initialize( "example_unpack_fortran" )
  call send_dataset%add_tensor("send_array_real_32", send_array_real_32, shape(send_array_real_32))

  call client%put_dataset( send_dataset )
  recv_dataset = client%get_dataset("example_unpack_fortran")

  call recv_dataset%unpack_dataset_tensor("send_array_real_32", recv_array_real_32, shape(recv_array_real_32))

end program
