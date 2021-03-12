program main

  use iso_c_binding
  use silc_client,  only : client_type
  use silc_dataset, only : dataset_type
  use test_utils,   only : use_cluster, irand

  implicit none

  integer, parameter :: dim1 = 10
  integer, parameter :: dim2 = 20
  integer, parameter :: dim3 = 30

  real(kind=c_float),      dimension(dim1, dim2, dim3) :: recv_array_real_32
  real(kind=c_double),     dimension(dim1, dim2, dim3) :: recv_array_real_64
  integer(kind=c_int8_t),  dimension(dim1, dim2, dim3) :: recv_array_integer_8
  integer(kind=c_int16_t), dimension(dim1, dim2, dim3) :: recv_array_integer_16
  integer(kind=c_int32_t), dimension(dim1, dim2, dim3) :: recv_array_integer_32
  integer(kind=c_int64_t), dimension(dim1, dim2, dim3) :: recv_array_integer_64

  real(kind=c_float),      dimension(dim1, dim2, dim3) :: true_array_real_32
  real(kind=c_double),     dimension(dim1, dim2, dim3) :: true_array_real_64
  integer(kind=c_int8_t),  dimension(dim1, dim2, dim3) :: true_array_integer_8
  integer(kind=c_int16_t), dimension(dim1, dim2, dim3) :: true_array_integer_16
  integer(kind=c_int32_t), dimension(dim1, dim2, dim3) :: true_array_integer_32
  integer(kind=c_int64_t), dimension(dim1, dim2, dim3) :: true_array_integer_64

  integer :: i, j, k
  type(client_type)  :: client
  type(dataset_type) :: send_dataset, recv_dataset

  integer :: err_code

  call client%initialize(use_cluster())

  call random_number(true_array_real_32)
  call random_number(true_array_real_64)
  do k=1,dim3; do j=1,dim2 ; do i=1,dim1
    true_array_integer_8(i,j,k) = irand()
    true_array_integer_16(i,j,k) = irand()
    true_array_integer_32(i,j,k) = irand()
    true_array_integer_64(i,j,k) = irand()
  enddo; enddo; enddo

  call random_number(recv_array_real_32)
  call random_number(recv_array_real_64)
  do k=1,dim3; do j=1,dim2; do i=1,dim1
    recv_array_integer_8(i,j,k) = irand()
    recv_array_integer_16(i,j,k) = irand()
    recv_array_integer_32(i,j,k) = irand()
    recv_array_integer_64(i,j,k) = irand()
  enddo; enddo; enddo

  call send_dataset%initialize( "test_dataset" )

  call send_dataset%add_tensor("true_array_real_32", true_array_real_32, shape(true_array_real_32))
  call send_dataset%add_tensor("true_array_real_64", true_array_real_64, shape(true_array_real_64))
  call send_dataset%add_tensor("true_array_integer_8",  true_array_integer_8, shape(true_array_integer_8))
  call send_dataset%add_tensor("true_array_integer_16", true_array_integer_16, shape(true_array_integer_16))
  call send_dataset%add_tensor("true_array_integer_32", true_array_integer_32, shape(true_array_integer_32))
  call send_dataset%add_tensor("true_array_integer_64", true_array_integer_64, shape(true_array_integer_64))

  call client%put_dataset( send_dataset )
  recv_dataset = client%get_dataset( "test_dataset" )

  call recv_dataset%unpack_dataset_tensor("true_array_real_32", recv_array_real_32, shape(recv_array_real_32))
  if (.not. all(true_array_real_32 == recv_array_real_32)) stop 'true_array_real_32: FAILED'
  call recv_dataset%unpack_dataset_tensor("true_array_real_64", recv_array_real_64, shape(recv_array_real_64))
  if (.not. all(true_array_real_64 == recv_array_real_64)) stop 'true_array_real_64: FAILED'
  call recv_dataset%unpack_dataset_tensor("true_array_integer_8", recv_array_integer_8, shape(recv_array_integer_8))
  if (.not. all(true_array_integer_8 == recv_array_integer_8)) stop 'true_array_integer_8: FAILED'
  call recv_dataset%unpack_dataset_tensor("true_array_integer_16", recv_array_integer_16, shape(recv_array_integer_16))
  if (.not. all(true_array_integer_16 == recv_array_integer_16)) stop 'true_array_integer_16: FAILED'
  call recv_dataset%unpack_dataset_tensor("true_array_integer_32", recv_array_integer_32, shape(recv_array_integer_32))
  if (.not. all(true_array_integer_32 == recv_array_integer_32)) stop 'true_array_integer_32: FAILED'
  call recv_dataset%unpack_dataset_tensor("true_array_integer_64", recv_array_integer_64, shape(recv_array_integer_64))
  if (.not. all(true_array_integer_64 == recv_array_integer_64)) stop 'true_array_integer_64: FAILED'

  print *, "Fortran Client put/get/unpack dataset: passed"

end program
