program main

  use iso_c_binding
  use smartredis_dataset, only : dataset_type
  use smartredis_client, only : client_type

  implicit none

#include "enum_fortran.inc"

  integer, parameter :: dim1 = 10
  integer, parameter :: dim2 = 20
  integer, parameter :: dim3 = 30

  real(kind=c_float),      dimension(dim1, dim2, dim3) :: recv_array_real_32

  real(kind=c_float),      dimension(dim1, dim2, dim3) :: true_array_real_32

  character(len=16) :: meta_float = 'meta_float'

  real(kind=c_float),  dimension(dim1) :: meta_flt_vec
  real(kind=c_float), dimension(:), pointer :: meta_flt_recv

  integer :: i
  type(dataset_type) :: dataset
  type(client_type) :: client
  integer(kind=enum_kind) :: result

  ! Fill array
  call random_number(true_array_real_32)

  ! Initialize a dataset
  result = dataset%initialize("example_fortran_dataset")
  if (result .ne. SRNoError) stop 'dataset initialization failed'

  ! Add a tensor to the dataset and verify that we can retrieve it
  result = dataset%add_tensor("true_array_real_32", true_array_real_32, shape(true_array_real_32))
  if (result .ne. SRNoError) stop 'dataset%add_tensor failed'
  result = dataset%unpack_dataset_tensor("true_array_real_32", recv_array_real_32, shape(recv_array_real_32))
  if (result .ne. SRNoError) stop 'dataset%unpack_dataset_tensor failed'

  call random_number(meta_flt_vec)

  ! Add metascalars to the dataset and verify that we can retrieve them
  do i=1,dim1
    result = dataset%add_meta_scalar(meta_float, meta_flt_vec(i))
    if (result .ne. SRNoError) stop 'dataset%add_meta_scalar failed'
  enddo
  result = dataset%get_meta_scalars(meta_float, meta_flt_recv)
  if (result .ne. SRNoError) stop 'dataset%get_meta_scalars failed'

  ! Initialize a client
  result = client%initialize(.true.) ! Change .false. to .true. if not using a clustered database
  if (result .ne. SRNoError) stop 'client%initialize failed'

  ! Send the dataset to the database via the client
  result = client%put_dataset(dataset)
  if (result .ne. SRNoError) stop 'client%put_dataset failed'

end program main
