program main

  use mpi
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

  character(len=16) :: meta_float = 'meta_flt'

  real(kind=c_float),  dimension(dim1) :: meta_flt_vec
  real(kind=c_float), dimension(:), pointer :: meta_flt_recv

  integer :: i, j, k
  type(dataset_type) :: dataset
  type(client_type) :: client
  integer(kind=enum_kind) :: result

  integer :: err_code, pe_id
  character(len=9) :: key_prefix

  ! Initialize MPI
  call MPI_init( err_code )
  call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code)
  write(key_prefix, "(A,I6.6)") "pe_",pe_id

  call random_number(true_array_real_32)
  call random_number(recv_array_real_32)

  ! Initialize a dataset
  result = dataset%initialize(key_prefix//"test")
  if (result .ne. SRNoError) stop 'dataset initialization failed'

  ! Add a tensor to the dataset and verify that we can retrieve it
  result = dataset%add_tensor("true_array_real_32", true_array_real_32, shape(true_array_real_32))
  if (result .ne. SRNoError) stop 'dataset%add_tensor failed'
  result = dataset%unpack_dataset_tensor("true_array_real_32", recv_array_real_32, shape(recv_array_real_32))
  if (result .ne. SRNoError) stop 'dataset%unpack_dataset_tensor failed'
  if (.not. all(true_array_real_32 == recv_array_real_32)) stop 'true_array_real_32: FAILED'

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

  ! Shut down MPI
  call mpi_finalize(err_code)
  if (pe_id == 0) write(*,*) "SmartRedis Fortran MPI Dataset example finished without errors."

end program main
