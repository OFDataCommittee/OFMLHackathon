program main

  use mpi
  use iso_c_binding
  use smartredis_client, only : client_type

  implicit none

#include "enum_fortran.inc"

  integer, parameter :: dim1 = 10
  integer, parameter :: dim2 = 20
  integer, parameter :: dim3 = 30

  real(kind=8),    dimension(dim1, dim2, dim3) :: recv_array_real_64

  real(kind=c_double),    dimension(dim1, dim2, dim3) :: true_array_real_64

  integer :: i, j, k
  type(client_type) :: client
  integer(kind=enum_kind) :: result

  integer :: err_code, pe_id
  character(len=9) :: key_prefix

  ! Initialize MPI
  call MPI_init( err_code )
  call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code)
  write(key_prefix, "(A,I6.6)") "pe_",pe_id

  call random_number(true_array_real_64)
  call random_number(recv_array_real_64)

  ! Initialize a client
  result = client%initialize(.true.) ! Change .false. to .true. if not using a clustered database
  if (result .ne. SRNoError) stop 'client%initialize failed'

  ! Add a tensor to the database and verify that we can retrieve it
  result = client%put_tensor(key_prefix//"true_array_real_64", true_array_real_64, shape(true_array_real_64))
  if (result .ne. SRNoError) stop 'client%put_tensor failed'
  result = client%unpack_tensor(key_prefix//"true_array_real_64", recv_array_real_64, shape(recv_array_real_64))
  if (result .ne. SRNoError) stop 'client%unpack_tensor failed'
  if (.not. all(true_array_real_64 == recv_array_real_64)) stop 'true_array_real_64: FAILED'

  ! Shut down MPI
  call mpi_finalize(err_code)
  if (pe_id == 0) write(*,*) "SmartRedis MPI Fortran example 3D put/get finished."

end program main
