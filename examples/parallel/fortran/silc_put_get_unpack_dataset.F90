program main

  use iso_c_binding
  use mpi
  use silc_client, only : client_type
  use silc_dataset, only : dataset_type

  implicit none

  integer, parameter :: dim1 = 10
  integer, parameter :: dim2 = 20
  integer, parameter :: dim3 = 30

  real(kind=c_float),      dimension(dim1, dim2, dim3) :: recv_array_real_32

  real(kind=c_float),      dimension(dim1, dim2, dim3) :: true_array_real_32

  integer :: i, j, k
  type(client_type)  :: client
  type(dataset_type) :: send_dataset, recv_dataset
  character(len=9) :: key_prefix

  integer :: err_code, pe_id

  call MPI_init( err_code )
  call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code)
  write(key_prefix, "(A,I6.6)") "pe_",pe_id

  call client%initialize(.false.)

  call random_number(true_array_real_32)

  call random_number(recv_array_real_32)

  call send_dataset%initialize( key_prefix//"test" )

  call send_dataset%add_tensor("true_array_real_32", true_array_real_32, shape(true_array_real_32))

  call client%put_dataset( send_dataset )
  recv_dataset = client%get_dataset( key_prefix//"test")

  call recv_dataset%unpack_dataset_tensor("true_array_real_32", recv_array_real_32, shape(recv_array_real_32))
  if (.not. all(true_array_real_32 == recv_array_real_32)) stop 'true_array_real_32: FAILED'

  call MPI_finalize(err_code)
  if (pe_id==0) print *, "SILC Fortran MPI put/get/unpack dataset example finished without errorrs."

end program
