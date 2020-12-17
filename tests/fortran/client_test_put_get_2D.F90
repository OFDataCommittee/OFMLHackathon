program main

  use mpi
  use silc_client, only : client_type

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

  integer :: err_code, pe_id
  character(len=9) :: key_prefix

  call MPI_init( err_code )
  call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code)
  write(key_prefix, "(A,I6.6)") "pe_",pe_id

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

  call client%initialize(cluster=.true.)

  call client%put_tensor(key_prefix//"true_array_real_32", true_array_real_32, shape(true_array_real_32))
  call client%unpack_tensor(key_prefix//"true_array_real_32", recv_array_real_32, shape(recv_array_real_32))
  if (.not. all(true_array_real_32 == recv_array_real_32)) stop 'true_array_real_32: FAILED'

  call client%put_tensor(key_prefix//"true_array_real_64", true_array_real_64, shape(true_array_real_64))
  call client%unpack_tensor(key_prefix//"true_array_real_64", recv_array_real_64, shape(recv_array_real_64))
  if (.not. all(true_array_real_64 == recv_array_real_64)) stop 'true_array_real_64: FAILED'

  call client%put_tensor(key_prefix//"true_array_integer_8", true_array_integer_8, shape(true_array_integer_8))
  call client%unpack_tensor(key_prefix//"true_array_integer_8", recv_array_integer_8, shape(recv_array_integer_8))
  if (.not. all(true_array_integer_8 == recv_array_integer_8)) stop 'true_array_integer_8: FAILED'

  call client%put_tensor(key_prefix//"true_array_integer_16", true_array_integer_16, shape(true_array_integer_16))
  call client%unpack_tensor(key_prefix//"true_array_integer_16", recv_array_integer_16, shape(recv_array_integer_16))
  if (.not. all(true_array_integer_16 == recv_array_integer_16)) stop 'true_array_integer_16: FAILED'

  call client%put_tensor(key_prefix//"true_array_integer_32", true_array_integer_32, shape(true_array_integer_32))
  call client%unpack_tensor(key_prefix//"true_array_integer_32", recv_array_integer_32, shape(recv_array_integer_32))
  if (.not. all(true_array_integer_32 == recv_array_integer_32)) stop 'true_array_integer_32: FAILED'

  call client%put_tensor(key_prefix//"true_array_integer_64", true_array_integer_64, shape(true_array_integer_64))
  call client%unpack_tensor(key_prefix//"true_array_integer_64", recv_array_integer_64, shape(recv_array_integer_64))
  if (.not. all(true_array_integer_64 == recv_array_integer_64)) stop 'true_array_integer_64: FAILED'

  write(*,*) "2D put/get: passed"

  call mpi_finalize(err_code)

end program main