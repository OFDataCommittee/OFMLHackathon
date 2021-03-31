program main

  use mpi
  use iso_c_binding
  use smartredis_dataset, only : dataset_type

  implicit none

  integer, parameter :: dim1 = 10
  integer, parameter :: dim2 = 20
  integer, parameter :: dim3 = 30

  real(kind=c_float),      dimension(dim1, dim2, dim3) :: recv_array_real_32
 
  real(kind=c_float),      dimension(dim1, dim2, dim3) :: true_array_real_32

  character(len=16) :: meta_flt = 'meta_flt'

  real(kind=c_float),  dimension(dim1) :: meta_flt_vec
  real(kind=c_float), dimension(:), pointer :: meta_flt_recv

  integer :: i, j, k
  type(dataset_type) :: dataset

  integer :: err_code, pe_id
  character(len=9) :: key_prefix

  call MPI_init( err_code )
  call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code)
  write(key_prefix, "(A,I6.6)") "pe_",pe_id

  call random_number(true_array_real_32)
  call random_number(recv_array_real_32)

  call dataset%initialize(key_prefix//"test")

  call dataset%add_tensor("true_array_real_32", true_array_real_32, shape(true_array_real_32))
  call dataset%unpack_dataset_tensor("true_array_real_32", recv_array_real_32, shape(recv_array_real_32))
  if (.not. all(true_array_real_32 == recv_array_real_32)) stop 'true_array_real_32: FAILED'

  call random_number(meta_flt_vec)

  do i=1,dim1
    call dataset%add_meta_scalar(meta_flt, meta_flt_vec(i))
  enddo

  call dataset%get_meta_scalars(meta_flt, meta_flt_recv)

  call mpi_finalize(err_code)
  if (pe_id == 0) write(*,*) "SmartRedis Fortran MPI Dataset example finished without errors."

end program main
