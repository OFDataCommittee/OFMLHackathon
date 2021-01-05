program main

  use mpi
  use iso_c_binding
  use silc_dataset, only : dataset_type

  implicit none

  integer, parameter :: dim1 = 10
  integer, parameter :: dim2 = 20
  integer, parameter :: dim3 = 30

  real(kind=c_float),       dimension(dim1, dim2, dim3) :: recv_array_real_32
  real(kind=c_double),      dimension(dim1, dim2, dim3) :: recv_array_real_64
  integer(kind=c_int8_t),  dimension(dim1, dim2, dim3) :: recv_array_integer_8
  integer(kind=c_int16_t), dimension(dim1, dim2, dim3) :: recv_array_integer_16
  integer(kind=c_int32_t), dimension(dim1, dim2, dim3) :: recv_array_integer_32
  integer(kind=c_int64_t), dimension(dim1, dim2, dim3) :: recv_array_integer_64

  real(kind=c_float),       dimension(dim1, dim2, dim3) :: true_array_real_32
  real(kind=c_double),      dimension(dim1, dim2, dim3) :: true_array_real_64
  integer(kind=c_int8_t),  dimension(dim1, dim2, dim3) :: true_array_integer_8
  integer(kind=c_int16_t), dimension(dim1, dim2, dim3) :: true_array_integer_16
  integer(kind=c_int32_t), dimension(dim1, dim2, dim3) :: true_array_integer_32
  integer(kind=c_int64_t), dimension(dim1, dim2, dim3) :: true_array_integer_64

  character(len=16) :: meta_dbl = 'meta_dbl'
  character(len=16) :: meta_flt = 'meta_flt'
  character(len=16) :: meta_int32 = 'meta_int32'
  character(len=16) :: meta_int64 = 'meta_int64'

  real(kind=c_float),  dimension(dim1) :: meta_flt_vec
  real(kind=c_float), dimension(:), pointer :: meta_flt_recv
  real(kind=c_double), dimension(dim1) :: meta_dbl_vec
  real(kind=c_double), dimension(:), pointer :: meta_dbl_recv
  integer(kind=c_int32_t), dimension(dim1) :: meta_int32_vec
  integer(kind=c_int32_t), dimension(:), pointer :: meta_int32_recv
  integer(kind=c_int64_t), dimension(dim1) :: meta_int64_vec
  integer(kind=c_int64_t), dimension(:), pointer :: meta_int64_recv

  integer :: i, j, k
  type(dataset_type) :: dataset

  integer :: err_code, pe_id

  call MPI_init( err_code )
  call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code)

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

  call dataset%initialize("test")

  ! Test adding and retrieving a tensor of every supported type
  call dataset%add_tensor("true_array_real_32", true_array_real_32, shape(true_array_real_32))
  call dataset%unpack_dataset_tensor("true_array_real_32", recv_array_real_32, shape(recv_array_real_32))
  if (.not. all(true_array_real_32 == recv_array_real_32)) stop 'true_array_real_32: FAILED'

  call dataset%add_tensor("true_array_real_64", true_array_real_64, shape(true_array_real_64))
  call dataset%unpack_dataset_tensor("true_array_real_64", recv_array_real_64, shape(recv_array_real_64))
  if (.not. all(true_array_real_64 == recv_array_real_64)) stop 'true_array_real_64: FAILED'

  call dataset%add_tensor("true_array_integer_8", true_array_integer_8, shape(true_array_integer_8))
  call dataset%unpack_dataset_tensor("true_array_integer_8", recv_array_integer_8, shape(recv_array_integer_8))
  if (.not. all(true_array_integer_8 == recv_array_integer_8)) stop 'true_array_integer_8: FAILED'

  call dataset%add_tensor("true_array_integer_16", true_array_integer_16, shape(true_array_integer_16))
  call dataset%unpack_dataset_tensor("true_array_integer_16", recv_array_integer_16, shape(recv_array_integer_16))
  if (.not. all(true_array_integer_16 == recv_array_integer_16)) stop 'true_array_integer_16: FAILED'

  call dataset%add_tensor("true_array_integer_32", true_array_integer_32, shape(true_array_integer_32))
  call dataset%unpack_dataset_tensor("true_array_integer_32", recv_array_integer_32, shape(recv_array_integer_32))
  if (.not. all(true_array_integer_32 == recv_array_integer_32)) stop 'true_array_integer_32: FAILED'

  call dataset%add_tensor("true_array_integer_64", true_array_integer_64, shape(true_array_integer_64))
  call dataset%unpack_dataset_tensor("true_array_integer_64", recv_array_integer_64, shape(recv_array_integer_64))
  if (.not. all(true_array_integer_64 == recv_array_integer_64)) stop 'true_array_integer_64: FAILED'

  ! Test adding scalar metadata of all supported types to the dataset
  call random_number(meta_dbl_vec)
  call random_number(meta_flt_vec)
  meta_int32_vec(:) = nint(255.*meta_dbl_vec)
  meta_int64_vec(:) = nint(255.*meta_flt_vec)

  do i=1,dim1
    call dataset%add_meta_scalar(meta_dbl, meta_dbl_vec(i))
    call dataset%add_meta_scalar(meta_flt, meta_flt_vec(i))
    call dataset%add_meta_scalar(meta_int32, meta_int32_vec(i))
    call dataset%add_meta_scalar(meta_int64, meta_int64_vec(i))
  enddo

  call dataset%get_meta_scalars(meta_dbl, meta_dbl_recv)
  if (.not. all(meta_dbl_recv == meta_dbl_vec)) stop 'meta_dbl: FAILED'
  call dataset%get_meta_scalars(meta_flt, meta_flt_recv)
  if (.not. all(meta_flt_recv == meta_flt_vec)) stop 'meta_flt: FAILED'
  call dataset%get_meta_scalars(meta_int32, meta_int32_recv)
  if (.not. all(meta_int32_recv == meta_int32_vec)) stop 'meta_int32: FAILED'
  call dataset%get_meta_scalars(meta_int64, meta_int64_recv)
  if (.not. all(meta_int64_recv == meta_int64_vec)) stop 'meta_int64: FAILED'

  call mpi_finalize(err_code)
  write(*,*) "Fortran Dataset tests passed"

end program main