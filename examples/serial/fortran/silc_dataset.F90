program main

  use iso_c_binding
  use silc_dataset, only : dataset_type

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

  integer :: err_code

  ! Fill array
  call random_number(true_array_real_32)

  call dataset%initialize("example_fortran_dataset")

  call dataset%add_tensor("true_array_real_32", true_array_real_32, shape(true_array_real_32))
  call dataset%unpack_dataset_tensor("true_array_real_32", recv_array_real_32, shape(recv_array_real_32))

  call random_number(meta_flt_vec)

  do i=1,dim1
    call dataset%add_meta_scalar(meta_flt, meta_flt_vec(i))
  enddo

  call dataset%get_meta_scalars(meta_flt, meta_flt_recv)

end program main
