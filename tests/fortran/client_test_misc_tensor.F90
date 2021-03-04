!> Tests a variety of client
program main

  use iso_c_binding
  use silc_client,  only : client_type
  use silc_dataset, only : dataset_type
  use test_utils,   only : use_cluster

  implicit none

  type(client_type)  :: client
  type(dataset_type) :: send_dataset
  character(len=9) :: key_prefix

  real, dimension(10,10,10) :: array

  integer :: err_code, pe_id

  write(key_prefix, "(A,I6.6)") "pe_",0

  call client%initialize(use_cluster())
  call client%put_tensor( key_prefix//"test", array, shape(array) )

  call client%rename_tensor( key_prefix//"test", key_prefix//"test_rename" )
  if (.not. client%key_exists( key_prefix//"test_rename" )) stop 'Renamed tensor does not exist'

  call client%copy_tensor( key_prefix//"test_rename", key_prefix//"test_copy" )
  if (.not. client%key_exists( key_prefix//"test_copy" )) stop 'Copied tensor does not exist'

  call client%delete_tensor( key_prefix//"test_copy" )
  if (client%key_exists( key_prefix//"test_copy" )) stop 'Copied tensor incorrectly exists'

  print *, "Fortran Client misc tensor: passed"

end program
