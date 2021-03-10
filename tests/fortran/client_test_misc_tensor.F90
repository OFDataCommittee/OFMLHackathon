!> Tests a variety of client
program main

  use iso_c_binding
  use silc_client,  only : client_type
  use silc_dataset, only : dataset_type
  use test_utils,   only : use_cluster

  implicit none

  type(client_type)  :: client
  type(dataset_type) :: send_dataset

  real, dimension(10,10,10) :: array

  integer :: err_code

  call client%initialize(use_cluster())
  print *, "Putting tensor"
  call client%put_tensor( "test", array, shape(array) )

  print *, "Renaming tensor"
  call client%rename_tensor( "test", "test_rename" )
  if (.not. client%key_exists( "test_rename" )) stop 'Renamed tensor does not exist'

  call client%copy_tensor( "test_rename", "test_copy" )
  if (.not. client%key_exists( "test_copy" )) stop 'Copied tensor does not exist'

  call client%delete_tensor( "test_copy" )
  if (client%key_exists( "test_copy" )) stop 'Copied tensor incorrectly exists'

  print *, "Fortran Client misc tensor: passed"

end program
