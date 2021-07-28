program main
  use smartredis_client, only : client_type
  use test_utils, only : use_cluster

  type(client_type) :: client

  if(client%isinitialized()) stop 'client not initialized'

  call client%initialize(use_cluster())

  if(.not. client%isinitialized()) stop 'client is initialized'

  write(*,*) "client initialized: passed"

end program main
