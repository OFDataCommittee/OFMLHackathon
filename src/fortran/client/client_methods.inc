subroutine initialize( this, cluster )
  class(client_type) :: this
  logical, optional :: cluster !< If true, client uses a database cluster (Default: .false.)

  if (present(cluster)) this%cluster = cluster
  this%client_ptr = c_constructor(this%cluster)

end subroutine initialize

subroutine destructor( this )
  class(client_type) :: this

  call c_destructor(this%client_ptr)
end subroutine destructor

!> Check if the specified key exists in the database
logical function key_exists(this, key)
  class(client_type)    :: this
  character(len=*)      :: key

  character(kind=c_char) :: c_key(len_trim(key))
  integer(kind=c_size_t) :: c_key_length

  c_key = trim(key)
  c_key_length = len_trim(key)

  key_exists = key_exists_c( this%client_ptr, c_key, c_key_length)

end function key_exists

!> Repeatedly poll the database until the key exists or the number of tries is exceeded
logical function poll_key( this, key, poll_frequency_ms, num_tries )
  class(client_type)    :: this
  character(len=*) :: key !< Key in the database to poll
  integer          :: poll_frequency_ms !< Frequency at which to poll the database (ms)
  integer          :: num_tries !< Number of times to poll the database before failing

  character(kind=c_char) :: c_key(len_trim(key))
  integer(kind=c_size_t) :: c_key_length
  integer(kind=c_int) :: c_poll_frequency, c_num_tries

  c_key = trim(key)
  c_key_length = len_trim(key)
  c_num_tries = num_tries
  c_poll_frequency = poll_frequency_ms

  poll_key = poll_key_c(this%client_ptr, c_key, c_key_length, c_poll_frequency, c_num_tries)

end function poll_key