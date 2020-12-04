subroutine initialize( this, cluster )
  class(client) :: this
  logical, optional :: cluster !< If true, client uses a database cluster (Default: .false.)

  if (present(cluster)) this%cluster = cluster
  this%client_ptr = c_constructor(this%cluster)

end subroutine initialize

subroutine destructor( this )
  class(client) :: this

  call c_destructor(this%client_ptr)
end subroutine destructor