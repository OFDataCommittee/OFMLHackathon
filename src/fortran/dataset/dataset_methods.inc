subroutine initialize( this )
  class(dataset_type) :: this

  this%dataset= dataset_constructor()
end subroutine initialize
