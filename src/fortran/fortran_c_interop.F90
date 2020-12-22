module fortran_c_interop

use iso_c_binding, only : c_ptr, c_char, c_size_t, c_loc

implicit none; private

public :: convert_char_array_to_c

contains

!> Returns pointers to the start of each string and lengths for each string in a Fortran character array
subroutine convert_char_array_to_c(character_array_f, character_array_c, string_ptrs, ptr_to_string_ptrs, &
                                   string_lengths, ptr_to_string_lengths, n_strings )
  !> The 2D Fortran character array
  character(len=*),             dimension(:),                      intent(in   ) :: character_array_f
  !> The character array converted to c_character types
  character(kind=c_char,len=:), dimension(:), allocatable, target, intent(  out) :: character_array_c
  !> C-style pointers to the start of each string
  type(c_ptr),                  dimension(:), allocatable, target, intent(  out) :: string_ptrs
  !> A pointer to the the string pointers
  type(c_ptr),                                                     intent(  out) :: ptr_to_string_ptrs
  !> The length of each string
  integer(kind=c_size_t),       dimension(:), allocatable, target, intent(  out) :: string_lengths
  !> Pointer to the array containing the string_lengths
  type(c_ptr),                                                     intent(  out) :: ptr_to_string_lengths
  !> The length of each string
  integer(kind=c_size_t),                                          intent(  out) :: n_strings

  integer :: i, max_length, length

  ! Find the size of the 2D array and allocate some of the 1D arrays
  n_strings= size(character_array_f)
  allocate(string_lengths(n_strings))
  allocate(string_ptrs(n_strings))

  ! Need to find the length of the string, so we can allocate the c_array
  max_length = 0
  do i=1,n_strings
    length = len_trim(character_array_f(i))
    max_length = max(max_length, length)
    string_lengths(i) = length
    print *, character_array_f(i), length
  enddo
  ptr_to_string_lengths = c_loc(string_lengths)
  allocate(character(len=max_length) :: character_array_c(n_strings))

  ! Copy the character into a c_char and create pointers to each of the strings
  do i=1,n_strings
    character_array_c(i) = transfer(character_array_f(i),character_array_c(i))
    string_ptrs(i) = c_loc(character_array_c(i))
  enddo
  ptr_to_string_ptrs = c_loc(string_ptrs)

end subroutine convert_char_array_to_c

end module fortran_c_interop