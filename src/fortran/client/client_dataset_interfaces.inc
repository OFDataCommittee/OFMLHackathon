! BSD 2-Clause License
!
! Copyright (c) 2021, Hewlett Packard Enterprise
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this
!    list of conditions and the following disclaimer.
!
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

interface
  function put_dataset_c( client, dataset ) bind(C, name="put_dataset")
    use iso_c_binding, only : c_ptr
    import :: enum_kind
    integer(kind=enum_kind)        :: put_dataset_c
    type(c_ptr), value, intent(in) :: client  !< Pointer to the initialized C-client
    type(c_ptr), value, intent(in) :: dataset !< Pointer to the dataset
  end function put_dataset_c
end interface
interface
  function get_dataset_c( client, c_name, name_length, dataset ) bind(C, name="get_dataset")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    integer(kind=enum_kind)       :: get_dataset_c
    type(c_ptr),            value :: client !< Pointer to the initialized C-client
    character(kind=c_char)        :: c_name(*) !< Name of the dataset to retrieve from the database
    integer(kind=c_size_t), value :: name_length !< Number of characters in the dataset's name
    type(c_ptr)                   :: dataset !< receives the dataset
  end function get_dataset_c
end interface
interface
  function rename_dataset_c( client, c_name, name_length, c_new_name, new_name_length ) &
    bind(C, name="rename_dataset")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    integer(kind=enum_kind)       :: rename_dataset_c
    type(c_ptr),            value :: client !< Pointer to the initialized C-client
    character(kind=c_char)        :: c_name(*) !< Original name of the dataset
    integer(kind=c_size_t), value :: name_length !< Number of characters in the dataset's name
    character(kind=c_char)        :: c_new_name(*) !< New name of the dataset to in the database
    integer(kind=c_size_t), value :: new_name_length !< Number of characters in the dataset's new name
  end function rename_dataset_c
end interface
interface
  function copy_dataset_c( client, c_name, name_length, c_new_name, new_name_length ) &
    bind(C, name="copy_dataset")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    integer(kind=enum_kind)       :: copy_dataset_c
    type(c_ptr),            value :: client !< Pointer to the initialized C-client
    character(kind=c_char)        :: c_name(*) !< Original name of the dataset
    integer(kind=c_size_t), value :: name_length !< Number of characters in the dataset's name
    character(kind=c_char)        :: c_new_name(*) !< New name of the dataset to in the database
    integer(kind=c_size_t), value :: new_name_length !< Number of characters in the dataset's new name
  end function copy_dataset_c
end interface
interface
  function delete_dataset_c( client, c_name, name_length ) bind(C, name="delete_dataset")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    integer(kind=enum_kind)       :: delete_dataset_c
    type(c_ptr),            value :: client    !< Pointer to the initialized C-client
    character(kind=c_char)        :: c_name(*) !< Name of the dataset to be deleted
    integer(kind=c_size_t), value :: name_length !< Number of characters in the dataset's name
  end function delete_dataset_c
end interface



