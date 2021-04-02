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
  subroutine rename_tensor_c(c_client, key, key_length, new_key, new_key_length) bind(c, name="rename_tensor")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    type(c_ptr),            value, intent(in) :: c_client   !< Pointer to the initialized client
    character(kind=c_char),        intent(in) :: key(*)     !< The key to use to place the tensor
    integer(kind=c_size_t), value, intent(in) :: key_length !< The length of the key c-string,
                                                            !! excluding null terminating character
    character(kind=c_char),        intent(in) :: new_key(*) !< The new tensor key
    integer(kind=c_size_t), value, intent(in) :: new_key_length !< The length of the new key c-string,
                                                                !! excluding null terminating character
  end subroutine rename_tensor_c
end interface

interface
  subroutine delete_tensor_c(c_client, key, key_length) bind(c, name="delete_tensor")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    type(c_ptr),            value, intent(in) :: c_client   !< Pointer to the initialized client
    character(kind=c_char),        intent(in) :: key(*)     !< The key of the tensor to delete
    integer(kind=c_size_t), value, intent(in) :: key_length !< The length of the key c-string,
  end subroutine delete_tensor_c
end interface

interface
  subroutine copy_tensor_c(c_client, src_name, src_name_length, dest_name, dest_name_length) &
      bind(c, name="copy_tensor")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    type(c_ptr),            value, intent(in) :: c_client        !< Pointer to the initialized client
    character(kind=c_char),        intent(in) :: src_name(*)     !< The key to use to place the tensor
    integer(kind=c_size_t), value, intent(in) :: src_name_length !< The length of the key c-string,
                                                                 !! excluding null terminating character
    character(kind=c_char),        intent(in) :: dest_name(*)    !< The new tensor key
    integer(kind=c_size_t), value, intent(in) :: dest_name_length !< The length of the new key c-string,
                                                                  !! excluding null terminating character
  end subroutine copy_tensor_c
end interface
