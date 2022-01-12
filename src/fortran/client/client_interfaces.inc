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
  function c_constructor(cluster, new_client) bind(c, name="SmartRedisCClient")
    use iso_c_binding, only : c_ptr, c_bool
    import :: enum_kind
    integer(kind=enum_kind)     :: c_constructor
    logical(kind=c_bool), value :: cluster !< True if a database cluster is being used
    type(c_ptr)                 :: new_client !< Receives the newly constructed client
  end function c_constructor
end interface

interface
  function c_destructor(client) bind(c, name="DeleteCClient")
    use iso_c_binding, only : c_ptr
    import :: enum_kind
    integer(kind=enum_kind)     :: c_destructor
    type(c_ptr)                 :: client
  end function c_destructor
end interface

interface
  function model_exists_c(client, model_name, model_name_length, exists) bind(c, name="model_exists")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t
    import :: enum_kind
    integer(kind=enum_kind)       :: model_exists_c
    type(c_ptr),            value :: client
    character(kind=c_char)        :: model_name(*)
    integer(kind=c_size_t), value :: model_name_length
    logical(kind=c_bool)          :: exists
  end function model_exists_c
end interface

interface
  function tensor_exists_c(client, tensor_name, tensor_name_length, exists) bind(c, name="tensor_exists")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t
    import :: enum_kind
    integer(kind=enum_kind)       :: tensor_exists_c
    type(c_ptr),            value :: client
    character(kind=c_char)        :: tensor_name(*)
    integer(kind=c_size_t), value :: tensor_name_length
    logical(kind=c_bool)          :: exists
  end function tensor_exists_c
end interface

interface
  function dataset_exists_c(client, dataset_name, dataset_name_length, exists) bind(c, name="dataset_exists")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t
    import :: enum_kind
    integer(kind=enum_kind)       :: dataset_exists_c
    type(c_ptr),            value :: client
    character(kind=c_char)        :: dataset_name(*)
    integer(kind=c_size_t), value :: dataset_name_length
    logical(kind=c_bool)          :: exists
  end function dataset_exists_c
end interface

interface
  function key_exists_c(client, key, key_length, exists) bind(c, name="key_exists")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t
    import :: enum_kind
    integer(kind=enum_kind)       :: key_exists_c
    type(c_ptr),            value :: client
    character(kind=c_char)        :: key(*)
    integer(kind=c_size_t), value :: key_length
    logical(kind=c_bool)          :: exists
  end function key_exists_c
end interface

interface
  function poll_model_c(client, model_name, model_name_length, poll_frequency_ms, num_tries, exists)&
  bind(c, name="poll_model")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t, c_int
    import :: enum_kind
    integer(kind=enum_kind)       :: poll_model_c
    type(c_ptr), value            :: client
    character(kind=c_char)        :: model_name(*)
    integer(kind=c_size_t), value :: model_name_length
    integer(kind=c_int),    value :: poll_frequency_ms
    integer(kind=c_int),    value :: num_tries
    logical(kind=c_bool)          :: exists
  end function poll_model_c
end interface

interface
  function poll_tensor_c(client, tensor_name, tensor_name_length, poll_frequency_ms, num_tries, exists)&
  bind(c, name="poll_tensor")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t, c_int
    import :: enum_kind
    integer(kind=enum_kind)       :: poll_tensor_c
    type(c_ptr), value            :: client
    character(kind=c_char)        :: tensor_name(*)
    integer(kind=c_size_t), value :: tensor_name_length
    integer(kind=c_int),    value :: poll_frequency_ms
    integer(kind=c_int),    value :: num_tries
    logical(kind=c_bool)          :: exists
  end function poll_tensor_c
end interface

interface
  function poll_dataset_c(client, tensor_name, tensor_name_length, poll_frequency_ms, num_tries, exists)&
  bind(c, name="poll_dataset")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t, c_int
    import :: enum_kind
    integer(kind=enum_kind)       :: poll_dataset_c
    type(c_ptr), value            :: client
    character(kind=c_char)        :: tensor_name(*)
    integer(kind=c_size_t), value :: tensor_name_length
    integer(kind=c_int),    value :: poll_frequency_ms
    integer(kind=c_int),    value :: num_tries
    logical(kind=c_bool)          :: exists
  end function poll_dataset_c
end interface

interface
  function poll_key_c(client, key, key_length, poll_frequency_ms, num_tries, exists) bind(c, name="poll_key")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t, c_int
    import :: enum_kind
    integer(kind=enum_kind)       :: poll_key_c
    type(c_ptr), value            :: client
    character(kind=c_char)        :: key(*)
    integer(kind=c_size_t), value :: key_length
    integer(kind=c_int),    value :: poll_frequency_ms
    integer(kind=c_int),    value :: num_tries
    logical(kind=c_bool)          :: exists
  end function poll_key_c
end interface
