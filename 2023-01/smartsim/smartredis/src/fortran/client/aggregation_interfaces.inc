! BSD 2-Clause License
!
! Copyright (c) 2021-2023, Hewlett Packard Enterprise
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
  function use_list_ensemble_prefix_c(c_client, use_prefix) bind(c, name="use_list_ensemble_prefix")
    use iso_c_binding, only : c_ptr, c_bool
    import :: enum_kind
    integer(kind=enum_kind)       :: use_list_ensemble_prefix_c
    type(c_ptr), value            :: c_client !< Pointer to the initialize C client
    logical(kind=c_bool), value   :: use_prefix !< If true, prefix aggregation with the ensemble prefix
  end function use_list_ensemble_prefix_c
end interface

interface
  function append_to_list_c(c_client, list_name, list_name_length, dataset) bind(c, name="append_to_list")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    integer(kind=enum_kind)       :: append_to_list_c
    type(c_ptr), value            :: c_client !< Pointer to the initialize C client
    character(kind=c_char)        :: list_name(*) !< Name of list to append to
    integer(kind=c_size_t), value :: list_name_length !< Number of characters in list name
    type(c_ptr), value            :: dataset !< The dataset to append
  end function append_to_list_c
end interface

interface
  function delete_list_c(c_client, list_name, list_name_length) bind(c, name="delete_list")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    integer(kind=enum_kind)       :: delete_list_c
    type(c_ptr), value            :: c_client !< Pointer to the initialize C client
    character(kind=c_char)        :: list_name(*) !< Name of list to append to
    integer(kind=c_size_t), value :: list_name_length !< Number of characters in list name
  end function delete_list_c
end interface

interface
  function copy_list_c(c_client, src_name, src_name_length, dest_name, dest_name_length) bind(c, name="copy_list")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    integer(kind=enum_kind)       :: copy_list_c
    type(c_ptr), value            :: c_client !< Pointer to the initialize C client
    character(kind=c_char)        :: src_name(*) !< Name of list to copy from
    integer(kind=c_size_t), value :: src_name_length !< Number of characters in source list name
    character(kind=c_char)        :: dest_name(*) !< Name of list to copy to
    integer(kind=c_size_t), value :: dest_name_length !< Number of characters in destination list name
  end function copy_list_c
end interface

interface
  function rename_list_c(c_client, src_name, src_name_length, dest_name, dest_name_length) bind(c, name="rename_list")
    use iso_c_binding, only : c_ptr, c_char, c_size_t
    import :: enum_kind
    integer(kind=enum_kind)       :: rename_list_c
    type(c_ptr), value            :: c_client !< Pointer to the initialize C client
    character(kind=c_char)        :: src_name(*) !< Name of list to rename from
    integer(kind=c_size_t), value :: src_name_length !< Number of characters in source list name
    character(kind=c_char)        :: dest_name(*) !< Name of list to rename to
    integer(kind=c_size_t), value :: dest_name_length !< Number of characters in destination list name
  end function rename_list_c
end interface

interface
  function get_list_length_c(c_client, list_name, list_name_length, result_length) bind(c, name="get_list_length")
    use iso_c_binding, only : c_ptr, c_char, c_size_t, c_int
    import :: enum_kind
    integer(kind=enum_kind)       :: get_list_length_c
    type(c_ptr), value            :: c_client !< Pointer to the initialize C client
    character(kind=c_char)        :: list_name(*) !< Name of list to append to
    integer(kind=c_size_t), value :: list_name_length !< Number of characters in list name
    integer(kind=c_int)           :: result_length !< The length of the list
  end function get_list_length_c
end interface

interface
  function poll_list_length_c(client, name, name_length, list_length, poll_frequency_ms, num_tries, exists) &
    bind(c, name="poll_list_length")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t, c_int
    import :: enum_kind
    integer(kind=enum_kind)       :: poll_list_length_c
    type(c_ptr), value            :: client
    character(kind=c_char)        :: name(*)
    integer(kind=c_size_t), value :: name_length
    integer(kind=c_int),    value :: list_length
    integer(kind=c_int),    value :: poll_frequency_ms
    integer(kind=c_int),    value :: num_tries
    logical(kind=c_bool)          :: exists
  end function poll_list_length_c
end interface

interface
  function poll_list_length_gte_c(client, name, name_length, list_length, poll_frequency_ms, num_tries, exists) &
    bind(c, name="poll_list_length_gte")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t, c_int
    import :: enum_kind
    integer(kind=enum_kind)       :: poll_list_length_gte_c
    type(c_ptr), value            :: client
    character(kind=c_char)        :: name(*)
    integer(kind=c_size_t), value :: name_length
    integer(kind=c_int),    value :: list_length
    integer(kind=c_int),    value :: poll_frequency_ms
    integer(kind=c_int),    value :: num_tries
    logical(kind=c_bool)          :: exists
  end function poll_list_length_gte_c
end interface

interface
  function poll_list_length_lte_c(client, name, name_length, list_length, poll_frequency_ms, num_tries, exists) &
    bind(c, name="poll_list_length_lte")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t, c_int
    import :: enum_kind
    integer(kind=enum_kind)       :: poll_list_length_lte_c
    type(c_ptr), value            :: client
    character(kind=c_char)        :: name(*)
    integer(kind=c_size_t), value :: name_length
    integer(kind=c_int),    value :: list_length
    integer(kind=c_int),    value :: poll_frequency_ms
    integer(kind=c_int),    value :: num_tries
    logical(kind=c_bool)          :: exists
  end function poll_list_length_lte_c
end interface

interface
  function get_datasets_from_list_c(client, list_name, list_name_length, datasets, num_datasets) &
    bind(c, name="get_datasets_from_list")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t, c_int
    import :: enum_kind
    integer(kind=enum_kind)       :: get_datasets_from_list_c
    type(c_ptr), value            :: client
    character(kind=c_char)        :: list_name(*)
    integer(kind=c_size_t), value :: list_name_length
    type(c_ptr), dimension(:)     :: datasets
    integer(kind=c_size_t)        :: num_datasets
  end function get_datasets_from_list_c
end interface

interface
  function get_dataset_list_range_allocated_c(client, list_name, list_name_length, start_index, end_index, &
    datasets) bind(c, name="_get_dataset_list_range_allocated")
    use iso_c_binding, only : c_ptr, c_char, c_bool, c_size_t, c_int
    import :: enum_kind
    integer(kind=enum_kind)       :: get_dataset_list_range_allocated_c
    type(c_ptr), value            :: client
    character(kind=c_char)        :: list_name(*)
    integer(kind=c_size_t), value :: list_name_length
    integer(kind=c_int),    value :: start_index
    integer(kind=c_int),    value :: end_index
    type(c_ptr), value            :: datasets
  end function get_dataset_list_range_allocated_c
end interface
