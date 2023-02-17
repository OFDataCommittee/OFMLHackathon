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

module smartredis_logcontext

use iso_c_binding,   only : c_ptr, c_null_ptr, c_char, c_int, c_size_t
use iso_c_binding,   only : c_loc, c_f_pointer
use fortran_c_interop, only : enum_kind

implicit none; private

include 'enum_fortran.inc'
include 'logcontext/logcontext_interfaces.inc'

public :: enum_kind !< The kind of integer equivalent to a C enum. According to C an Fortran
                    !! standards this should be c_int, but is renamed here to ensure that
                    !! users do not have to import the iso_c_binding module into their
                    !! programs

!> Contains a context against to emit log messages
type, public :: logcontext_type
  type(c_ptr) :: logcontext_ptr !< A pointer to the initialized logcontext object

  contains

  !> Initialize a new dataset with a given name
  procedure :: initialize => initialize_logcontext
  !> Access the raw C pointer for the client
  procedure :: get_c_pointer

end type logcontext_type

contains

!> Initialize the logcontext
function initialize_logcontext(self, context) result(code)
  class(logcontext_type), intent(inout) :: self    !< Receives the logcontext
  character(len=*),       intent(in)    :: context !< Context for the logcontext
  integer(kind=enum_kind)               :: code    !< Result of the operation

  ! Local variables
  integer(kind=c_size_t) :: context_length
  character(kind=c_char, len=len_trim(context)) :: c_context

  context_length = len_trim(context)
  c_context = trim(context)

  code = logcontext_constructor(c_context, context_length, self%logcontext_ptr)
end function initialize_logcontext

!> Access the raw C pointer for the logcontext
function get_c_pointer(self)
  type(c_ptr)                        :: get_c_pointer
  class(logcontext_type), intent(in) :: self
  get_c_pointer = self%logcontext_ptr
end function get_c_pointer

end module smartredis_logcontext
