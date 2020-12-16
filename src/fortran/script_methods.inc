!> Retrieve the script from the database
subroutine get_script(this, key, script)
  class(client),                    intent(in   ) :: this  !< An initialized SILC client
  character(len=*),                 intent(in   ) :: key   !< The key to use to place the script
  character(len=*),                 intent(  out) :: script !< The script as a continuous buffer

  character(kind=c_char,len=len_trim(key)) :: c_key
  integer(kind=c_size_t) :: key_length, script_length
  character(kind=c_char), dimension(:), pointer :: f_str_ptr
  type(c_ptr) :: c_str_ptr
  integer :: i

  c_key = trim(key)
  key_length = len_trim(key)

  call get_script_c(this%client_ptr, key, key_length, c_str_ptr, script_length)

  call c_f_pointer(c_str_ptr, f_str_ptr, [ script_length ])

  do i=1,script_length
    script(i:i) = f_str_ptr(i)
  enddo

end subroutine get_script

subroutine set_script_from_file( this, key, device, script_file )
  class(client),                  intent(in) :: this        !< An initialized SILC client
  character(len=*),               intent(in) :: key         !< The key to use to place the script
  character(len=*),               intent(in) :: device      !< The name of the device (CPU, GPU, GPU:0, GPU:1...)
  character(len=*),               intent(in) :: script_file !< The file storing the script

  character(kind=c_char) :: c_key(len_trim(key)), c_script_file(len_trim(script_file)), c_device(len_trim(device))

  integer(kind=c_size_t) :: key_length, script_file_length, device_length

  c_key = trim(key)
  c_script_file = trim(script_file)
  c_device = trim(device)

  key_length = len_trim(key)
  script_file_length = len_trim(script_file)
  device_length = len_trim(device)

  call set_script_from_file_c(this%client_ptr, c_key, key_length, c_device, device_length, &
                              c_script_file, script_file_length)

end subroutine set_script_from_file

subroutine set_script( this, key, device, script )
  class(client),                  intent(in) :: this   !< An initialized SILC client
  character(len=*),               intent(in) :: key    !< The key to use to place the script
  character(len=*),               intent(in) :: device !< The name of the device (CPU, GPU, GPU:0, GPU:1...)
  character(len=*),               intent(in) :: script !< The file storing the script

  character(kind=c_char) :: c_key(len_trim(key)), c_script(len_trim(script)), c_device(len_trim(device))

  integer(kind=c_size_t) :: key_length, script_length, device_length

  c_key = trim(key)
  c_script = trim(script)
  c_device = trim(device)

  key_length = len_trim(key)
  script_length = len_trim(script)
  device_length = len_trim(device)

  call set_script_c(this%client_ptr, c_key, key_length, c_device, device_length, c_script, script_length)

end subroutine set_script

subroutine run_script(this, key, func, inputs, outputs)
  class(client),                  intent(in) :: this           !< An initialized SILC client
  character(len=*),               intent(in) :: key            !< The key to use to place the script
  character(len=*),               intent(in) :: func           !< The name of the function in the script to call
  character(len=*), dimension(:), intent(in) :: inputs         !< One or more names of script input nodes (TF scripts)
  character(len=*), dimension(:), intent(in) :: outputs        !< One or more names of script output nodes (TF scripts)

  character(kind=c_char) :: c_key(len_trim(key)), c_func(len_trim(func))
  character(kind=c_char, len=:), allocatable, target :: c_inputs(:), c_outputs(:)

  integer(c_size_t), dimension(:), allocatable, target :: input_lengths, output_lengths
  integer(kind=c_size_t) :: n_inputs, n_outputs, key_length, func_length
  type(c_ptr)            :: inputs_ptr, input_lengths_ptr, outputs_ptr, output_lengths_ptr
  integer :: i
  integer :: max_length, length

  c_key = trim(key)
  c_func = trim(func)

  key_length = len_trim(key)
  func_length = len_trim(func)

  ! Copy the input array into a c_char array
  n_inputs = size(inputs)
  allocate(input_lengths(n_inputs))

  max_length = 0
  do i=1,n_inputs
    length = len_trim(inputs(i))
    input_lengths(i) = length
    max_length = max(max_length, length)
  enddo
  allocate(character(len=max_length) :: c_inputs(n_inputs))
  do i=1,n_inputs
    c_inputs(i) = inputs(i)
  enddo
  inputs_ptr = c_loc(c_inputs)
  input_lengths_ptr = c_loc(input_lengths)

  ! Copy the output array into a c_char array
  n_outputs = size(outputs)
  allocate(output_lengths(n_outputs))
  max_length = 0
  do i=1,n_outputs
    length = len_trim(outputs(i))
    output_lengths(i) = length
    max_length = max(max_length, length)
  enddo
  allocate(character(len=max_length) :: c_outputs(n_outputs))
  do i=1,n_outputs
    c_outputs(i) = outputs(i)
  enddo
  outputs_ptr = c_loc(c_outputs)
  output_lengths_ptr = c_loc(output_lengths)

  call run_script_c(this%client_ptr, c_key, key_length, func, func_length, inputs_ptr, input_lengths_ptr, n_inputs, &
                    outputs_ptr, output_lengths_ptr, n_outputs)

  deallocate(c_inputs)
  deallocate(input_lengths)
  deallocate(c_outputs)
  deallocate(output_lengths)

end subroutine run_script