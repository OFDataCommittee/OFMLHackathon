!> Retrieve the script from the database
subroutine get_script(this, key, script)
  class(client_type),               intent(in   ) :: this  !< An initialized SILC client
  character(len=*),                 intent(in   ) :: key   !< The key to use to place the script
  character(len=*),                 intent(  out) :: script !< The script as a continuous buffer

  character(kind=c_char,len=len_trim(key)) :: c_key
  integer(kind=c_size_t) :: key_length, script_length
  character(kind=c_char), dimension(:), pointer :: f_str_ptr
  type(c_ptr) :: c_str_ptr
  integer :: i

  c_key = transfer(trim(key), c_key)
  key_length = len_trim(key)

  call get_script_c(this%client_ptr, key, key_length, c_str_ptr, script_length)

  call c_f_pointer(c_str_ptr, f_str_ptr, [ script_length ])

  do i=1,script_length
    script(i:i) = f_str_ptr(i)
  enddo

end subroutine get_script

subroutine set_script_from_file( this, key, device, script_file )
  class(client_type),             intent(in) :: this        !< An initialized SILC client
  character(len=*),               intent(in) :: key         !< The key to use to place the script
  character(len=*),               intent(in) :: device      !< The name of the device (CPU, GPU, GPU:0, GPU:1...)
  character(len=*),               intent(in) :: script_file !< The file storing the script

  character(kind=c_char) :: c_key(len_trim(key)), c_script_file(len_trim(script_file)), c_device(len_trim(device))

  integer(kind=c_size_t) :: key_length, script_file_length, device_length

  c_key = transfer(trim(key), c_key)
  c_script_file = transfer(trim(script_file), c_script_file)
  c_device = transfer(trim(device), c_device)

  key_length = len_trim(key)
  script_file_length = len_trim(script_file)
  device_length = len_trim(device)

  call set_script_from_file_c(this%client_ptr, c_key, key_length, c_device, device_length, &
                              c_script_file, script_file_length)

end subroutine set_script_from_file

subroutine set_script( this, key, device, script )
  class(client_type),             intent(in) :: this   !< An initialized SILC client
  character(len=*),               intent(in) :: key    !< The key to use to place the script
  character(len=*),               intent(in) :: device !< The name of the device (CPU, GPU, GPU:0, GPU:1...)
  character(len=*),               intent(in) :: script !< The file storing the script

  character(kind=c_char) :: c_key(len_trim(key)), c_script(len_trim(script)), c_device(len_trim(device))

  integer(kind=c_size_t) :: key_length, script_length, device_length

  c_key    = transfer(trim(key), c_key)
  c_script = transfer(trim(script), c_script)
  c_device = transfer(trim(device), c_device)

  key_length = len_trim(key)
  script_length = len_trim(script)
  device_length = len_trim(device)

  call set_script_c(this%client_ptr, c_key, key_length, c_device, device_length, c_script, script_length)

end subroutine set_script

subroutine run_script(this, key, func, inputs, outputs)
  class(client_type),             intent(in) :: this           !< An initialized SILC client
  character(len=*),               intent(in) :: key            !< The key to use to place the script
  character(len=*),               intent(in) :: func           !< The name of the function in the script to call
  character(len=*), dimension(:), intent(in) :: inputs         !< One or more names of script input nodes (TF scripts)
  character(len=*), dimension(:), intent(in) :: outputs        !< One or more names of script output nodes (TF scripts)

  character(kind=c_char) :: c_key(len_trim(key)), c_func(len_trim(func))
  character(kind=c_char, len=:), allocatable, target :: c_inputs(:), c_outputs(:)

  integer(c_size_t), dimension(:), allocatable, target :: input_lengths, output_lengths
  integer(kind=c_size_t) :: n_inputs, n_outputs, key_length, func_length
  type(c_ptr)            :: inputs_ptr, input_lengths_ptr, outputs_ptr, output_lengths_ptr
  type(c_ptr), dimension(:), allocatable :: ptrs_to_inputs, ptrs_to_outputs

  integer :: i
  integer :: max_length, length

  c_key  = transfer(trim(key), c_key)
  c_func = transfer(trim(func), c_func)

  key_length = len_trim(key)
  func_length = len_trim(func)

  call convert_char_array_to_c( inputs, c_inputs, ptrs_to_inputs, inputs_ptr, input_lengths, input_lengths_ptr, &
                                n_inputs)
  call convert_char_array_to_c( outputs, c_outputs, ptrs_to_outputs, outputs_ptr, output_lengths, &
                                output_lengths_ptr, n_outputs)

  call run_script_c(this%client_ptr, c_key, key_length, c_func, func_length, inputs_ptr, input_lengths_ptr, n_inputs, &
                    outputs_ptr, output_lengths_ptr, n_outputs)

  deallocate(c_inputs)
  deallocate(input_lengths)
  deallocate(c_outputs)
  deallocate(output_lengths)

end subroutine run_script