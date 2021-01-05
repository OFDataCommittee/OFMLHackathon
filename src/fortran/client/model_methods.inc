!> Retrieve the model from the database
subroutine get_model(this, key, model)
  class(client_type),               intent(in   ) :: this  !< An initialized SILC client
  character(len=*),                 intent(in   ) :: key   !< The key to use to place the model
  character(len=*),                 intent(  out) :: model !< The model as a continuous buffer

  character(kind=c_char,len=len_trim(key)) :: c_key
  integer(kind=c_size_t) :: key_length, model_length
  character(kind=c_char), dimension(:), pointer :: f_str_ptr
  type(c_ptr) :: c_str_ptr
  integer :: i

  c_key = trim(key)
  key_length = len_trim(key)

  c_str_ptr = get_model_c(this%client_ptr, key, key_length, c_str_ptr, model_length)

  call c_f_pointer(c_str_ptr, f_str_ptr, [ model_length ])

  do i=1,model_length
    model(i:i) = f_str_ptr(i)
  enddo

end subroutine get_model

!> Load the machine learning model from a file and set the configuration
subroutine set_model_from_file( this, key, model_file, backend, device, batch_size, min_batch_size, tag, &
    inputs, outputs )
  class(client_type),             intent(in) :: this           !< An initialized SILC client
  character(len=*),               intent(in) :: key            !< The key to use to place the model
  character(len=*),               intent(in) :: model_file     !< The file storing the model
  character(len=*),               intent(in) :: backend        !< The name of the backend (TF, TFLITE, TORCH, ONNX)
  character(len=*),               intent(in) :: device         !< The name of the device (CPU, GPU, GPU:0, GPU:1...)
  integer,                        optional,  intent(in) :: batch_size     !< The batch size for model execution
  integer,                        optional,  intent(in) :: min_batch_size !< The minimum batch size for model execution
  character(len=*),               optional,  intent(in) :: tag            !< A tag to attach to the model for
                                                                          !! information purposes
  character(len=*), dimension(:), optional,  intent(in) :: inputs         !< One or more names of model input nodes (TF
                                                                          !! models)
  character(len=*), dimension(:), optional,  intent(in) :: outputs        !< One or more names of model output nodes (TF models)

  character(kind=c_char, len=len_trim(key)) :: c_key
  character(kind=c_char, len=len_trim(model_file)) :: c_model_file
  character(kind=c_char, len=len_trim(backend)) :: c_backend
  character(kind=c_char, len=len_trim(device)) :: c_device
  character(kind=c_char, len=:), allocatable :: c_tag

  character(kind=c_char, len=:), allocatable, target :: c_inputs(:), c_outputs(:)
  character(kind=c_char,len=1), target, dimension(1) :: dummy_inputs, dummy_outputs

  integer(c_size_t), dimension(:), allocatable, target :: input_lengths, output_lengths
  integer(kind=c_size_t) :: key_length, model_file_length, backend_length, device_length, tag_length, n_inputs, &
                            n_outputs
  integer(kind=c_int)    :: c_batch_size, c_min_batch_size
  type(c_ptr)            :: inputs_ptr, input_lengths_ptr, outputs_ptr, output_lengths_ptr
  type(c_ptr), dimension(:), allocatable :: ptrs_to_inputs, ptrs_to_outputs

  integer :: i
  integer :: max_length, length

  ! Set default values for the optional inputs
  c_batch_size = 0
  if (present(batch_size)) c_batch_size = batch_size
  c_min_batch_size = 0
  if (present(min_batch_size)) c_min_batch_size = min_batch_size
  if (present(tag)) then
    allocate( character(kind=c_char, len=len_trim(tag)) :: c_tag)
    c_tag = tag
    tag_length = len_trim(tag)
  else
    allocate( character(kind=c_char, len=1) :: c_tag)
    c_tag = ''
    tag_length = 1
  endif

  ! Cast to c_char kind stringts
  c_key = trim(key)
  c_model_file = trim(model_file)
  c_backend = trim(backend)
  c_device = trim(device)

  key_length = len_trim(key)
  model_file_length = len_trim(model_file)
  backend_length = len_trim(backend)
  device_length = len_trim(device)

  dummy_inputs = ''
  if (present(inputs)) then
    call convert_char_array_to_c( inputs, c_inputs, ptrs_to_inputs, inputs_ptr, input_lengths, input_lengths_ptr, &
                                  n_inputs)
  else
    call convert_char_array_to_c( dummy_inputs, c_inputs, ptrs_to_inputs, inputs_ptr, input_lengths, input_lengths_ptr,&
                                  n_inputs)
  endif

  dummy_outputs =''
  if (present(outputs)) then
    call convert_char_array_to_c( outputs, c_outputs, ptrs_to_outputs, outputs_ptr, output_lengths, output_lengths_ptr,&
                                  n_outputs)
  else
    call convert_char_array_to_c( dummy_outputs, c_outputs, ptrs_to_outputs, outputs_ptr, output_lengths, &
                                  output_lengths_ptr, n_outputs)
  endif

  call set_model_from_file_c(this%client_ptr, c_key, key_length, c_model_file, model_file_length,               &
                             c_backend, backend_length, c_device, device_length, c_batch_size, c_min_batch_size,&
                             c_tag, tag_length, inputs_ptr, input_lengths_ptr, n_inputs, outputs_ptr,           &
                             output_lengths_ptr, n_outputs)
  deallocate(c_inputs)
  deallocate(input_lengths)
  deallocate(c_outputs)
  deallocate(output_lengths)
  deallocate(c_tag)
end subroutine set_model_from_file

subroutine set_model( this, key, model, backend, device, batch_size, min_batch_size, tag, &
    inputs, outputs )
  class(client_type),             intent(in) :: this           !< An initialized SILC client
  character(len=*),               intent(in) :: key            !< The key to use to place the model
  character(len=*),               intent(in) :: model          !< The binary representaiton o
  character(len=*),               intent(in) :: backend        !< The name of the backend (TF, TFLITE, TORCH, ONNX)
  character(len=*),               intent(in) :: device         !< The name of the device (CPU, GPU, GPU:0, GPU:1...)
  integer,                        intent(in) :: batch_size     !< The batch size for model execution
  integer,                        intent(in) :: min_batch_size !< The minimum batch size for model execution
  character(len=*),               intent(in) :: tag            !< A tag to attach to the model for information purposes
  character(len=*), dimension(:), intent(in) :: inputs         !< One or more names of model input nodes (TF models)
  character(len=*), dimension(:), intent(in) :: outputs        !< One or more names of model output nodes (TF models)

  character(kind=c_char, len=len_trim(key)) :: c_key
  character(kind=c_char, len=len_trim(model)) :: c_model
  character(kind=c_char, len=len_trim(backend)) :: c_backend
  character(kind=c_char, len=len_trim(device)) :: c_device
  character(kind=c_char, len=len_trim(tag)) :: c_tag

  character(kind=c_char, len=:), allocatable, target :: c_inputs(:), c_outputs(:)

  integer(c_size_t), dimension(:), allocatable, target :: input_lengths, output_lengths
  integer(kind=c_size_t) :: key_length, model_length, backend_length, device_length, tag_length, n_inputs, &
                            n_outputs
  integer(kind=c_int)    :: c_batch_size, c_min_batch_size
  type(c_ptr)            :: inputs_ptr, input_lengths_ptr, outputs_ptr, output_lengths_ptr
  type(c_ptr), dimension(:), allocatable :: ptrs_to_inputs, ptrs_to_outputs

  integer :: i
  integer :: max_length, length

  c_key = trim(key)
  c_model = trim(model)
  c_backend = trim(backend)
  c_device = trim(device)
  c_tag = trim(tag)

  key_length = len_trim(key)
  model_length = len_trim(model)
  backend_length = len_trim(backend)
  device_length = len_trim(device)
  tag_length = len_trim(tag)

  ! Copy the input array into a c_char
  call convert_char_array_to_c( inputs, c_inputs, ptrs_to_inputs, inputs_ptr, input_lengths, input_lengths_ptr, &
                                n_inputs)
  call convert_char_array_to_c( outputs, c_outputs, ptrs_to_outputs, outputs_ptr, output_lengths, &
                                output_lengths_ptr, n_outputs)

  ! Cast the batch sizes to C integers
  c_batch_size = batch_size
  c_min_batch_size = min_batch_size

  call set_model_c(this%client_ptr, c_key, key_length, c_model, model_length, c_backend, backend_length, &
                 c_device, device_length, batch_size, min_batch_size, c_tag, tag_length,                 &
                 inputs_ptr, input_lengths_ptr, n_inputs, outputs_ptr, output_lengths_ptr, n_outputs)

  deallocate(c_inputs)
  deallocate(input_lengths)
  deallocate(c_outputs)
  deallocate(output_lengths)
end subroutine set_model

subroutine run_model(this, key, inputs, outputs)
  class(client_type),             intent(in) :: this           !< An initialized SILC client
  character(len=*),               intent(in) :: key            !< The key to use to place the model
  character(len=*), dimension(:), intent(in) :: inputs         !< One or more names of model input nodes (TF models)
  character(len=*), dimension(:), intent(in) :: outputs        !< One or more names of model output nodes (TF models)

  character(kind=c_char, len=len_trim(key)) :: c_key
  character(kind=c_char, len=:), allocatable, target :: c_inputs(:), c_outputs(:)

  integer(c_size_t), dimension(:), allocatable, target :: input_lengths, output_lengths
  integer(kind=c_size_t) :: n_inputs, n_outputs, key_length
  type(c_ptr)            :: inputs_ptr, input_lengths_ptr, outputs_ptr, output_lengths_ptr
  type(c_ptr), dimension(:), allocatable :: ptrs_to_inputs, ptrs_to_outputs

  integer :: i
  integer :: max_length, length

  c_key = trim(key)
  key_length = len_trim(key)

  call convert_char_array_to_c( inputs, c_inputs, ptrs_to_inputs, inputs_ptr, input_lengths, input_lengths_ptr, &
                                n_inputs)
  call convert_char_array_to_c( outputs, c_outputs, ptrs_to_outputs, outputs_ptr, output_lengths, &
                                output_lengths_ptr, n_outputs)

  call run_model_c(this%client_ptr, c_key, key_length, inputs_ptr, input_lengths_ptr, n_inputs, outputs_ptr, &
                  output_lengths_ptr, n_outputs)

  deallocate(c_inputs)
  deallocate(input_lengths)
  deallocate(c_outputs)
  deallocate(output_lengths)

end subroutine run_model