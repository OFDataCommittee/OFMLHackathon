
!> Retrieve the model from the database
subroutine get_model(this, key, model)
  class(client),                    intent(in   ) :: this  !< An initialized SILC client
  character(len=*),                 intent(in   ) :: key   !< The key to use to place the model
  character(len=*),                 intent(  out) :: model !< The model as a continuous buffer

  character(kind=c_char,len=len_trim(key)) :: c_key
  integer(kind=c_size_t) :: key_length, model_length
  character(kind=c_char), dimension(:), pointer :: f_str_ptr
  type(c_ptr) :: c_str_ptr
  integer :: i

  c_key = trim(key)
  key_length = len_trim(key)

  call get_model_c(this%client_ptr, key, key_length, c_str_ptr, model_length)

  call c_f_pointer(c_str_ptr, f_str_ptr, [ model_length ])

  do i=1,model_length
    model(i:i) = f_str_ptr(i)
  enddo

end subroutine get_model

!> Load the machine learning model from a file and set the configuration
subroutine set_model_from_file( this, key, model_file, backend, device, batch_size, min_batch_size, tag, &
    inputs, outputs )
  class(client),                  intent(in) :: this           !< An initialized SILC client
  character(len=*),               intent(in) :: key            !< The key to use to place the model
  character(len=*),               intent(in) :: model_file     !< The file storing the model
  character(len=*),               intent(in) :: backend        !< The name of the backend (TF, TFLITE, TORCH, ONNX)
  character(len=*),               intent(in) :: device         !< The name of the device (CPU, GPU, GPU:0, GPU:1...)
  integer,                        intent(in) :: batch_size     !< The batch size for model execution
  integer,                        intent(in) :: min_batch_size !< The minimum batch size for model execution
  character(len=*),               intent(in) :: tag            !< A tag to attach to the model for information purposes
  character(len=*), dimension(:), intent(in) :: inputs         !< One or more names of model input nodes (TF models)
  character(len=*), dimension(:), intent(in) :: outputs        !< One or more names of model output nodes (TF models)

  character(kind=c_char) :: c_key(len_trim(key)), c_model_file(len_trim(model_file)), &
                            c_backend(len_trim(backend)), c_device(len_trim(device)), c_tag

  character(kind=c_char, len=:), allocatable, target :: c_inputs(:), c_outputs(:)

  integer(c_size_t), dimension(:), allocatable, target :: input_lengths, output_lengths
  integer(kind=c_size_t) :: key_length, model_file_length, backend_length, device_length, tag_length, n_inputs, &
                            n_outputs
  integer(kind=c_int)    :: c_batch_size, c_min_batch_size
  type(c_ptr)            :: inputs_ptr, input_lengths_ptr, outputs_ptr, output_lengths_ptr
  integer :: i
  integer :: max_length, length

  c_key = trim(key)
  c_model_file = trim(model_file)
  c_backend = trim(backend)
  c_device = trim(device)
  c_tag = trim(tag)

  key_length = len_trim(key)
  model_file_length = len_trim(model_file)
  backend_length = len_trim(backend)
  device_length = len_trim(device)
  tag_length = len_trim(tag)

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

  ! Cast the batch sizes to C integers
  c_batch_size = batch_size
  c_min_batch_size = min_batch_size

  call set_model_from_file_c(this%client_ptr, c_key, key_length, c_model_file, model_file_length,               &
                             c_backend, backend_length, c_device, device_length, batch_size, min_batch_size,    &
                             c_tag, tag_length, inputs_ptr, input_lengths_ptr, n_inputs, outputs_ptr,           &
                             output_lengths_ptr, n_outputs)

  deallocate(c_inputs)
  deallocate(input_lengths)
  deallocate(c_outputs)
  deallocate(output_lengths)
end subroutine set_model_from_file

subroutine set_model( this, key, model, backend, device, batch_size, min_batch_size, tag, &
    inputs, outputs )
  class(client),                  intent(in) :: this           !< An initialized SILC client
  character(len=*),               intent(in) :: key            !< The key to use to place the model
  character(len=*),               intent(in) :: model          !< The binary representaiton o
  character(len=*),               intent(in) :: backend        !< The name of the backend (TF, TFLITE, TORCH, ONNX)
  character(len=*),               intent(in) :: device         !< The name of the device (CPU, GPU, GPU:0, GPU:1...)
  integer,                        intent(in) :: batch_size     !< The batch size for model execution
  integer,                        intent(in) :: min_batch_size !< The minimum batch size for model execution
  character(len=*),               intent(in) :: tag            !< A tag to attach to the model for information purposes
  character(len=*), dimension(:), intent(in) :: inputs         !< One or more names of model input nodes (TF models)
  character(len=*), dimension(:), intent(in) :: outputs        !< One or more names of model output nodes (TF models)

  character(kind=c_char) :: c_key(len_trim(key)), c_model(len_trim(model)), &
                            c_backend(len_trim(backend)), c_device(len_trim(device)), c_tag

  character(kind=c_char, len=:), allocatable, target :: c_inputs(:), c_outputs(:)

  integer(c_size_t), dimension(:), allocatable, target :: input_lengths, output_lengths
  integer(kind=c_size_t) :: key_length, model_length, backend_length, device_length, tag_length, n_inputs, &
                            n_outputs
  integer(kind=c_int)    :: c_batch_size, c_min_batch_size
  type(c_ptr)            :: inputs_ptr, input_lengths_ptr, outputs_ptr, output_lengths_ptr
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
  class(client),                  intent(in) :: this           !< An initialized SILC client
  character(len=*),               intent(in) :: key            !< The key to use to place the model
  character(len=*), dimension(:), intent(in) :: inputs         !< One or more names of model input nodes (TF models)
  character(len=*), dimension(:), intent(in) :: outputs        !< One or more names of model output nodes (TF models)

  character(kind=c_char) :: c_key(len_trim(key))
  character(kind=c_char, len=:), allocatable, target :: c_inputs(:), c_outputs(:)

  integer(c_size_t), dimension(:), allocatable, target :: input_lengths, output_lengths
  integer(kind=c_size_t) :: n_inputs, n_outputs, key_length
  type(c_ptr)            :: inputs_ptr, input_lengths_ptr, outputs_ptr, output_lengths_ptr
  integer :: i
  integer :: max_length, length

  c_key = trim(key)
  key_length = len_trim(key)

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

  call run_model_c(this%client_ptr, c_key, key_length, inputs_ptr, input_lengths_ptr, n_inputs, outputs_ptr, &
                  output_lengths_ptr, n_outputs)

  deallocate(c_inputs)
  deallocate(input_lengths)
  deallocate(c_outputs)
  deallocate(output_lengths)

end subroutine run_model