***************
Data Structures
***************

RedisAI defines three new data structures to be
used in redis databases: tensor, model, and script.
In addition, SmartRedis defines an additional data
structure ``DataSet``.  In this section, the SmartRedis
API for interacting with these data structures
will be described, and when applicable,
comments on performance and best practices will be made.

In general, concepts and capabilities will be
demonstrated for the Python and C++ API.
The C and Fortran function signatures closely
resemble the C++ API, and as a result,
they are not discussed in detail in the interest
of brevity.  For more detailed explanations of the C
and Fortran API, refer to the documentation pages for those
clients.


.. _data_structures_tensor:

Tensor
======

An n-dimensional tensor is used by RedisAI to store and
manipulate numerical data. SmartRedis provides functions to
put a key and tensor pair into the Redis database and retrieve
a tensor associated with a key from the database.

.. note::
    When utilizing SmartRedis with SmartSim ensemble functionality,
    the name provided by the user may be manipulated before placement
    in or retrieval from the database in order to avoid key collisions
    between ensemble members.  Therefore, when using SmartSim ensemble
    functionality, retrieval of a tensor using the Redis command line
    interface (CLI) will require adapting the tensor name.

Sending
-------

In Python, the ``Client`` infers the type and dimensions of the
tensor from the NumPy array data structure, and as a result,
only the name and NumPy array are needed to place a key and tensor
pair in the database.  Currently, only NumPy arrays
are supported as inputs and outputs of Python ``Client``
tensor functions.

.. code-block:: python

    # Python put_tensor() interface
    put_tensor(self, name, data)

In the compiled clients, more information is needed to inform the
``Client`` about the tensor properties.  In the C++ API,
the dimensions of the tensor are provided via a
``std::vector<size_t>`` input parameter.  Additionally, the type
associated with the tensor data (e.g. ``double``, ``float``)
is specified with a ``SRTensorType`` enum.
Finally, the ``Client`` must know the memory
layout of the provided tensor data in order to traverse the
tensor data to generate a tensor data buffer. In C++, tensor
data can either be in a contiguous memory layout or in a nested,
non-contiguous memory layout (i.e. nested pointer arrays to
underlying allocated memory). The memory layout of the tensor
data to place in the database is specified
with a ``SRMemoryLayout`` enum input parameter.

.. code-block:: cpp

    // C++ put_tensor interface
    void put_tensor(const std::string& name,
                    void* data,
                    const std::vector<size_t>& dims,
                    const SRTensorType type,
                    const SRMemoryLayout mem_layout);

C and Fortran have similar function prototypes compared
to the C++ client, except the C client uses only C data
types and the Fortran client does not require the
specification of the tensor memory layout because it is
assumed that Fortran array memory is allocated in a column-major,
contiguous manner.

Retrieving
----------

The C++, C, and Fortran clients provide two methods for retrieving
tensors from the Redis database. The first method is referred to
as *unpacking* a tensor.  When a tensor is retrieved via
``unpack_tensor()``, the memory space to store the retrieved
tensor data is provided by the user. This has the advantage
of giving the user the ability to manage the scope of the retrieved
tensor allocated memory and reuse application memory.

The C++ function signature for ``unpack_tensor()`` is shown below.
In the case of ``unpack_tensor()`` the parameters ``dims``,
``type``, and ``mem_layout`` are used to specify the
characteristics of the user-provided memory space.
The type and dimensions are compared to the tensor that is retrieved
from the database, and if the type does not match or if the
allocated space is insufficient,
an error will be thrown.  Otherwise, the memory space pointed
to by the ``data`` pointer will be filled consistent with the
specified memory layout.

.. code-block:: cpp

    // C++ unpack_tensor() interface
    void unpack_tensor(const std::string& name,
                       void* data,
                       const std::vector<size_t>& dims,
                       const SRTensorType type,
                       const SRMemoryLayout mem_layout);

.. note::

    When using ``unpack_tensor()`` with a user-provided
    ``SRMemLayoutContiguous`` memory space,
    the provided dimensions should be a
    ``std::vector<size_t>`` with a single value
    equal the total number of allocated
    values in the memory space, not the expected
    dimensions of the retrieved tensor.

The other option for retrieving a tensor with the
C++, C, and Fortran clients is ``get_tensor()``.
With ``get_tensor()``, it is assumed that the user does not
know the dimensions or type of the tensor, and as a result, the
``Client`` allocates and manages memory necessary for the retrieved
tensor data.  The C++ function signature for ``get_tensor()`` is shown
below.  Note that a pointer to the newly allocated data, tensor
dimensions, and tensor type are returned to the user via
modifying referenced variables that the user declares before the
``get_tensor()`` call.  This is done to provide a similar
experience across the C++, C, and Fortran clients.

.. code-block:: cpp

    // C++ get_tensor interface
    void get_tensor(const std::string& name,
                    void*& data,
                    std::vector<size_t>& dims,
                    SRTensorType& type,
                    const SRMemoryLayout mem_layout);

.. note::
    Memory allocated by C++, C, and Fortran
    ``Client`` during a ``get_tensor()``
    call will be valid and not freed until the ``Client``
    object is destroyed.  Therefore, if the type and dimensions
    of the tensor are known, it is recommended that
    ``unpack_tensor()`` is used in memory-constrained situations.

The Python client currently only offers a ``get_tensor()`` option for
retrieving tensors.  In this methodology, a NumPy array is returned
to the user, and the only required input to the function is the
name of the tensor to retrieve because its type and dimensions
are embedded in the NumPy array object. The Python interface for
``get_tensor()`` is shown below.  In the Python implementation of
``get_tensor()``, the memory associated with the retrieved tensor
will be freed when the NumPy array goes out of scope or is deleted.

.. code-block:: python

    # Python get_tensor() interface
    get_tensor(self, name):

Note that all of the client ``get_tensor()`` functions will internally
modify the provided tensor name if the client is being used with
SmartSim ensemble capabilities.

.. _data_structures_dataset:

Dataset
=======

In many situations, a ``Client``  might be tasked with sending a
group of tensors and metadata which are closely related and
naturally grouped into a collection for future retrieval.
The ``DataSet`` object stages these items so that they can be
more efficiently placed in the redis database and can later be
retrieved with the name given to the ``DataSet``.

Listed below are the supported tensor and metadata types.
In the following sections, building, sending, and retrieving
a ``DataSet`` will be described.

.. list-table:: Supported Data Types
   :widths: 25 25 25
   :header-rows: 1
   :align: center

   * - Data Type
     - Tensor (n-dim arrays)
     - Metadata (1-D arrays)
   * - Float
     - X
     - X
   * - Double
     - X
     - X
   * - Int64
     - X
     - X
   * - Int32
     - X
     -
   * - Int16
     - X
     -
   * - Int8
     - X
     -
   * - UInt64
     -
     - X
   * - UInt32
     -
     - X
   * - UInt16
     - X
     -
   * - UInt8
     - X
     -
   * - String
     -
     - X

Sending
-------

When building a ``DataSet`` to be stored in the database,
a user can add any combination of tensors and metadata.
To add a tensor to the ``DataSet``, the user simply uses
the ``DataSet.add_tensor()`` function defined in
each language.  The ``DataSet.add_tensor()`` parameters are the same
as ``Client.put_tensor()``, and as a result, details of the function
signatures will not be reiterated here.

.. note::
    ``DataSet.add_tensor()`` copies the tensor data
    provided by the user to eliminate errors from user-provided
    data being cleared or deallocated. This additional memory
    will be freed when the DataSet
    object is destroyed.

Metadata can be added to the ``DataSet`` with the
``DataSet.add_meta_scalar()`` and ``DataSet.add_meta_string()``
functions.  As the aforementioned function names suggest,
there are separate functions to add metadata that is a scalar
(e.g. double) and a string. For both functions, the first
function input is the name of the metadata field.  This field
name is an internal ``DataSet`` identifier for the metadata
value(s) that is used for future retrieval, and because it
is an internal identifier, the user does not have to worry
about any key conflicts in the database (i.e. multiple ``DataSet``
can have the same metadata field names).  To clarify these
and future descriptions, the C++ interface for adding
metadata is shown below:

.. code-block:: cpp

    // C++ add_meta_scalar() interface
    void add_meta_scalar(const std::string& name,
                         const void* data,
                         const SRMetaDataType type);

    // C++ add_meta_string() interface
    void add_meta_string(const std::string& name,
                         const std::string& data);


When adding a scalar or string metadata value, the value
is copied by the ``DataSet``, and as a result, the user
does not need to ensure that the metadata values provided
are still in memory after they have been added.
Additionally, multiple metadata values can be added to a
single field, and the default behavior is to append the value to the
existing field.  In this way, the ``DataSet`` metadata supports
one-dimensional arrays, but the entries in the array must be added
iteratively by the user.  Also, note that in the above C++ example,
the metadata scalar type must be specified with a
``SRMetaDataType`` enum value, and similar
requirements exist for C and Fortran ``DataSet`` implementations.

Finally, the ``DataSet`` object is sent to the database using the
``Client.put_dataset()`` function, which is uniform across all clients.


Retrieving
----------

In all clients, the ``DataSet`` is retrieved with a single
function call to ``Client.get_dataset()``, which requires
only the name of the ``DataSet`` (i.e. the name used
in the constructor of the ``DataSet`` when it was
built and placed in the database).  ``Client.get_dataset()``
returns to the user a DataSet object or a pointer to a
DataSet object that can be used to access all of the
dataset tensors and metadata.

The functions for retrieving tensors from ``DataSet``
are identical to the functions provided by ``Client``,
and the same return values and memory management
paradigm is followed.  As a result, please refer to
the previous section for details on tensor retrieve
function calls.

There are two functions for retrieving metadata:
``get_meta_scalars()`` and ``get_meta_strings()``.
As the names suggest, the first function
is used for retrieving numerical metadata values,
and the second is for retrieving metadata string
values.  The metadata retrieval function prototypes
vary across the clients based on programming language constraints,
and as a result, please refer to the ``DataSet`` API documentation
for a description of input parameters and memory management.  It is
important to note, however, that all functions require the name of the
metadata field to be retrieved, and this name is the same name that
was used when constructing the metadata field with
``add_meta_scalar()`` and ``add_meta_string()`` functions.

Aggregating
-----------

An API is provided to aggregate multiple ``DataSet`` objects that
are stored on one or more database nodes.  This is accomplished
through an interface referred to as ``aggregation lists``.
An ``aggregation list`` in SmartRedis stores references to
``DataSet`` objects that are stored in the database.  ``DataSet``
objects can be appended to the ``aggregation list`` and then
``SmartRedis`` clients in the same application or a different application
can retrieve all or some of the ``DataSet`` objects referenced in that
``aggregation list``.

For example, the C++ client function to append a ``DataSet`` to an
aggregation list is shown below:

.. code-block:: cpp

    # C++ aggregation list append interface
    void append_to_list(const std::string& list_name,
                        const DataSet& dataset);

The above function will append the provided ``DataSet`` to the
``aggregation list``, which can be referenced in all user-facing functions
by the provided list name.  Note that a list can be appended by
any client in the same or different application.  Additionally, all
appends are performed at the end of the list, and if the list does not
already exist, it is automatically created.

For retrieval of ``aggregation list`` contents,
the SmartRedis ``Client`` method provides an API function that
will return an iterable container with all of the ``DataSet`` objects
that were appended to the ``aggregation list``.  For example, the C++ client
function to retrieve the ``aggregation list`` contents is shown below:

.. code-block:: cpp

    # C++ aggregation list retrieval interface
    std::vector<DataSet> get_datasets_from_list(const std::string& list_name);

Additional functions are provided to retrieve only a portion of the
``aggregation list`` contents, copy an ``aggregation list``, rename
an ``aggregation list` and retrieve ``aggregation list`` length.
A blocking method to poll the ``aggregation list`` length is also
provided as a means to wait for list completion before performing
another task in the same application or a separate application.


Model
=====

Like tensors, the RedisAI model data structure is exposed to users
through ``Client`` function calls to place a model in the database,
retrieve a model from the database, and run a model.  Note that
RedisAI supports PyTorch, TensorFlow, TensorFlow Lite, and ONNX
backends, and specifying the backend to be used is done
through the ``Client`` function calls.

Sending
-------

A model is placed in the database through the ``Client.set_model()``
function.  While data types may differ, the function parameters
are uniform across all SmartRedis clients, and as an example, the C++
``set_model()`` function is shown below.

.. code-block:: cpp

    # C++ set_model interface
    void set_model(const std::string& name,
                   const std::string_view& model,
                   const std::string& backend,
                   const std::string& device,
                   int batch_size = 0,
                   int min_batch_size = 0,
                   const std::string& tag = "",
                   const std::vector<std::string>& inputs
                       = std::vector<std::string>(),
                   const std::vector<std::string>& outputs
                       = std::vector<std::string>());

All of the parameters in ``set_model()`` follow the RedisAI
API for the the RedisAI ``AI.MODELSET`` command, and as a result,
the reader is encouraged to read the SmartRedis client code
documentation or the RedisAI documentation for a description
of each parameter.

.. note::
    With a Redis cluster configuration, ``Client.set_model()``
    will distribute a copy of the model to each database node in the
    cluster.  As a result, the model that has been
    placed in the cluster with ``Client.set_model()``
    will not be addressable directly with the Redis CLI because
    of key manipulation that is required to accomplish
    this distribution.  Despite the internal key
    manipulation, models in a Redis cluster that have been
    set through the SmartRedis ``Client`` can be accessed
    and run through the SmartRedis ``Client`` API
    using the name provided to ``set_model()``.  The user
    does not need any knowledge of the cluster model distribution
    to perform RedisAI model actions.  Moreover,
    a model set by one SmartRedis client (e.g. Python) on a Redis
    cluster is addressable with the same name through another
    client (e.g. C++).

Finally, there is a similar function in each client,
``Client.set_model_from_file()``, that will read a
model from file and set it in the database.

Retrieving
----------

A model can be retrieved from the database using the
``Client.get_model()`` function.  While the return
type varies between languages, only the model name
that was used with ``Client.set_model()`` is needed
to reference the model in the database.  Note that
in a Redis cluster configuration, only one copy of the
model is returned to the user.

.. note::

    ``Client.get_model()`` will allocate memory to retrieve
    the model from the database, and this memory will not
    be freed until the Client object is destroyed.

Executing
---------

A model can be executed using the ``Client.run_model()`` function.
The only required inputs to execute a model are the model name,
a list of input tensor names, and a list of output tensor names.
If using a Redis cluster configuration, a copy of the model
referenced by the provided name will be chosen based on data locality.
It is worth noting that the names of input and output tensors will be
altered with ensemble member identifications if the SmartSim
ensemble compatibility features are used.

.. note::

    DataSet tensors can be used as ``run_model()`` input tensors,
    but the name provided to ``run_model()`` must be prefixed with
    the ``DataSet`` name in the pattern ``{dataset_name}.tensor_name``.

Support on Systems with Multiple GPUs
-------------------------------------

SmartRedis has special support for models on systems with multiple GPUs.
On these systems, the model can be set via the ``Client.set_model_multigpu()``
function, which differs from the ``Client.set_model()`` function only in that
(1) there is no need to specify a device (GPU is implicit) and (2) the caller
must supply the index of the first GPU to use with the model and the total
number of GPUs on the system's nodes to use with the model. The function will
then create separate copies of the model for each GPU by appending ``.GPU:n``
to the supplied name, where ``n`` is a number from ``first_gpu`` to
``first_gpu + num_gpus - 1``, inclusive.

Executing models on systems with multiple GPUs may be done via the
``Client.run_model_multigpu()`` function. This method parallels
``Client.run_model()`` except that it requires three additional parameters:
the first GPU to use for execution, the number of GPUs to use for execution,
and an offset for the currently executing thread or image. The model execution
is then dispatched to the copy of the script on the GPU corresponding to
``first_gpu`` plus the offset modulo ``num_gpus``.  The image offset may
be an MPI rank, or a thread ID, or any other indexing scheme.

Finally, models stored for multiple GPUs may be deleted via the
``Client.delete_model_multigpu()`` function. This method parallels
``Client.delete_model()`` except that it requires two additional parameters:
the first GPU and the number of GPUs that the model was stored with. This
function will delete all the extra copies of the model that were stored
via ``Client.set_model_multigpu()``.

.. note::

    In order for a model to be executed via ``Client.run_model_multigpu()``,
    or deleted via ``Client.delete_model_multigpu()``,
    it must have been set via ``Client.set_model_multigpu()``. The
    ``first_gpu`` and ``num_gpus`` parameters must be constant across both calls.

Script
======

Data processing is an essential step in most machine
learning workflows.  For this reason, RedisAI provides
the ability to evaluate PyTorch programs using the hardware
co-located with the Redis database (either CPU or GPU).
The SmartRedis ``Client`` provides functions for users to
place a script in the database, retrieve a script from the
database, and run a script.

Sending
-------

A script is placed in the database through the ``Client.set_script()``
function.  While data types may differ, the function parameters
are uniform across all SmartRedis clients, and as an example, the C++
``set_script()`` function is shown below.  The function signature
is quite simple for placing a script in the database, only
a name for the script, hardware for execution, and the script text
need to be provided by the user.

.. code-block:: cpp

    void set_script(const std::string& name,
                    const std::string& device,
                    const std::string_view& script);

.. note::
    With a Redis cluster configuration, ``Client.set_script()``
    will distribute a copy of the script to each database node in the
    cluster.  As a result, the script that has been
    placed in the cluster with ``Client.set_script()``
    will not be addressable directly with the Redis CLI because
    of key manipulation that is required to accomplish
    this distribution.  Despite the internal key
    manipulation, scripts in a Redis cluster that have been
    set through the SmartRedis ``Client`` can be accessed
    and run through the SmartRedis ``Client`` API
    using the name provided to ``set_script()``.  The user
    does not need any knowledge of the cluster script distribution
    to perform RedisAI script actions.  Moreover,
    a script set by one SmartRedis client (e.g. Python) on a Redis
    cluster is addressable with the same name through another
    client (e.g. C++).

Finally, there is a similar function in each client,
``Client.set_script_from_file()``, that will read a
script from file and set it in the database.

Retrieving
----------

A script can be retrieved from the database using the
``Client.get_script()`` function.  While the return
type varies between languages, only the script name
that was used with ``Client.set_script()`` is needed
to reference the script in the database.  Note that
in a Redis cluster configuration, only one copy of the
script is returned to the user.

.. note::

    ``Client.get_script()`` will allocate memory to retrieve
    the script from the database, and this memory will not
    be freed until the Client object is destroyed.

Executing
---------

A script can be executed using the ``Client.run_script()`` function.
The only required inputs to execute a script are the script name,
the name of the function in the script to execute, a list of input
tensor names, and a list of output tensor names.
If using a Redis cluster configuration, a copy of the script
referenced by the provided name will be chosen based on data locality.
It is worth noting that the names of input and output tensors will be
altered with ensemble member identifications if the SmartSim
ensemble compatibility features are used.

.. note::
    DataSet tensors can be used as ``run_script()`` input tensors,
    but the name provided to ``run_script()`` must be prefixed with
    the ``DataSet`` name in the pattern ``{dataset_name}.tensor_name``.

Support on Systems with Multiple GPUs
-------------------------------------

SmartRedis has special support for scripts on systems with multiple GPUs.
On these systems, the script can be set via the ``Client.set_script_multigpu()``
function, which differs from the ``Client.set_script()`` function only in that
(1) there is no need to specify a device (GPU is implicit) and (2) the caller
must supply the index of the first GPU to use with the script and the total
number of GPUs on the system's nodes to use with the script. The function will
then create separate copies of the script for each GPU by appending ``.GPU:n``
to the supplied name, where ``n`` is a number from ``first_gpu`` to
``first_gpu + num_gpus - 1``, inclusive.

Executing scripts on systems with multiple GPUs may be done via the
``Client.run_script_multigpu()`` function. This method parallels
``Client.run_script()`` except that it requires three additional parameters:
the first GPU to use for execution, the number of GPUs to use for execution,
and an offset for the currently executing thread or image. The script execution
is then dispatched to the copy of the script on the GPU corresponding to
``first_gpu`` plus the offset modulo ``num_gpus``.  The image offset may
be an MPI rank, or a thread ID, or any other indexing scheme.

Finally, scripts stored for multiple GPUs may be deleted via the
``Client.delete_script_multigpu()`` function. This method parallels
``Client.delete_script()`` except that it requires two additional parameters:
the first GPU and the number of GPUs that the model was stored with. This
function will delete all the extra copies of the model that were stored
via ``Client.set_script_multigpu()``.

.. note::

    In order for a script to be executed via ``Client.run_script_multigpu()``,
    or deleted via ``Client.delete_script_multigpu()``,
    it must have been set via ``Client.set_script_multigpu()``. The
    ``first_gpu`` and ``num_gpus`` parameters must be constant across both calls.
