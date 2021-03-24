***************
Data Structures
***************

RedisAI defines three new data structures to be used in redis databases: tensor, model, and script.
In addition, SILC defines an additional data structure ``DataSet``.  In this section, the SILC
API for interacting with these data structures will be described, and when applicable,
comments on performance and best practices will be made.

In general, concepts and capabilities will be demonstrated for the Python and C++ API.
The C and Fortran function signatures closely resemble the C++ API, and as a result,
they are not discussed in detail in the interest of brevity.
For more detailed explanations of the C and Fortran API, refer to the documentation
pages for those clients.

Tensor
======

An n-dimensional tensor is used by RedisAI to store and manipulate numerical data. SILC
provides functions to put a key and tensor pair into the Redis database and retrieve
a tensor associated with a key from the database.

.. note::
    When utilizing SILC with SmartSim ensemble functionality,
    the key provided by the user may be
    manipulated before placement in or retrieval from the database in order to
    avoid key collisions between ensemble members.  Therefore, when using
    SmartSim ensemble functionality, retrieval of a tensor using the Redis
    command line interface (CLI) will require an adapting the expected
    key.

Sending
-------

In Python, the ``Client`` infers the type and dimensions of the tensor from the
NumPy array data structure, and as a result, only the key and NumPy array are needed to
place a key and tensor pair in the database.  Currently, only NumPy arrays
are supported as inputs and outputs of Python ``Client`` tensor functions.

.. code-block:: python

    # Python put_tensor() interface
    put_tensor(self, key, data)

In the compiled clients, more information is needed to inform the ``Client`` about
the tensor properties.  In the C++ API, the dimensions of the tensor are provided
via a ``std::vector<size_t>`` input parameter.  Additionally, the type associated
with the tensor data (e.g. ``double``, ``float``) is specified with a
``SILC::TensorType`` enum.  Finally, the ``Client`` must know the memory
layout of the provided tensor data in order to generate a tensor data buffer.
In C++, tensor data can either be in a contiguous memory layout or in a nested,
non-contiguous memory layout (i.e. nested pointer arrays to underlying allocated memory).
The memory layout of the tensor data to place in the database is specified
with a ``SILC::MemoryLayout`` enum input parameter.

.. code-block:: cpp

    // C++ put_tensor interface
    void put_tensor(const std::string& key,
                    void* data,
                    const std::vector<size_t>& dims,
                    const TensorType type,
                    const MemoryLayout mem_layout);

C and Fortran have similar function prototypes compared to the C++ client,
except the C client uses only C data types and the Fortran client does
not require the specification of the tensor memory layout because it is assumed
that Fortran array memory is allocated in a column-major, contiguous manner.

Retrieving
----------

The C++, C, and Fortran clients provide two methods for retrieving
tensors from the Redis database. The first method is referred to as *unpacking* a
tensor.  When a tensor is retrieved via ``unpack_tensor()``, the memory space to
store the retrieved tensor data is provided by the user.
This has the advantage of giving the user the ability to manage the scope of the
retrieved tensor allocated memory and reuse application memory.

The C++ function signature for ``unpack_tensor()`` is shown below.  In the case
of ``unpack_tensor()`` the parameters ``dims``, ``type``, and ``mem_layout``
are used to specify the characteristics of the user-provided memory space.
The type and dimensions are compared to the tensor that is retrieved
from the database, and if the dimensions and type do not match, an error
will be thrown.  Otherwise, the memory space pointed to by the ``data``
pointer will be filled consistent with the specified memory layout.

.. code-block:: cpp

    // C++ unpack_tensor() interface
    void unpack_tensor(const std::string& key,
                       void* data,
                       const std::vector<size_t>& dims,
                       const TensorType type,
                       const MemoryLayout mem_layout);

The other option for retrieving a tensor with the
C++, C, and Fortran clients is ``get_tensor()``.  With ``get_tensor()``,
it is assumed that the user does not know the dimensions or type of the tensor,
and as a result, the ``Client`` allocates and manages memory necessary for the
retrieved tensor data.  The C++ function signature for ``get_tensor()`` is shown
below.  Note that a pointer to the newly allocated data, tensor dimensions, and
tensor type are returned to the user via modifying referenced variables that the
user declares before the ``get_tensor()`` call.  This is done to provide a similar
experience across the C++, C, and Fortran clients.

.. code-block:: cpp

    // C++ get_tensor interface
    void get_tensor(const std::string& key,
                    void*& data,
                    std::vector<size_t>& dims,
                    TensorType& type,
                    const MemoryLayout mem_layout);

.. note::
    Memory allocated by ``Client`` during a ``get_tensor()`` call will be valid
    and not freed until the ``Client`` object is destroyed.  Therefore, if the
    type and dimensions of the tensor are known, it is recommended that
    ``unpack_tensor()`` is used in memory-constrained situations.

The Python client currently only offers a ``get_tensor()`` option for
retrieving tensors.  In this methodology, a NumPy array is returned
to the user, and the only required input to the function is the
name of the tensor to retrieve because all the type and dimensions
information are embedded in the NumPy array object.
The Python interface for ``get_tensor()`` is shown below.

.. code-block:: python

    # Python get_tensor() interface
    get_tensor(self, key):

Note that all of the client ``get_tensor()`` functions will internally
modify the provided tensor name if the client is being used with
SmartSim ensemble capabilities.

Dataset
=======

In many situations, a ``Client``  might be tasked with sending a group of tensors and
metadata that are closely related and naturally grouped into a collection for
future retrieval.  The ``DataSet`` object stages these items so that they
can be more efficiently placed in the redis database and can later be retrieved with a
single key.

Listed below are the supported tensor and metadata types.  In the following sections,
building, sending, and retrieving a ``DataSet`` will be described.

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

When building a ``DataSet`` to be stored in the database, a user can add
any combination of tensors and metadata.  To add a tensor to the ``DataSet``,
the user simply uses the ``DataSet.add_tensor()`` function defined in
each language.  The ``DataSet.add_tensor()`` parameters are the same
as ``Client.put_tensor()``, and as a result, details of the function
signatures will not be reiterated here.

.. note::
    ``DataSet.add_tensor()`` copies the tensor data provided by the user to
    eliminate errors from user-provided data being cleared or deallocated.
    This additional memory will be freed when the DataSet
    object is destroyed.

Metadata can be added to the ``DataSet`` with the ``DataSet.add_meta_scalar()``
and ``DataSet.add_meta_string()`` functions.  As the aforementioned function
names suggest, there are separate functions to add metadata that is a scalar
(e.g. double) and a string. For both functions, the first function input
is the name of the metadata field.  This field name is an internal ``DataSet``
identifier for the metadata value(s) that is used for future retrieval,
and because it is an internal identifier, the user does not have to worry
about any key conflicts in the database (i.e. multiple ``DataSet`` can have
the same metadata field names).  To clarify these and future descriptions,
the C++ interface for adding metadata is shown below:

.. code-block:: cpp

    // C++ add_meta_scalar() interface
    void add_meta_scalar(const std::string& name,
                         const void* data,
                         const MetaDataType type);

    // C++ add_meta_string() interface
    void add_meta_string(const std::string& name,
                         const std::string& data);


When adding a scalar or string metadata value, the value is copied
by the ``DataSet``, and as a result, the user does not need to ensure
that the metadata values provided are still in memory after they have
been added.  Additionally, multiple metadata values can be added to a
single field, and the default behavior is to append the value to the
existing field.  In this way, the ``DataSet`` metadata supports
one-dimensional arrays, but the entries in the array must be added
iteratively by the user.  Also, note that in the above C++ example,
the metadata scalar type must be specified with a ``SILC::MetaDataType``
enum value, and similar requirements exist for C and Fortran ``DataSet``
implementations.

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

The functions for retrieving tensors from ``DataSet`` are identical
to the functions provided by ``Client``, and the same return
values and memory management paradigm is followed.  As a result,
please refer to the previous section for details on tensor retrieve
function calls.

There are two functions for retrieving metadata: ``get_meta_scalars()``
and ``get_meta_strings()``.  As the names suggest, the first function
is used for retrieving numerical metadata values, and the second is
for retrieving metadata string values.  The metadata retrieval function
prototypes vary across the clients based on programming language constraints,
and as a result, please refer to the ``DataSet`` API documentation
for a description of input parameters and memory management.  It is
important to note, however, that all functions require the name of the
metadata field to be retrieved, and this name is the same name that
was used when constructing the metadata field with ``add_meta_scalar()``
and ``add_meta_string()`` functions.

Model
=====

Like tensors, the RedisAI model data structure is exposed to users
through ``Client`` function calls to place a model in the database,
retrieve a model from the database, and run a model.  Note that
RedisAI supports PyTorch, TensorFlow, TensorFlow Lite, and ONNX backends,
and specifying the backend to be used is done through the ``Client``
function calls.

Sending
-------

A model is placed in the database through the ``Client.set_model()``
function.  While data types may differ, the function parameters
are uniform across all SILC clients, and as an example, the C++
``set_model()`` function is shown below.

.. code-block:: cpp

    # C++ set_model interface
    void set_model(const std::string& key,
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
the reader is encouraged to read the SILC client code
documentation or the RedisAI documentation for a description
of each parameter.

.. note::
    With a Redis cluster configuration, ``Client.set_model()`` will distribute
    a copy of the model to each database node in the
    cluster.  As a result, the model that has been
    placed in the cluster with ``Client.set_model()``
    will not be addressable directly with the Redis CLI because
    of key manipulation that is required to accomplish
    this distribution.  Despite the internal key
    manipulation, models in a Redis cluster that have been
    set through the SILC ``Client`` can be accessed
    and run through the SILC ``Client`` API
    using the key provided to ``set_model()``.  The user
    does not need any knowledge of the cluster model distribution
    to perform RedisAI model actions.  Moreover,
    a model set by one SILC client (e.g. Python) on a Redis
    cluster is addressable with the same key through another
    client (e.g. C++).

Finally, there is a similar function in each client,
``Client.set_model_from_file()``, that will read a
model from file and set it in the database.

Retrieving
----------

A model can be retrieved from the database using the
```Client.get_model()``` function.  While the return
type varies between languages, only the model key
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
The only required inputs to execute a model are the model key,
a list of input tensor names, and a list of output tensor names.
If using a Redis cluster configuration, a copy of the model
referenced by the provided key will be chosen based on data locality.
It is worth noting that the names of input and output tensor will be
altered with ensemble member identifications if this SmartSim
ensemble compatibility features are used.

DataSet tensors can be used as ``run_model()`` input tensors,
but the key provided to ``run_model()`` must be prefixed with
the DataSet name in the pattern ``{dataset_name}.tensor_name``.

Script
======

Data processing is an essential step in most machine
learning workflows.  For this reason, RedisAI provides
the ability to evaluate PyTorch programs using the hardware
co-located with the Redis database (either CPU or GPU).
The SILC ``Client`` provides functions for users to
place a script in the database, retrieve a script from the
database, and run a script.

Sending
-------

A script is placed in the database through the ``Client.set_script()``
function.  While data types may differ, the function parameters
are uniform across all SILC clients, and as an example, the C++
``set_script()`` function is shown below.  The function signature
is quite simple for placing a script in the database, only
a name for the script, hardware for execution, and the script text
need to be provided by the user.

.. code-block:: cpp

    void set_script(const std::string& key,
                    const std::string& device,
                    const std::string_view& script);

.. note::
    With a Redis cluster configuration, ``Client.set_script()`` will distribute
    a copy of the script to each database node in the
    cluster.  As a result, the script that has been
    placed in the cluster with ``Client.set_script()``
    will not be addressable directly with the Redis CLI because
    of key manipulation that is required to accomplish
    this distribution.  Despite the internal key
    manipulation, scripts in a Redis cluster that have been
    set through the SILC ``Client`` can be accessed
    and run through the SILC ``Client`` API
    using the key provided to ``set_script()``.  The user
    does not need any knowledge of the cluster script distribution
    to perform RedisAI script actions.  Moreover,
    a script set by one SILC client (e.g. Python) on a Redis
    cluster is addressable with the same key through another
    client (e.g. C++).

Finally, there is a similar function in each client,
``Client.set_script_from_file()``, that will read a
script from file and set it in the database.

Retrieving
----------

A script can be retrieved from the database using the
```Client.get_script()``` function.  While the return
type varies between languages, only the script key
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
The only required inputs to execute a script are the script key,
the name of the function in the script to executive, a list of input
tensor names, and a list of output tensor names.
If using a Redis cluster configuration, a copy of the script
referenced by the provided key will be chosen based on data locality.
It is worth noting that the names of input and output tensor will be
altered with ensemble member identifications if this SmartSim
ensemble compatibility features are used.

DataSet tensors can be used as ``run_script()`` input tensors,
but the key provided to ``run_script()`` must be prefixed with
the DataSet name in the pattern ``{dataset_name}.tensor_name``.