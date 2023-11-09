********************
Runtime Requirements
********************

At runtime, the clients need to know where to look
for the Redis database.  Additionally,  if
SmartSim ensemble functionality is being used,
information is needed to prevent key collisions
and to retrieve the correct information from the
database.  In the following sections,
these requirements will be described.

Setting Redis Database Location and Type
========================================

The C++, C, and Fortran clients retrieve
the Redis database location from the
``SSDB`` environment variable that is set
by the user.  If the SmartSim infrastructure
library is being used, then the infrastructure
library will set the value of ``SSDB`` for the user.


The ``SSDB`` environment variable should have the format
of ``address:port``, and may optionally be prefixed with
a protocol, such as ``tcp://``.  For a cluster, the addresses
and ports should be separated by a "," character. If no
protocol is specified, the default will be taken as TCP.
Unix domain sockets are supported via the protocol prefex
``unix://``; however, Unix domain sockets are not supported
in clusters.

Below is an example of setting ``SSDB`` for a Redis cluster
at three different addresses, each using port ``6379``:

.. code-block:: bash

    export SSDB="10.128.0.153:6379,10.128.0.154:6379,10.128.0.155:6379"


There are two types of Redis databases that can be used by the
SmartRedis library. A ``Clustered`` database, such as the one in
the previous example, is replicated across multiple shards.
By way of comparison, a ``Standalone`` database only has a single
shard that services all traffic; this is the form used when a
colocated database or a standard deployment with a non-sharded
database is requested.

The ``SR_DB_TYPE`` environment variable informs the SmartRedis
library which form is in use. Below is an example of setting
``SR_DB_TYPE`` for a Redis cluster:

.. code-block:: bash

    export SR_DB_TYPE="Clustered"

Logging Environment Variables
=============================

SmartRedis will log events to a file on behalf of a client. There
are two main environment variables that affect logging, ``SR_LOG_FILE``
and ``SR_LOG_LEVEL``.

``SR_LOG_FILE`` is the specifier for the location of the file that
receives logging information. Each entry in the file will be prefixed
with a timestamp and the identifier of the client that invoked the logging
message. The path may be relative or absolute, though a relative path risks
creation of multiple log files if the executables that instantiate SmartRedis
clients are run from different directories. If this variable is not set,
logging information will be sent to standard console output (STDOUT in C and
C++; output_unit in Fortran).

``SR_LOG_LEVEL`` relates to the verbosity of information that wil be logged.
It may be one of three levels: ``QUIET`` disables logging altogether.
``INFO`` provides informational logging, such as exception events that
transpire within the SmartRedis library and creation or destruction of a
client object.  ``DEBUG`` provides more verbose logging, including information
on the activities of the SmartRedis thread pool and API function entry and exit.
Debug level logging will also log the absence of an expected environment variable,
though this can happen only if the variables to set up logging are in place. If
this parameter is not set, a default logging level of ``INFO`` will be adopted.

The runtime impact of log levels NONE or INFO should be minimal on
client performance; however, setting the log level to DEBUG may cause some
degradation.

Ensemble Environment Variables
==============================

The clients work with SmartSim ensemble functionality through
environment variables.  There are two environment variables
that are used for ensembles, ``SSKEYIN`` and ``SSKEYOUT``.

``SSKEYOUT`` defines the prefix that is attached to
tensors, datasets, models, and scripts sent from the client
to the database.  This prefixing prevents key collisions for
objects sent from the client to the database.  ``SSKEYOUT``
should be set to a single string value.  If using the
SmartSim infrastructure library to launch the ensemble,
``SSKEYOUT`` will be set by SmartSim.  An example
value of ``SSKEYOUT`` is:

.. code-block:: bash

    export SSKEYOUT="model_1"


``SSKEYIN`` defines prefixes that can be attached to
tensors, datasets, models, and scripts when retrieving
data from the database.  ``SSKEYIN`` can have multiple,
comma separated values, however, only one of the values
can be used at a time.  ``SSKEYIN`` allows a client
in an application to retrieve data from clients
that were part of ensemble when placing data in the
database.  An example value of ``SSKEYIN`` is:

.. code-block:: bash

    export SSKEYIN="model_2,model_3,model_4"

In the case of multiple ``SSKEYIN`` values, the ``Client``
API provides a function ``Client.set_data_source()``
to select which ``SSKEYIN`` value is used.  The
default is to use the first value of ``SSKEYIN``,
and any value specified using ``Client.set_data_source()``
must be present in ``SSKEYIN`` when the ``Client``
is created.


The ``Client`` API provides functions to activate or
deactivate the use of ``SSKEYIN`` and ``SSKEYOUT``.
These functions are split by the data type
that prefixes are applied to in order to give the
user fine control of prefixing in advanced applications.
The default is to use prefixes on tensors and datasets
if ``SSKEYIN`` and ``SSKEYOUT`` are present.  The default
is not to use prefixes on scripts and models.
The functions for changing this default behavior are:

.. code-block:: cpp

    void use_tensor_ensemble_prefix(bool use_prefix);

    void use_dataset_ensemble_prefix(bool use_prefix);

    void use_model_ensemble_prefix(bool use_prefix);


.. note::

    The function ``Client.use_tensor_ensemble_prefix()`` controls
    object prefixing for objects stored with ``Client.put_tensor()``.

.. note::

    The function ``Client.use_dataset_ensemble_prefix()`` controls
    object prefixing for``DataSet`` components added via
    ``DataSet.add_tensor()``, ``DataSet.add_meta_scalar()``, and
    ``DataSet.add_meta_string()``.

.. note::

    The function ``Client.use_model_ensemble_prefix()`` controls
    object prefixing for model and script data.

Model Execution Environment Variable
====================================

The ``SR_MODEL_TIMEOUT`` environment variable defines a timeout
on the length of time SmartRedis will wait for a model to
execute. The value for this variable is measured in milliseconds,
and the default value is one minute.

Connection and Command Execution Environment Variables
======================================================

SmartRedis allows for client connection and command execution
behavior to be adjusted via environment variables.

During client initialization, the environment variables ``SR_CONN_INTERVAL``
and ``SR_CONN_TIMEOUT`` are used by SmartRedis to determine
the frequency of connection attempts and the cumulative amount of time
before a timeout error is thrown, respectively.  The user can set
these environment variables to adjust client connection behavior.
``SR_CONN_INTERVAL`` should be specified in milliseconds and
``SR_CONN_TIMEOUT`` should be specified in seconds.

The environment variables ``SR_CMD_INTERVAL`` and ``SR_CMD_TIMEOUT``
are used are used by SmartRedis to determine
the frequency of command execution attempts and the
cumulative amount of time before a timeout error is thrown, respectively.
The user can set these environment variables to adjust command execution behavior.
``SR_CMD_INTERVAL`` should be specified in milliseconds and
``SR_CMD_TIMEOUT`` should be specified in seconds.  Note that ``SR_CMD_INTERVAL``
and ``SR_CMD_TIMEOUT`` are read during client initialization and not
before each command execution.

The environment variable ``SR_THREAD_COUNT`` is used by SmartRedis to determine
the number of threads to initialize when building a worker pool for parallel task
execution. The default value is four. If the variable is set to zero, SmartRedis
will use a default number of threads equal to one per hardware context in the
processor on which the library is running (more specifically, SmartRedis will
use the result of a call to std::thread::hardware_concurrency() as the number
of threads to create). This default will generally give good
performance; however, if the SmartRedis library is sharing the processor hardware
with other software, it may be useful to specify a smaller number of threads for
some workloads.
