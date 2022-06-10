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

Setting Redis Database Location
===============================

The C++, C, and Fortran clients retrieve
the Redis database location from the
``SSDB`` environment variable that is set
by the user.  If the SmartSim infrastructure
library is being used, then the infrastructure
library will set the value of ``SSDB`` for the user.


The ``SSDB`` environment variable should have the format
of ``address:port``.  For a cluster, the addresses
and ports should be separated by a "," character.
Below is an example of setting ``SSDB`` for a Redis cluster
at three different addresses using port ``6379``:

.. code-block:: bash

    export SSDB="10.128.0.153:6379,10.128.0.154:6379,10.128.0.155:6379"

The Python client also relies on ``SSDB`` to determine database
location.  However, the Python ``Client`` constructor also allows
for the database location to be set as an input parameter.

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

    void use_model_ensemble_prefix(bool use_prefix);


.. note::

    The function ``Client.use_tensor_ensemble_prefix()`` controls
    object prefixing for objects stored with ``Client.put_tensor()``
    and all ``DataSet`` components added via ``DataSet.add_tensor()``,
    ``DataSet.add_meta_scalar()``, and ``DataSet.add_meta_string()``.

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
