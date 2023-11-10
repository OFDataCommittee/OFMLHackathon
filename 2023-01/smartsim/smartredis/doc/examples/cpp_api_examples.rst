.. _cpp_client_examples:

***
C++
***

In this section, examples are presented using the SmartRedis C++
API to interact with the RedisAI tensor, model, and script
data types.  Additionally, an example of utilizing the
SmartRedis ``DataSet`` API is also provided.

.. note::

    The C++ API examples rely on the ``SSDB`` environment
    variable being set to the address and port of the Redis database.

.. note::

    The C++ API examples are written
    to connect to a clustered backend database.  Update the
    ``Client`` constructor call to connect to a non-clustered backend database.

Tensors
=======

The following example shows how to send a receive a tensor using the
SmartRedis C++ client API.

.. literalinclude:: ../../examples/serial/cpp/smartredis_put_get_3D.cpp
  :linenos:
  :language: C++

DataSets
========

The C++ ``Client`` API stores and retrieve datasets from the backend database. The C++
``DataSet`` API can store and retrieve tensors and metadata from an in-memory ``DataSet`` object.
To reiterate, the actual interaction with the backend database,
where a snapshot of the ``DataSet`` object is sent, is handled by the Client API.
For further information about datasets, please refer to the :ref:`Dataset
section of the Data Structures documentation page <data-structures-dataset>`.

The code below shows how to store and retrieve tensors and metadata
which belong to a ``DataSet``.

.. literalinclude:: ../../examples/serial/cpp/smartredis_dataset.cpp
  :linenos:
  :language: C++

.. _SR CPP Models:

Models
======

The following example shows how to store, and use a DL model
in the database with the C++ Client.  The model is stored a file
in the ``../../../common/mnist_data/`` path relative to the
compiled executable.  Note that this example also sets and
executes a preprocessing script.

.. literalinclude:: ../../examples/serial/cpp/smartredis_model.cpp
  :linenos:
  :language: C++

.. _SR CPP Scripts:

Scripts
=======

The example in :ref:`SR CPP Models` shows how to store, and use a PyTorch script
in the database with the C++ Client.  The script is stored a file
in the ``../../../common/mnist_data/`` path relative to the
compiled executable.  Note that this example also sets and
executes a PyTorch model.

.. _SR CPP Parallel MPI:

Parallel (MPI) execution
========================

In this example, the example shown in :ref:`SR CPP Models` and
:ref:`SR CPP Scripts` is adapted to run in parallel using MPI.
This example has the same functionality, however,
it shows how keys can be prefixed to prevent key
collisions across MPI ranks.  Note that only one
model and script are set, which is shared across
all ranks.

For completeness, the pre-processing script
source code is also shown.

**C++ program**

.. literalinclude:: ../../examples/parallel/cpp/smartredis_mnist.cpp
  :linenos:
  :language: C++

**Python Pre-Processing**

.. literalinclude:: ../../examples/common/mnist_data/data_processing_script.txt
  :linenos:
  :language: Python
  :lines: 15-20