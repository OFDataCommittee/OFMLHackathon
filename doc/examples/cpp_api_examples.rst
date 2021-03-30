.. _cpp_client_examples:

***
C++
***

In this section, examples are presented using the SILC C++
API to interact with the RedisAI tensor, model, and script
data types.  Additionally, an example of utilizing the
SILC ``DataSet`` API is also provided.

.. note::

    The C++ API examples rely on the ``SSDB`` environment
    variable being set to the address and port of the Redis database.

.. note::

    The C++ API examples are written
    to connect to a non-cluster Redis database.  Update the
    ``Client`` constructor call to connect to a Redis cluster.

Tensors
=======

The following example shows how to send a receive a tensor using the
SILC C++ client API.

.. literalinclude:: ../../examples/serial/cpp/silc_put_get_3D.cpp
  :linenos:
  :language: C++

DataSets
========

The C++ client can store and retrieve tensors and metadata in datasets.
For further information about datasets, please refer to the :ref:`Dataset
section of the Data Structures documentation page <data_structures_dataset>`.

The code below shows how to store and retrieve tensors and metadata
which belong to a ``DataSet``.

.. literalinclude:: ../../examples/serial/cpp/silc_dataset.cpp
  :linenos:
  :language: C++

Models
======

The following example shows how to store, and use a DL model
in the database with the C++ Client.  The model is stored a file
in the ``../../../common/mnist_data/`` path relative to the
compiled executable.  Note that this example also sets and
executes a preprocessing script.

.. literalinclude:: ../../examples/serial/cpp/silc_model.cpp
  :linenos:
  :language: C++

Scripts
=======

The example in `Models`_ shows how to store, and use a PyTorch script
in the database with the C++ Client.  The script is stored a file
in the ``../../../common/mnist_data/`` path relative to the
compiled executable.  Note that this example also sets and
executes a PyTorch model.