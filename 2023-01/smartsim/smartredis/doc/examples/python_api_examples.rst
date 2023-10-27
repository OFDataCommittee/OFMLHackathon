.. _python_client_examples:

******
Python
******

In this section, examples are presented using the SmartRedis Python
API to interact with the RedisAI tensor, model, and script
data types.  Additionally, an example of utilizing the
SmartRedis ``DataSet`` API is also provided.

.. note::

    The Python API examples are written to connect to a
    database at ``127.0.0.1:6379``.  When running this example,
    ensure that the address and port of your Redis instance are used.

.. note::

    The Python API examples are written
    to connect to a clustered backend database.  Update the
    ``Client`` constructor call to connect to a non-clustered backend database.

Tensors
=======
The Python client has the ability to send and receive tensors from
the backend database.  The tensors are stored in the backend database
as RedisAI data structures.  Additionally, Python client API
functions involving tensor data are compatible with Numpy arrays
and do not require any other data types.

.. literalinclude:: ../../examples/serial/python/example_put_get_tensor.py
  :language: python
  :linenos:


Datasets
========

The Python ``Client`` API stores and retrieve datasets from the backend database. The Python
``DataSet`` API can store and retrieve tensors and metadata from an in-memory ``DataSet`` object.
To reiterate, the actual interaction with the backend database, 
where a snapshot of the ``DataSet`` object is sent, is handled by the Client API.
For further information about datasets, please refer to the :ref:`Dataset
section of the Data Structures documentation page <data_structures_dataset>`.

The code below shows how to store and retrieve tensors which belong to a ``DataSet``.

.. literalinclude:: ../../examples/serial/python/example_put_get_dataset.py
  :language: python
  :linenos:

Models
======

The SmartRedis clients allow users to set and use a PyTorch, ONNX, TensorFlow,
or TensorFlow Lite model in the database. Models can be sent to the database directly
from memory or from a file. The code below illustrates how a
jit-traced PyTorch model can be used with the Python client library.

.. literalinclude:: ../../examples/serial/python/example_model_torch.py
  :language: python
  :linenos:

Models can also be set from a file, as in the code below.

.. literalinclude:: ../../examples/serial/python/example_model_file_torch.py
  :language: python
  :linenos:

Scripts
=======

Scripts are a way to store python-executable code in the database. The Python
client can send scripts to the dataset from a file, or directly from memory.

As an example, the code below illustrates how a function can be defined and sent
to the database on the fly, without storing it in an intermediate file.

.. literalinclude:: ../../examples/serial/python/example_script.py
  :language: python
  :linenos:

The code below shows how to set a script from a file.  Running the
script set from file uses the same API calls as the example shown
above.

.. literalinclude:: ../../examples/serial/python/example_script_file.py
  :language: python
  :linenos:

The content of the script file has to be written
in Python. For the example above, the file ``data_processing_script.txt``
looks like this:

.. literalinclude:: ../../examples/serial/python/data_processing_script.txt
  :language: python
  :linenos: