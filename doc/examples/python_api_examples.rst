.. _python_client_examples:

*************************
Using the SILC Python API
*************************


Models
======

The Python client allows the user to set and use a PyTorch, ONNX, TensorFlow,
or TensorFlow Lite model in the database. Models can be sent to the database directly
from memory or from a file. The code below illustrates how a jit-traced PyTorch model can be used.

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

The code below shows how to set a script from a file.

.. literalinclude:: ../../examples/serial/python/example_script_file.py
  :language: python
  :linenos:

The content of the script file has to be written
in Python. For the example above, the file ``data_processing_script.txt``
looks like this:

.. literalinclude:: ../../examples/serial/python/data_processing_script.txt
  :language: python
  :linenos:

Datasets
========

The Python client can store and retrieve tensors in datasets. For further 
information about datasets, please refer to the :ref:`Dataset section of 
the Data Structures documentation page <data_structures_dataset>`.

The code below shows how to store and retrieve tensors which belong to a dataset.

.. literalinclude:: ../../examples/serial/python/example_put_get_dataset.py
  :language: python
  :linenos: