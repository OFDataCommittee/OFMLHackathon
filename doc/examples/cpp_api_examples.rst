.. _cpp_client_examples:

**********************
Using the SILC C++ API
**********************


Models
======

The following example shows how to store, retrieve, and use a pre-processing script and a DL model in the database with the C++ Client. 
The model and the script are stored as files in the ``../../../common/mnist_data/`` path relative to the compiled executable.

.. literalinclude:: ../../examples/serial/cpp/silc_mnist.cpp
  :linenos:
  :language: C++
  :lines: 42-64
  :lineno-start: 45

The complete source code for this example is available in the `C++ example directory of the repository`_.

.. _C++ example directory of the repository: https://github.com/CrayLabs/SILC/examples/serial/cpp/