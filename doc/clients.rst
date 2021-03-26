
****
SILC
****

Client Overview
===============

The SmartSim Infrastructure Library Clients are essentially
Redis clients with additional functionality. In particular,
the SILC clients allow a user to send and receive n-dimensional
``Tensors`` with metadata, a crucial data format for data
analysis and machine learning.

Each client in SILC has distributed support for Redis clusters
and can work with both Redis and KeyDB.

Furthermore, the client implementations in SILC are all
RedisAI compatable meaning that they can directly set
and run Machine Learning and Deep Learning models stored
within a Redis database.


.. list-table:: Supported Languages
   :widths: 25 25 25
   :header-rows: 1
   :align: center

   * - Language
     - Version/Standard
     - Status
   * - Python
     - 3.7+
     - In Development
   * - C++
     - C++11
     - Stable
   * - C
     - C99
     - Stable
   * - Fortran
     - Fortran 2003 +
     - In Development

Data Formats
============

There are two data formats within SILC: Tensor and Dataset.

Tensors
-------

The fundamental data format of the SILC clients is an n-dimensional tensor. This
data structure, however, is opaque to the user in that n-dimensional arrays in the
host language (C, C++, Fortran, Python) are transformed into the tensor format at runtime.
Another way to say this is that tensors are not a user-space data structure, but rather
work directly with the data format already present in the program the SILC clients are
embedded into.

For example, in Python, the SILC client works directly with NumPy arrays.
For C and C++, both nested and contiguous memory arrays are supported.


.. _silc_dataset:

Dataset
-------

In many scientific applications, groups of n-dimensional tensors need to be grouped
as they have some association. For example, an CFD modeler might want to send grid or
coordinate information along with the value tensors. Similarly, a scientist might want
to label the dimensions of a tensor. For this specific reason, we create a user-facing
object called the Dataset.

Datasets are groups of n-dimensional tensors that can also be supplied with metadata.
The Dataset enables users to construct and send batches of tensors with metadata to the
Orchestrator and receive them from the Orchestrator with only a single key. Users need
not know where the tensors and metadata within the Dataset object are stored once they
have been sent to the Orchestrator. Users only need to know the key of the dataset was
stored in order to retrieve a Dataset.


Supported Data Types
--------------------

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


Inference API
=============

The inference API refers to the scripting API with TorchScript and the Model API
which supports ONNX, Pytorch, Tensorflow and Tensorflow-Lite placement and execution
on CPU and GPU.

TorchScript API
---------------

The ability to perform online data processing is essential to enabling online inference.
Most machine learning algorithms will not perform unless the input data has been processed.
Often this processing is as simple as a normalization. However, if there was an intermediate
data processing step between a Model and the Orchestrator (where the ML model is hosted for
inference), this would result in significant latency penalties. For this reason, the
Orchestrator is capable of executing TorchScript programs inside of the Redis database.

The SILC clients are capable of putting, getting, and executing TorchScript programs
remotely. The scripts are JIT-traced python programs that can operate on any data stored
within the Orchestrator. Scripts are executed and the result returned by the TorchScript
program is stored within the database. Once the result is stored, it can be used in
executions of ML models that also reside within the database. By co-locating script
and models, the process of using a ML model in real-time becomes much more performant.

Torchscript programs can be run on GPU or CPU. The API for script placement and execution
can be found in the API documentation.


Model API
---------

In addition to supporting the transfer of n-dimensional tensors, SILC clients support the
remote execution of Pytorch , TensorFlow, TensorFlow-Lite and ONNX models that are stored
within the Orchestrator. SILC clients can be embedded into simulations with the goal of being
able to augment simulations with machine learning models co-located in an in-memory database
reachable by Unix socket on node or through TCP over the network.

SILC clients support putting, retrieving and executing ML models in the aforementioned
frameworks. When a call to ``client.set_model()`` is performed, a single copy of the
model is placed on every node of the database. The reason for this is data locality.
When performing the remote execution of a model through the SILC client, the model chosen
for execution is the model closest to the node in the database where the input data to the
model is stored. In the case where all of the requested data for a model execution is
not on the same node of the database, SILC clients will move the requested data.
When required, the data movement process is completely opaque to the user, which
makes remote model execution simpler from an implementation standpoint.

Design
======

Simulation and data analytics codes communicate with the database using
SmartSim clients written in the native language of the codebase. These
clients perform two essential tasks (both of which are opaque to the application):

 1. Serialization/deserialization of data
 2. Communication with the database

The API for these clients are designed so that implementation within
simulation and analysis codes requires minimal modification to the underlying
codebase.


.. |SmartSim Clients| image:: images/Smartsim_Client_Communication.png
  :width: 500
  :alt: Alternative text

|SmartSim Clients|
