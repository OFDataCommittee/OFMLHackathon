
********
Overview
********

The SmartSim Infrastructure Library Clients (SILC) are a set of
Redis clients that support RedisAI capabilities with additional
features for high performance computing (HPC) applications.
Key features of RedisAI that are supported by SILC include:

-   A tensor data type in Redis
-   TensorFlow, TensorFlow Lite, Torch,
    and ONNXRuntime backends for model evaluations
-   TorchScript storage and evaluation

In additional to the RedisAI capabilities above,
SILC includes the following features developed for
large, distributed architectures:

-   Redis cluster support for distributed data storage
    and model serving
-   Distributed model and script placement for parallel
    evaluation to maximize hardware utilization and throughput
-   A ``DataSet`` storage format to aggregate multiple tensors
    and metadata into a single Redis cluster hash slot
    to prevent data scatter on Redis clusters.  This is useful
    when clients produce tensors and metadata that are
    referenced or utilized together.
-   Compatibility with SmartSim ensemble capabilities to
    prevent key collisions with
    tensors, ``DataSet``, models, and scripts when
    clients are part of an ensemble of applications

SILC provides clients in Python, C++, C, and Fortran.
These clients have been written to provide a
consistent API experience across languages, within
the constraints of language capabilities.  The table
below summarizes the language standards required to build
the clients.

.. list-table:: Supported Languages
   :widths: 25 25 25
   :header-rows: 1
   :align: center

   * - Language
     - Version/Standard
     - Status
   * - Python
     - 3.7
     - Stable
   * - C++
     - C++17
     - Stable
   * - C
     - C99
     - Stable
   * - Fortran
     - Fortran 2018
     - Stable
