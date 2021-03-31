
********
Overview
********

SmartRedis is a collection of Redis clients that support
RedisAI capabilities and include additional
features for high performance computing (HPC) applications.
Key features of RedisAI that are supported by SmartRedis include:

-   A tensor data type in Redis
-   TensorFlow, TensorFlow Lite, Torch,
    and ONNXRuntime backends for model evaluations
-   TorchScript storage and evaluation

In addition to the RedisAI capabilities above,
SmartRedis includes the following features developed for
large, distributed HPC architectures:

-   Redis cluster support for RedisAI data types (tensors,
    models, and scripts)
-   Distributed model and script placement for parallel
    evaluation that maximizes hardware utilization and throughput
-   A ``DataSet`` storage format to aggregate multiple tensors
    and metadata into a single Redis cluster hash slot
    to prevent data scatter on Redis clusters and
    maintain contextual relationships between tensors.
    This is useful when clients produce tensors and
    metadata that are referenced or utilized together.
-   Compatibility with SmartSim ensemble capabilities to
    prevent key collisions with
    tensors, ``DataSet``, models, and scripts when
    clients are part of an ensemble of applications

SmartRedis provides clients in Python, C++, C, and Fortran.
These clients have been written to provide a
consistent API experience across languages, within
the constraints of language capabilities.  The table
below summarizes the language standards for each client.

.. list-table:: Supported Languages
   :widths: 35 35
   :header-rows: 1
   :align: center

   * - Language
     - Version/Standard
   * - Python
     - 3.7
   * - C++
     - C++17
   * - C
     - C99
   * - Fortran
     - Fortran 2018
