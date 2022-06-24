Changelog
=========

0.3.1
-----

Released on June 24, 2022

Description

Version 0.3.1 adds new functionality in the form of DataSet aggregation lists for pipelined retrieval of data, convenient support for multiple GPUs, and the ability to delete scripts and models from the backend database. It also introduces multithreaded execution for certain tasks that span multiple shards of a clustered database, and it incorporates a variety of internal improvements that will enhance the library going forward.

Detailed Notes

- Implemented DataSet aggregation lists in all client languages, for pipelined retrieval of data across clustered and non-clustered backend databases. (PR258_) (PR257_) (PR256_) (PR248_) New commands are:

  - append_to_list()
  - delete_list()
  - copy_list()
  - rename_list()
  - get_list_length()
  - poll_list_length()
  - poll_list_length_gte()
  - poll_list_length_lte()
  - get_datasets_from_list()
  - get_dataset_list_range()
  - use_list_ensemble_prefix()

- Implemented multithreaded execution for parallel dataset list retrieval on clustered databases. The number of threads devoted for this purpose is controlled by the new environment variable SR_THERAD_COUNT. The value defaults to 4, but may be any positive integer or special value zero, which will cause the SmartRedis runtime to allocate one thread for each available hardware context. (PR251_) (PR246_)

- Augmented support for GPUs by implementing multi-GPU convenience functions for all client languages. (PR254_) (PR250_) (PR244_) New commands are:

  - set_model_from_file_multigpu()
  - set_model_multigpu()
  - set_script_from_file_multigpu()
  - set_script_multigpu()
  - run_model_multigpu()
  - run_script_multigpu()
  - delete_model_multigpu()
  - delete_script_multigpu()

- Added API calls for all clients to delete models and scripts from the backend database. (PR240_) New commands are:

  - delete_script()
  - delete_model()

- Updated the use of backend RedisAI API calls to discontinue use of deprecated methods for model selection (AI.MODELSET) and execution (AI.MODELRUN) in favor of current methods AI.MODELSTORE and AI.MODELEXECUTE, respectively. (PR234_)

- SmartRedis will no longer call the C runtime method srand() to ensure that it does not interfere with random number generation in client code. It now uses a separate instance of the C++ random number generator. (PR233_)

- Updated the way that the Fortran enum_kind type defined in the fortran_c_interop module is defined in order to better comply with Fortran standard and not interfere with GCC 6.3.0. (PR231_)

- Corrected the spelling of the word "command" in a few error message strings. (PR221_)

- SmartRedis now requires a CMake version 3.13 or later in order to utilize the add_link_options CMake command. (PR217_)

- Updated and improved the documentation of the SmartRedis library. In particular, a new SmartRedis Integration Guide provides an introduction to using the SmartRedis library and integrating it with existing software. (PR261_) (PR260_) (PR259_) (SSPR214_)

- Added clustered Redis testing to automated GitHub check-in testing. (PR239_)

- Updated the SmartRedis internal API for building commands for the backend database. (PR223_) This change should not be visible to clients.

- The SmartRedis example code is now validated through the automated GitHub checkin process. This will help ensure that the examples do not fall out of date. (PR220_)

- Added missing copyright statements to CMakeLists.txt and the SmartRedis examples. (PR219_)

- Updated the C++ test coverage to ensure that all test files are properly executed when running "make test". (PR218_)

- Fixed an internal naming conflict between a local variable and a class member variable in the DataSet class. (PR215_)  This should not be visible to clients.

- Updated the internal documentation of methods in SmartRedis C++ classes with the override keyword to improve compliance with the latest C++ standards. (PR214_) This change should not be visible to clients.

- Renamed variables internally to more cleanly differentiate between names that are given to clients for tensors, models, scripts, datasets, etc., and the keys that are used when storing them in the backend database. (PR213_) This change should not be visible to clients.

 .. _SSPR214: https://github.com/CrayLabs/SmartSim/pull/214
 .. _PR261: https://github.com/CrayLabs/SmartRedis/pull/261
 .. _PR260: https://github.com/CrayLabs/SmartRedis/pull/260
 .. _PR259: https://github.com/CrayLabs/SmartRedis/pull/259
 .. _PR258: https://github.com/CrayLabs/SmartRedis/pull/258
 .. _PR257: https://github.com/CrayLabs/SmartRedis/pull/257
 .. _PR256: https://github.com/CrayLabs/SmartRedis/pull/256
 .. _PR254: https://github.com/CrayLabs/SmartRedis/pull/254
 .. _PR251: https://github.com/CrayLabs/SmartRedis/pull/251
 .. _PR250: https://github.com/CrayLabs/SmartRedis/pull/250
 .. _PR248: https://github.com/CrayLabs/SmartRedis/pull/248
 .. _PR246: https://github.com/CrayLabs/SmartRedis/pull/246
 .. _PR244: https://github.com/CrayLabs/SmartRedis/pull/244
 .. _PR240: https://github.com/CrayLabs/SmartRedis/pull/240
 .. _PR239: https://github.com/CrayLabs/SmartRedis/pull/239
 .. _PR234: https://github.com/CrayLabs/SmartRedis/pull/234
 .. _PR233: https://github.com/CrayLabs/SmartRedis/pull/233
 .. _PR231: https://github.com/CrayLabs/SmartRedis/pull/231
 .. _PR223: https://github.com/CrayLabs/SmartRedis/pull/223
 .. _PR221: https://github.com/CrayLabs/SmartRedis/pull/221
 .. _PR220: https://github.com/CrayLabs/SmartRedis/pull/220
 .. _PR219: https://github.com/CrayLabs/SmartRedis/pull/219
 .. _PR218: https://github.com/CrayLabs/SmartRedis/pull/218
 .. _PR217: https://github.com/CrayLabs/SmartRedis/pull/217
 .. _PR215: https://github.com/CrayLabs/SmartRedis/pull/215
 .. _PR214: https://github.com/CrayLabs/SmartRedis/pull/214
 .. _PR213: https://github.com/CrayLabs/SmartRedis/pull/213

0.3.0
-----

Released on Febuary 11, 2022

Description

- Improve error handling across all SmartRedis clients (PR159_) (PR191_) (PR199_) (PR205_) (PR206_)

  - Includes changes to C and Fortran function prototypes that are not backwards compatible
  - Includes changes to error class names and enum type names that are not backwards compatible

- Add ``poll_dataset`` functionality to all SmartRedis clients (PR184_)

  - Due to other breaking changes made in this release, applications using methods other than ``poll_dataset`` to check for the existence of a dataset should now use ``poll_dataset``

- Add environment variables to control client connection and command timeout behavior (PR194_)
- Add AI.INFO command to retrieve statistics on scripts and models via Python and C++ clients (PR197_)
- Create a Dockerfile for SmartRedis (PR180_)
- Update ``redis-plus-plus`` version to 1.3.2 (PR162_)
- Internal client performance and API improvements (PR138_) (PR141_) (PR163_) (PR203_)
- Expose Redis ``FLUSHDB``, ``CONFIG GET``, ``CONFIG SET``, and ``SAVE`` commands to the Python client (PR139_) (PR160_)
- Extend inverse CRC16 prefixing to all hash slots (PR161_)
- Improve backend dataset representation to enable performance optimization (PR195_)
- Simplify SmartRedis build proccess (PR189_)
- Fix zero-length array transfer in Fortran ``convert_char_array_to_c`` (PR170_)
- Add continuous integration for all SmartRedis tests (PR165_) (PR173_) (PR177_)
- Update SmartRedis docstrings (PR200_) (PR207_)
- Update SmartRedis documentation and examples (PR202_) (PR208_) (PR210_)

.. _PR138: https://github.com/CrayLabs/SmartRedis/pull/138
.. _PR139: https://github.com/CrayLabs/SmartRedis/pull/139
.. _PR141: https://github.com/CrayLabs/SmartRedis/pull/141
.. _PR159: https://github.com/CrayLabs/SmartRedis/pull/159
.. _PR160: https://github.com/CrayLabs/SmartRedis/pull/160
.. _PR161: https://github.com/CrayLabs/SmartRedis/pull/161
.. _PR162: https://github.com/CrayLabs/SmartRedis/pull/162
.. _PR163: https://github.com/CrayLabs/SmartRedis/pull/163
.. _PR165: https://github.com/CrayLabs/SmartRedis/pull/165
.. _PR170: https://github.com/CrayLabs/SmartRedis/pull/170
.. _PR173: https://github.com/CrayLabs/SmartRedis/pull/173
.. _PR177: https://github.com/CrayLabs/SmartRedis/pull/177
.. _PR180: https://github.com/CrayLabs/SmartRedis/pull/180
.. _PR183: https://github.com/CrayLabs/SmartRedis/pull/183
.. _PR184: https://github.com/CrayLabs/SmartRedis/pull/184
.. _PR189: https://github.com/CrayLabs/SmartRedis/pull/189
.. _PR191: https://github.com/CrayLabs/SmartRedis/pull/191
.. _PR194: https://github.com/CrayLabs/SmartRedis/pull/194
.. _PR195: https://github.com/CrayLabs/SmartRedis/pull/195
.. _PR197: https://github.com/CrayLabs/SmartRedis/pull/197
.. _PR198: https://github.com/CrayLabs/SmartRedis/pull/198
.. _PR199: https://github.com/CrayLabs/SmartRedis/pull/199
.. _PR200: https://github.com/CrayLabs/SmartRedis/pull/200
.. _PR202: https://github.com/CrayLabs/SmartRedis/pull/202
.. _PR203: https://github.com/CrayLabs/SmartRedis/pull/203
.. _PR205: https://github.com/CrayLabs/SmartRedis/pull/205
.. _PR206: https://github.com/CrayLabs/SmartRedis/pull/206
.. _PR207: https://github.com/CrayLabs/SmartRedis/pull/207
.. _PR208: https://github.com/CrayLabs/SmartRedis/pull/208
.. _PR210: https://github.com/CrayLabs/SmartRedis/pull/210

0.2.0
-----

Released on August, 5, 2021

Description

- Improved tensor memory management in the Python client (PR70_)
- Improved metadata serialization and removed protobuf dependency (PR61_)
- Added unit testing infrastructure for the C++ client (PR96_)
- Improve command execution fault handling (PR65_) (PR97_) (PR105_)
- Bug fixes (PR52_) (PR72_) (PR76_) (PR84_)
- Added copy, rename, and delete tensor and DataSet commands in the Python client (PR66_)
- Upgrade to RedisAI 1.2.3 (PR101_)
- Fortran and C interface improvements (PR93_) (PR94_) (PR95_) (PR99_)
- Add Redis INFO command execution to the Python client (PR83_)
- Add Redis CLUSTER INFO command execution to the Python client (PR105_)

.. _PR52: https://github.com/CrayLabs/SmartRedis/pull/52
.. _PR61: https://github.com/CrayLabs/SmartRedis/pull/61
.. _PR65: https://github.com/CrayLabs/SmartRedis/pull/65
.. _PR66: https://github.com/CrayLabs/SmartRedis/pull/66
.. _PR70: https://github.com/CrayLabs/SmartRedis/pull/70
.. _PR72: https://github.com/CrayLabs/SmartRedis/pull/72
.. _PR76: https://github.com/CrayLabs/SmartRedis/pull/76
.. _PR83: https://github.com/CrayLabs/SmartRedis/pull/83
.. _PR84: https://github.com/CrayLabs/SmartRedis/pull/84
.. _PR93: https://github.com/CrayLabs/SmartRedis/pull/93
.. _PR94: https://github.com/CrayLabs/SmartRedis/pull/94
.. _PR95: https://github.com/CrayLabs/SmartRedis/pull/95
.. _PR96: https://github.com/CrayLabs/SmartRedis/pull/96
.. _PR97: https://github.com/CrayLabs/SmartRedis/pull/97
.. _PR99: https://github.com/CrayLabs/SmartRedis/pull/99
.. _PR101: https://github.com/CrayLabs/SmartRedis/pull/101
.. _PR105: https://github.com/CrayLabs/SmartRedis/pull/105

0.1.1
-----

Released on May 5, 2021

Description

- Compiled client library build and install update to remove environment variables (PR47_)
-  Pip install for Python client (PR45_)

.. _PR47: https://github.com/CrayLabs/SmartRedis/pull/47
.. _PR45: https://github.com/CrayLabs/SmartRedis/pull/45

0.1.0
-----

Released on April 1, 2021

Description

- Initial 0.1.0 release of SmartRedis
