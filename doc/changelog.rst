Changelog
=========

0.3.0
-----

Release on Febuary 11, 2022

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

Release on August, 5, 2021

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
