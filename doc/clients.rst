
*******
Clients
*******

.. list-table:: Supported Languages
   :widths: 25 25
   :header-rows: 1
   :align: center

   * - Language
     - Version/Standard
   * - Python
     - 3.7+
   * - C++
     - C++11
   * - C
     - C99
   * - Fortran
     - Fortran 2003 +


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
