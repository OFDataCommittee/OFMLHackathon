********
C++ APIs
********

The following page provides a comprehensive overview of the SmartRedis C++ 
Client and Dataset APIs. 
Further explanation and details of each are presented below.

Client API
==========

The Client API is purpose-built for interaction with the backend database, 
which extends the capabilities of the Redis in-memory data store. 
It's important to note that the SmartRedis Client API is the exclusive 
means for altering, transmitting, and receiving data within the backend 
database. More specifically, the Client API is responsible for both 
creating and modifying data structures, which encompass :ref:`Models <data-structures-model>`, 
:ref:`Scripts <data-structures-script>`, and :ref:`Tensors <data-structures-tensor:>`.  
It also handles the transmission and reception of 
the aforementioned data structures in addition to :ref:`Dataset <data-structures-dataset>` 
data structure. Creating and modifying the ``DataSet`` object 
is confined to local operation by the DataSet API.

.. doxygenclass:: SmartRedis::Client
   :project: cpp_client
   :members:
   :undoc-members:


Dataset API
===========

The C++ DataSet API enables a user to manage a group of tensors 
and associated metadata within a datastructure called a ``DataSet`` object. 
The DataSet API operates independently of the database and solely 
maintains the dataset object in-memory. The actual interaction with the Redis database, 
where a snapshot of the DataSet object is sent, is handled by the Client API. For more 
information on the ``DataSet`` object, click :ref:`here <data-structures-dataset>`.

.. doxygenclass:: SmartRedis::DataSet
   :project: cpp_client
   :members:
   :undoc-members:

