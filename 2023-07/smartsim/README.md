# OpenFOAM - ML Hackathon SmartSim Folder

## Set the environment variables 

The command

```
    ?> source configure_smartredis.sh
```

sets the env variables, sourcing should be done from the folder that contains `configure_smartredis.sh`.  Note that your smartredis folder may lie in another place than in $HOME in which you need to adapt configure_smartredis.sh before building the application. Before the client opens a connection in the smartredis database, the application can be compiled and executed as any other OpenFOAM application. 

The foamSmartSimMapFields application includes client.h and links to the smartredis database library. 

An OpenFOAM application/solver/functionObject/fvOption/meshMotionSolver/.... is always going to be a client of the SmartRedis database. To open the connection to the database the OpenFOAM client needs to include client.h from smartredis and link to the library. To do this we need to inform the wmake build system on where the include directory is of smartsim and where the dynamically linked library resides. We do this in in OpenFOAM by modifying [Make/options](https://github.com/OFDataCommittee/OFMLHackathon/blob/main/2023-07/smartsim/foamSmartSimMapFields/Make/options) and point OpenFOAM's `wmake` build system to appropriate INCLUDE (`-I`) and LINK (`-L`) folders defined by environmental variables FOAM_SMARTREDIS_* that are set in configure_smartredis.sh 

## Compile the foamSmartSimMapFields application

Compile the foamSmartSimMapFields application with 

```
    ?> ./Allwmake
```

## Run the pitzDailyCoarse case

```
    ?> blockMesh -case pitzDailyCoarse
    ?> simpleFoam -case pitzDailyCoarse
```

## Run the Jupyter Notebook as a SmartSim Orchestrator 



As soon as the code is added to open a connection, a database must be created and running. If the OpenFOAM client does not create the database (e.g. if a Python client does it), then the application will hang. As soon as we start working with the dbase, we need to switch to a Jupyter Notebook, that will start/stop dbase and coordinate data exchange.

The OpenFOAM application `foamSmartSimMapFields` constructs a smartredis client and opens a connection to a database. For this, a database must be available. The python (Jupyter Notebook) smartredis client 'foam-smartsim-map-fields.ipynb' creates the database in this case because we govern the whole workflow primarily from python. 

The application requires following execution arguments 

```
    ?> foamSmartSimMapFields -inputCase pitzDailyCoarse -outputCase pitzDaily -field p
```

 - `field`: name of the mapped field
 - `inputCase`: OpenFOAM case where the input field `field` and input cell centers reside 
 - `outputCase`: OpenFOAM case where we will map the input field to output field associated to cell centers of the `outputCase`.  

These arguments are passed to the SmartSim Model in the SmartSim Jupyter Notebook. See the `foam-smartsim-map-fields.ipynb` for information about the next steps in the python smartredis client and smartsim orchestration, and `foamSmartSimMapFields.C` for the information about the next steps in the OpenFOAM client. 

To run the case, run the `foam-smartsim-map-fields.ipynb` Jupyter notebook.


