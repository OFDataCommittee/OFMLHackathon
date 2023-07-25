# OpenFOAM - ML Hackathon SmartSim Folder

## Compilation

Compile the foamSmartSimMapFields application with 

```
    ?> ./Allwmake
```

at this point it does not yet use smartredis, that is the next step. 

## Usage 

So now the foamSmartSimMapFields application includes client.h and links to the smartredis database library. 

The OpenFOAM application/solver/functionObject/fvOption/meshMotionSolver/.... is always going to be a client of the SmartRedis database. To open the connection to the database the OpenFOAM client needs to include client.h from smartredis and link to the library. To do this we need to inform the wmake build system on where the include directory is of smartsim and where the dynamically linked library resides. We do this in Make/options in OpenFOAM:

https://github.com/OFDataCommittee/OFMLHackathon/blob/main/2023-07/smartsim/foamSmartSimMapFields/Make/options

using environmental variables FOAM_SMARTREDIS_* that are set in configure_smartredis.sh 

Note that your smartredis folder may lie in another place than in $HOME in which you need to adapt configure_smartredis.sh before building the application. Before the client opens a connection in the smartredis database, the application can be compiled and executed as any other OpenFOAM application. As soon as the code is added to open a connection, a database must be created and running. If the OpenFOAM client does not create the database (e.g. if a Python client does it), then the application will hang. As soon as we start working with the dbase, we need to switch to a Jupyter Notebook, that will start/stop dbase and coordinate data exchange.

At this point, without SmartSim python script, the application only links to smartredis and includes "client.h" and can be run as

```
    ?> foamSmartSimMapFields -inputCase pitzDailyCoarse -outputCase pitzDaily -field p
```

These arguments will need to be passed to the SmartSim Model in the SmartSim Jupyter Notebook once the client opens the connection to the SmartRedis Database. 
