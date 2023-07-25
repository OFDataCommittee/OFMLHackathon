# OpenFOAM - ML Hackathon SmartSim Folder

## Compilation

Compile the foamSmartSimMapFields application with 

```
    ?> ./Allwmake
```

at this point it does not yet use smartredis, that is the next step. 

## Usage 

At this point, without SmartSim python script, the application only links to smartredis and includes "client.h" and can be run as

```
    ?> foamSmartSimMapFields -inputCase pitzDailyCoarse -outputCase pitzDaily -field p
```

These arguments will need to be passed to the SmartSim Model in the SmartSim Jupyter Notebook once the client opens the connection to the SmartRedis Database. 
