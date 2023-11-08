# Running the application

From within `cavity` case

```
cavity> foamSmartSimDmd -inputCase . -fields '(p)'
```

Outside of the `cavity` case

```
streaming-dmd> foamSmartSimDmd -inputCase cavity -fields '(p)'
```
