#!/bin/bash
cd "${0%/*}" || exit
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions
#------------------------------------------------------------------------------
## parallel run
# mpirun -np $(getNumberOfProcessors)  --bind-to none buoyantBoussinesqPimpleFoam -parallel > log.buoyantBoussinesqPimpleFoam 2>&1 &

## single run
runApplication $(getApplication)
