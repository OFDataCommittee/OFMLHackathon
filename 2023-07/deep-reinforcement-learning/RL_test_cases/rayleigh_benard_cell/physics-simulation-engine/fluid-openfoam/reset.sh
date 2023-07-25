#!/bin/bash
cd "${0%/*}" || exit
set -e
. ${WM_PROJECT_DIR:?}/bin/tools/CleanFunctions

#------------------------------------------------------------------------------

(
    cleanAdiosOutput
    cleanAuxiliary
    cleanDynamicCode
    cleanOptimisation
    rm -rf ./preCICE-output/
    rm -rf ./preCICE-*/
    rm -rf  ./postProcessing/*/*[1-9]*
    rm -f log.*
    touch fluid-openfoam.foam
)
