#!/usr/bin/bash -l
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --output=log.%x
#SBATCH --partition=queue-1
#SBATCH --constraint=c5a.24xlarge
#
source /fsx/OpenFOAM/OpenFOAM-v2212/etc/bashrc
conda activate /fsx/bias/bias_env
cd "${1}" || exit                                # Run from this directory
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions        # Tutorial run functions
#------------------------------------------------------------------------------

set -e
function getParam() {
    foamDictionary -entry "$1" -value system/geometryDict
}

nCurvePnts=$(getParam "nCurvePnts")
nCtrlPnts=$(getParam "nCtrlPnts")
nNewCtrlPnts=$((nCtrlPnts-2))
L=$(getParam "L")

counter=1
ctrlPnts="["
for i in $(seq 1 $nNewCtrlPnts); do
    ctrlPnts="$ctrlPnts [$(getParam "Ctrl${counter}X"), $(getParam "Ctrl${counter}Y"), 0],"
    counter=$((counter+1))
done
ctrlPnts="$ctrlPnts ]"

## First generate the geometry
runApplication openscad -o geometry/pitzDaily.stl -D "nCurvePnts=$nCurvePnts;nCtrlPnts=$nCtrlPnts;L=$L;newCtrlPnts=$ctrlPnts" openscad/pitzDaily.scad

## Then remove the faces whose normals are in Z
sed -r '/facet normal [-+]?0 [-+]?0 /,+6d' geometry/pitzDaily.stl > geometry/pitzDaily2D.stl

# Check the surface file
runApplication surfaceCheck geometry/pitzDaily2D.stl

# Split patches and put back together
runApplication pvpython patches.py "$PWD"/geometry
echo "Writing geometry/pitzDaily.stl"
sed "s/^solid.*/solid inlet/g" geometry/inlet.stl > geometry/pitzDaily.stl
sed "s/^solid.*/solid outlet/g" geometry/outlet.stl >> geometry/pitzDaily.stl
sed "s/^solid.*/solid upperWall/g" geometry/upperWall.stl >> geometry/pitzDaily.stl
counter=1
for ii in geometry/patch*.stl; do
    sed "s/^solid.*/solid wall_$counter/g" "$ii" >> geometry/pitzDaily.stl
    counter=$((counter+1))
done

# Generate the mesh
runApplication cartesian2DMesh

cp -rT 0orig 0
runApplication decomposePar

# Parallel run on the slurm cluster
mpirun --output-filename log $(getApplication) -parallel
runApplication reconstructPar
