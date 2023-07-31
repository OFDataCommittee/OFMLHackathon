#!/opt/paraview-5.10/bin/pvpython
import os
import sys

import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
import paraview.servermanager as servermanager
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'STL Reader'
mainstl = STLReader(registrationName='pitzDaily2D.stl', FileNames=[f"{sys.argv[1]}/pitzDaily2D.stl"])

# create a new 'Generate Surface Normals'
generateSurfaceNormals1 = GenerateSurfaceNormals(registrationName='GenerateSurfaceNormals1', Input=mainstl)

# create a new 'Connectivity'
connectivity1 = Connectivity(registrationName='Connectivity1', Input=generateSurfaceNormals1)
data = servermanager.Fetch(connectivity1)
rng = data.GetCellData().GetArray('RegionId').GetRange()
print(rng[0], rng[1])

i = 1
for j in range(int(rng[0]), int(rng[1])+1):
    threshold = Threshold(registrationName=f'Threshold_{j}', Input=connectivity1)
    threshold.Scalars = ['POINTS', 'RegionId']
    threshold.LowerThreshold = j
    threshold.UpperThreshold = j
    data = servermanager.Fetch(threshold)
    if data.GetNumberOfCells() > 1:
        extractSurface = ExtractSurface(registrationName=f'ExtractSurface_{j}', Input=threshold)

        patchesToLookFor = {}
        functionToLook   = {"upperWall": "coordsY>=1.5e-2", "inlet": "coordsX<=-0.02", "outlet": "coordsX>=0.28"}

        for k in functionToLook.keys():
            calculator1 = Calculator(registrationName=f'Calculator{k}{i}', Input=extractSurface)
            calculator1.ResultArrayName = f'Res{k}{i}'
            calculator1.Function = functionToLook[k]
            arr = servermanager.Fetch(calculator1)
            fld = arr.GetPointData().GetArray(f'Res{k}{i}')
            isFound = True
            for ii in range(fld.GetSize()):
                if fld.GetValue(ii) < 1.0:
                    isFound = False
            if isFound:
                patchesToLookFor[f"patch_{i}"] = k

        patchName = patchesToLookFor[f"patch_{i}"] if f"patch_{i}" in patchesToLookFor.keys() else f"patch_{i}"
        print(f"Writing {patchName}.stl ...")
        SaveData(f'{sys.argv[1]}/{patchName}.stl',
            proxy=extractSurface,
            PointDataArrays=[],
            CellDataArrays=[],
            FileType='Ascii')
        i+=1
