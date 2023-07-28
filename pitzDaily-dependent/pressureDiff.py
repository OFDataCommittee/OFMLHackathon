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

# create a new 'OpenFOAMReader'
casefoam = OpenFOAMReader(registrationName='case.foam', FileName=f'{sys.argv[1]}/case.foam')
casefoam.MeshRegions = ['patch/inlet']
casefoam.CellArrays = ['total(p)']
casefoam.Decomposepolyhedra = 0

UpdatePipeline(int(sys.argv[2]), casefoam)

casefoam.MeshRegions = ['patch/inlet']

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=casefoam)
calculator1.Function = 'avg("total(p)")'
arr = servermanager.Fetch(calculator1)
pIn = arr.GetBlock(0).GetBlock(0).GetPointData().GetArray("Result").GetValue(0)

casefoam.MeshRegions = ['patch/outlet']
arr = servermanager.Fetch(calculator1)
pOut = arr.GetBlock(0).GetBlock(0).GetPointData().GetArray("Result").GetValue(0)

print(f"{abs(pIn-pOut)}")

# Alternative way with  probes; less robust

## create a new 'Probe Location'
#probeLocation1 = ProbeLocation(registrationName='ProbeLocation1', Input=casefoam,
#    ProbeType='Fixed Radius Point Source')
## init the 'Fixed Radius Point Source' selected for 'ProbeType'
#probeLocation1.ProbeType.Center = [-0.020599400624632835, 0.011522900313138962, 0.001]
#
#prb1 = servermanager.Fetch(probeLocation1)
#p1 = prb1.GetPointData().GetArray("p").GetValue(0)
#
## create a new 'Probe Location'
#probeLocation2 = ProbeLocation(registrationName='ProbeLocation2', Input=casefoam,
#    ProbeType='Fixed Radius Point Source')
#
## init the 'Fixed Radius Point Source' selected for 'ProbeType'
#probeLocation2.ProbeType.Center = [0.2899929881095886, 0.0045229000970721245, 0.001]
#
#prb2 = servermanager.Fetch(probeLocation2)
#p2 = prb2.GetPointData().GetArray("p").GetValue(0)
#print(f"{abs(p1-p2)}")
