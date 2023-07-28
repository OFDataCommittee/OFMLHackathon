#!/opt/paraview-5.10/bin/pvpython
import os
import sys

import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 10
#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'OpenFOAMReader'
casefoam = OpenFOAMReader(registrationName='case.foam', FileName=f'case.foam')
casefoam.MeshRegions = ['internalMesh']
casefoam.CellArrays = ['U', 'epsilon', 'k', 'nut', 'p']
casefoam.Decomposepolyhedra = 0

LoadPalette(paletteName='WhiteBackground')

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# get the material library
materialLibrary1 = GetMaterialLibrary()

# get display properties
casefoamDisplay = GetDisplayProperties(casefoam, view=renderView1)

# get color transfer function/color map for 'p'
pLUT = GetColorTransferFunction('p')

# get opacity transfer function/opacity map for 'p'
pPWF = GetOpacityTransferFunction('p')

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

#change interaction mode for render view
renderView1.InteractionMode = '2D'

# reset view to fit data bounds
renderView1.ResetCamera(-0.020599400624632835, 0.2899929881095886, -0.023620499297976494, 0.02539060078561306, 0.0, 0.0010000000474974513, False)

# reset view to fit data bounds
renderView1.ResetCamera(-0.020599400624632835, 0.2899929881095886, -0.023620499297976494, 0.02539060078561306, 0.0, 0.0010000000474974513, False)

# reset view to fit data
renderView1.ResetCamera(False)

# reset view to fit data
renderView1.ResetCamera(False)

# reset view to fit data
renderView1.ResetCamera(False)

# reset view to fit data
renderView1.ResetCamera(False)

# Properties modified on casefoam
casefoam.CellArrays = []

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on casefoam
casefoam.CellArrays = ['U']

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(casefoamDisplay, ('CELLS', 'U', 'Magnitude'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(pLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
casefoamDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
casefoamDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'U'
uLUT = GetColorTransferFunction('U')

# get opacity transfer function/opacity map for 'U'
uPWF = GetOpacityTransferFunction('U')

# reset view to fit data
renderView1.ResetCamera(False)

# get color legend/bar for uLUT in view renderView1
uLUTColorBar = GetScalarBar(uLUT, renderView1)

# change scalar bar placement
uLUTColorBar.Orientation = 'Horizontal'
uLUTColorBar.WindowLocation = 'Any Location'
uLUTColorBar.Position = [0.35741403026134816, 0.2300630517023959]
uLUTColorBar.ScalarBarLength = 0.33000000000000035

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(1454, 793)

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.1346967937424779, 0.0008850507438182831, 0.6079459234934522]
renderView1.CameraFocalPoint = [0.1346967937424779, 0.0008850507438182831, 0.0005000000237487257]
renderView1.CameraParallelScale = 0.15721857386384755

# save screenshot
SaveScreenshot(f'{sys.argv[1]}.png', renderView1, ImageResolution=[2908, 1586], 
    # PNG options
    CompressionLevel='3')

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1454, 793)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.1346967937424779, 0.0008850507438182831, 0.6079459234934522]
renderView1.CameraFocalPoint = [0.1346967937424779, 0.0008850507438182831, 0.0005000000237487257]
renderView1.CameraParallelScale = 0.15721857386384755

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
