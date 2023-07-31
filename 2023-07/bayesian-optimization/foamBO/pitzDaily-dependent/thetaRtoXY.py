#!/usr/bin/env python3

"""
    ./thetaRtoXY
    We get nothing, we only parse the configuration file
    We return:
        X and Y for each new control point
"""

from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import numpy as np

paramFile = ParsedParameterFile("./system/geometryDict")
L = paramFile["L"]
nNewCtrls=paramFile["nCtrlPnts"]-2
c0X=0.0
c0Y=0.0
c1X=0.29
c1Y=-0.0166
Xs = []
Ys = []

def movePoint(origin, rr, theta):
    print(f"Moving {origin} by {rr} and {theta}")
    return[origin[0] + L*rr*np.cos(theta*np.pi/180.), origin[1] + L*rr*np.sin(theta*np.pi/180.)]


for i in range(nNewCtrls//2):
    print("###")
    origin = [c0X, c0Y]
    for j in range(i+1):
        rr = paramFile[f'rr{j+1}']
        theta = paramFile[f'theta{j+1}']
        origin =  movePoint(origin, rr, theta)
    Xs.append(origin[0])
    Ys.append(origin[1])
print(Xs, Ys)
