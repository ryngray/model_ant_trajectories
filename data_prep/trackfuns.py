# Module for manipulating trajectories
# Use %run ./trackfuns.ipynb in your import section to use functions of this script

# Contents:
# addMetrics: adds s, theta, alpha to track DataFrame
# alphaToTrack: creates track (x, y) from alpha

import numpy as np
import copy
import Splines
def get_edr(trackDat, s=2):
    edr = Splines.get_spline(trackDat,s)
    return edr

def addMetrics(trackDat):
    # Calculates step length, heading angle, and turn angle for each track in the data.
    # In: pd.DataFrame of all tracks w/ [x, y, t, id]
    # Out: all tracks w/ [x,y,t,id,s,theta,alpha]
    # s & theta calculated as currPoint - prevPoint; alpha: angle around the current point (thus, NaNs)
    nanArr = np.empty(len(trackDat))
    nanArr[:] = np.NaN
    s = copy.deepcopy(nanArr)
    theta = copy.deepcopy(nanArr)
    alpha = copy.deepcopy(nanArr)
    ids = np.unique(trackDat.id)
    for id in ids:
        idx = trackDat.id == id
        dist = np.diff(trackDat[idx], axis=0)
        s[idx] = np.concatenate(([np.nan], np.hypot(dist[:,0], dist[:,1])),axis=0) # Distances between points
        thetaPre = np.arctan2(np.diff(trackDat.x[idx]), np.diff(trackDat.y[idx])) # Heading angle (N = 0, E = +)
        alphaPre = np.diff(thetaPre) # Turn angle (left = -, right = +)
        alpha[idx] = np.concatenate(([np.nan], np.degrees((alphaPre + np.pi) % (2*np.pi) - np.pi), [np.nan]), axis=0)
        theta[idx] = np.concatenate(([np.nan], np.degrees(thetaPre)),axis=0)
    trackDat['s'] = s
    trackDat['theta'] = theta
    trackDat['alpha'] = alpha
    return trackDat


def alphaToTrack(allAngles):
    # Input: turn angles
    # Output: x, y of the trajectories, starting @ 0, with steplength = 1
    theta = np.cumsum(allAngles)
    dx = [0] + np.cos(theta)
    dy = [0] + np.sin(theta)
    xCoords = np.cumsum(dx)
    yCoords = np.cumsum(dy)
    return xCoords, yCoords

