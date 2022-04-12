# Module for manipulating trajectories

# Contents:
# 1. addMetrics: adds s, theta, alpha to track DataFrame
# 2. alphaToTrack: creates track (x, y) from alpha
# 3. get_edr: EquiDistantly Resamples track

import numpy as np
import copy
import math
import pandas as pd
import circ_stats as cs

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
    trackDat.loc[:,'s'] = s
    trackDat.loc[:,'theta'] = theta
    trackDat.loc[:,'alpha'] = alpha
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



def get_edr(inDat, s, M=10):
    # https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values, answered by Ubuntu
    # Find lots of points on the piecewise linear curve defined by x and y
    # Inputs: inDat: table of tracks
    #         s: Resample step length [mm] (should be ~median or mode step length)
    #         M: #of interpolated points between each input point

    # find lots of points on the piecewise linear curve defined by x and y
    M = 100*len(inDat.x)
    t = np.linspace(0, len(inDat.x), M).T
    x = np.interp(t, np.arange(len(inDat.x)), inDat.x).T
    y = np.interp(t, np.arange(len(inDat.y)), inDat.y).T
    tol = 2
    i, idx = 0, [0]
    while i < len(x):
        total_dist = 0
        for j in range(i+1, len(x)):
            #total_dist += math.sqrt((x[j]-x[j-1])**2 + (y[j]-y[j-1])**2) # Distance traveled
            total_dist = math.sqrt((x[j]-x[i])**2 + (y[j]-y[i])**2) # Displacement
            if total_dist >= tol:
                idx.append(j)
                break
        i = j+1

    edrTrack = pd.DataFrame(np.vstack((x[idx],y[idx],t[idx],np.multiply(np.ones(len(idx)),inDat.id[0]))).T, columns=['x','y','t','id'])
    return edrTrack


def turnac(inDat,tauMax):
    # Circular autocorrelation of turn angles.
    # Outputs [id, tau (=timelag), rho (=correlation coefficient [-1,1])]
    ids = np.unique(inDat.id)
    rhoVec = np.nan
    for id in ids:
        track = inDat.alpha[inDat.id==id].values
        a = track[1:-2] # omitting NaNs
        for tau in range(tauMax): # rho in 2nd col
            rhoVec = np.append(rhoVec, cs.corrcoef(a[:-(tau+2)], a[tau:-2]))
    rhoVec = rhoVec[~np.isnan(rhoVec)]
    idVec = np.repeat(ids,tauMax)
    tauVec = np.tile(np.add(range(tauMax),1),len(ids))
    rhoMat = np.vstack((idVec,tauVec,rhoVec)).T
    rhoTau = pd.DataFrame(rhoMat,columns=['id','tau','rho'])
    return rhoTau

def sum_stats(dat):
    ids = np.unique(dat.id)

    turnChg = np.empty(len(ids))
    msd = np.empty(len(ids))
    nrCross = np.empty(len(ids))
    turnAcRho = np.empty(len(ids))

    tra = 0 # Current track, for row assignment
    for id in ids:
        tr = dat[dat.id==id].reset_index() # tr for track

        # Number of turn direction changes
        asign = np.sign(tr.alpha)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        signchange[0] = 0
        turnChg[tra] = sum(signchange)

        # MSD
        T = 10 # Nr of sample points
        tInds = np.linspace(0,len(tr)-1,T).astype(int) # Indices of sample points
        dists = []
        j = 1
        for t in tInds:
            for tau in tInds[j:]:
                dists = np.append(dists,(tr.x[tau] - tr.x[t])**2 + (tr.y[tau] - tr.y[t])**2)
            j+=1
        msd[tra] = np.mean(dists)


        # Nr of path crosses
        def ccw(A,B,C):
            return ((C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0]))

        # Return true if line segments AB and CD intersect
        def intersect(A,B,C,D):
            return (ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D))

        a = np.asarray(tr)
        crossCount = 0
        for i in range(len(a)-1):
            for j in range(i+2, len(a)-1):
                A = [a[i][0], a[i][1]]
                B = [a[i+1][0], a[i+1][1]]
                C = [a[j][0], a[j][1]]
                D = [a[j+1][0], a[j+1][1]]
                if(intersect(A,B,C,D)):
                    crossCount += 1
        nrCross[tra] = crossCount

        # Turn autocorrelation
        tauMax = 50
        a = tr.alpha[1:-1].values # omitting NaNs
        for tau in range(tauMax): # rho in 2nd col
            rhoVec = cs.corrcoef(a[:-(tau+2)], a[tau:-2])
        minRho = np.min(rhoVec)
        minTau = np.argmin(rhoVec)
        turnAcRho[tra] = minRho

        tra+=1

    mu = dat.groupby(['id']).mean()
    SD = dat.groupby(['id']).std()

    sumStats = pd.DataFrame(ids,columns=['id'])
    sumStats['sMu'] = mu.s.values
    sumStats['sSD'] = SD.s.values
    sumStats['alphaMu'] = mu.alpha.values
    sumStats['alphaSD'] = SD.alpha.values
    sumStats['turnChg'] = turnChg
    sumStats['MSD'] = msd
    sumStats['nrCross'] = nrCross
    sumStats['turnAcRho'] = turnAcRho
    return sumStats