
import numba, numpy as np
from scipy import interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
# Linear Approximator related functions
import warnings
warnings.filterwarnings('ignore')
# Spline value calculating function, given params and "x"
@numba.njit(cache = True, fastmath = True, inline = 'always')
def func_linear(x, ix, x0, y0, k):
    return (x - x0[ix]) * k[ix] + y0[ix]
    
# Compute piece-wise linear function for "x" out of sorted "x0" points
@numba.njit([f'f{ii}[:](f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:])' for ii in (4, 8)],
    cache = True, fastmath = True, inline = 'always')
def piece_wise_linear(x, x0, y0, k, dummy0, dummy1):
    xsh = x.shape
    x = x.ravel()
    ix = np.searchsorted(x0[1 : -1], x)
    y = func_linear(x, ix, x0, y0, k)
    y = y.reshape(xsh)
    return y
    
# Spline Approximator related functions
    
# Solves linear system given by Tridiagonal Matrix
# Helper for calculating cubic splines
@numba.njit(cache = True, fastmath = True, inline = 'always')
def tri_diag_solve(A, B, C, F):
    n = B.size
    assert A.ndim == B.ndim == C.ndim == F.ndim == 1 and (
        A.size == B.size == C.size == F.size == n
    ) #, (A.shape, B.shape, C.shape, F.shape)
    Bs, Fs = np.zeros_like(B), np.zeros_like(F)
    Bs[0], Fs[0] = B[0], F[0]
    for i in range(1, n):
        Bs[i] = B[i] - A[i] / Bs[i - 1] * C[i - 1]
        Fs[i] = F[i] - A[i] / Bs[i - 1] * Fs[i - 1]
    x = np.zeros_like(B)
    x[-1] = Fs[-1] / Bs[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (Fs[i] - C[i] * x[i + 1]) / Bs[i]
    return x
    
# Calculate cubic spline params
@numba.njit(cache = True, fastmath = True, inline = 'always')
def calc_spline_params(x, y):
    a = y
    h = np.diff(x)
    c = np.concatenate((np.zeros((1,), dtype = y.dtype),
        np.append(tri_diag_solve(h[:-1], (h[:-1] + h[1:]) * 2, h[1:],
        ((a[2:] - a[1:-1]) / h[1:] - (a[1:-1] - a[:-2]) / h[:-1]) * 3), 0)))
    d = np.diff(c) / (3 * h)
    b = (a[1:] - a[:-1]) / h + (2 * c[1:] + c[:-1]) / 3 * h
    return a[1:], b, c[1:], d
    
# Spline value calculating function, given params and "x"
@numba.njit(cache = True, fastmath = True, inline = 'always')
def func_spline(x, ix, x0, a, b, c, d):
    dx = x - x0[1:][ix]
    return a[ix] + (b[ix] + (c[ix] + d[ix] * dx) * dx) * dx
    
# Compute piece-wise spline function for "x" out of sorted "x0" points
@numba.njit([f'f{ii}[:](f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:])' for ii in (4, 8)],
    cache = True, fastmath = True, inline = 'always')
def piece_wise_spline(x, x0, a, b, c, d):
    xsh = x.shape
    x = x.ravel()
    ix = np.searchsorted(x0[1 : -1], x)
    y = func_spline(x, ix, x0, a, b, c, d)
    y = y.reshape(xsh)
    return y
    
# Appximates function given by (x0, y0) by piece-wise spline or linear
def approx_func(x0, y0, t = 'spline'): # t is spline/linear
    assert x0.ndim == 1 and y0.ndim == 1 and x0.size == y0.size#, (x0.shape, y0.shape)
    n = x0.size - 1
    if t == 'linear':
        k = np.diff(y0) / np.diff(x0)
        return piece_wise_linear, (x0, y0, k, np.zeros((0,), dtype = y0.dtype), np.zeros((0,), dtype = y0.dtype))
    elif t == 'spline':
        a, b, c, d = calc_spline_params(x0, y0)
        return piece_wise_spline, (x0, a, b, c, d)
    else:
        assert False, t

# Main function that computes Euclidian Equi-Distant points based on approximation function
@numba.njit(
    [f'f{ii}[:, :](f{ii}[:], f{ii}[:], f{ii}, b1, b1, f{ii}, f{ii}, f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:])' for ii in (4, 8)],
    cache = True, fastmath = True)
def _resample_inner(x, y, d, is_spline, strict, aerr, rerr, a0, a1, a2, a3, a4):
    rs, r = 0, np.zeros((1 << 10, 2), dtype = y.dtype)
    t0 = np.zeros((1,), dtype = y.dtype)
    i, x0, y0 = 0, x[0], y[0]
    #print(i, x0, y0, np.sin(x0))
    while True:
        if rs >= r.size:
            r = np.concatenate((r, np.zeros(r.shape, dtype = r.dtype))) # Grow array
        r[rs, 0] = x0
        r[rs, 1] = y0
        rs += 1
        if i + 1 >= x.size:
            break
        ie = min(i + 1 + np.searchsorted(x[i + 1:], x0 + d), x.size - 1)
        for ie in range(i + 1 if strict else ie, ie + 1):
            xl = max(x0, x[ie - 1 if strict else i])
            xr = max(x0, x[ie])
            # Do binary search to find next point
            for ii in range(1000):
                if xr - xl <= aerr:
                    break # Already very small delta X interval
                xm = (xl + xr) / 2
                t0[0] = xm
                if is_spline:
                    ym = piece_wise_spline(t0, a0, a1, a2, a3, a4)[0]
                else:
                    ym = piece_wise_linear(t0, a0, a1, a2, a3, a4)[0]
                
                # Compute Euclidian distance
                dx_, dy_ = xm - x0, ym - y0
                dm = np.sqrt(dx_ * dx_ + dy_ * dy_)

                if -rerr <= dm / d - 1. <= rerr:
                    break # We got d with enough precision
                if dm >= d:
                    xr = xm
                else:
                    xl = xm
            else:
                assert False # To many iterations
            if -rerr <= dm / d - 1. <= rerr:
                break # Next point found
        else:
            break # No next point found, we're finished
        i = np.searchsorted(x, xm) - 1
        #print('_0', i, x0, y0, np.sin(x0), dist(x0, xm, y0, ym), dist(x0, xm, np.sin(x0), np.sin(xm)))
        x0, y0 = xm, ym
        #print('_1', i, x0, y0, np.sin(x0), dm)
    return r[:rs]
    
# Resamples (x, y) points using given approximation function type
# so that euclidian distance between each resampled points equals to "d".
# If strict = True then strictly closest (by X) next point at distance "d"
# is chosen, which can imply more computations, when strict = False then
# any found point with distance "d" is taken as next.
def resample_euclid_equidist(
    x, y, d, *,
    aerr = 2 ** -21, rerr = 2 ** -9, approx = 'spline',
    return_approx = False, strict = True,
):
    assert d > 0, d
    dtype = np.dtype(y.dtype).type
    x, y, d, aerr, rerr = [dtype(e) for e in [x, y, d, aerr, rerr]]
    ixs = np.argsort(x)
    x, y = x[ixs], y[ixs]
    f, fargs = approx_func(x, y, approx)
    r = _resample_inner(x, y, d, approx == 'spline', strict, aerr, rerr, *fargs)
    return (r[:, 0], r[:, 1]) + ((), (lambda x: f(x, *fargs),))[return_approx]

def test():
    import matplotlib.pyplot as plt, numpy as np, time
    np.random.seed(0)
    # Input
    n = 50
    x = np.sort(np.random.uniform(0., 10 * np.pi, (n,)))
    y = np.sin(x) * 5 + np.sin(1 + 2.5 * x) * 3 + np.sin(2 + 0.5 * x) * 2
    # Visualize
    for isl, sl in enumerate(['spline', 'linear']):
        # Compute resampled points
        for i in range(3):
            tb = time.time()
            xa, ya, fa = resample_euclid_equidist(x, y, 2, approx = sl, return_approx = True)
            print(sl, 'try', i, 'run time', round(time.time() - tb, 4), 'sec', flush = True)
        # Compute spline/linear approx points
        fax = np.linspace(x[0], x[-1], 1000)
        fay = fa(fax)
        # Plotting
        plt.rcParams['figure.figsize'] = (7.2, 4.5)
        for ci, (cx, cy, fn) in enumerate([
            (x, y, 'original'), (fax, fay, f'approx_{sl}'), (xa, ya, 'euclid_euqidist'),
        ]):
            p, = plt.plot(cx, cy)
            p.set_label(fn)
            if ci >= 2:
                plt.scatter(cx, cy, marker = '.', color = p.get_color())
                if False:
                    # Show distances
                    def dist(x0, x1, y0, y1):
                        # Compute Euclidian distance
                        dx, dy = x1 - x0, y1 - y0
                        return np.sqrt(dx * dx + dy * dy)
                    for i in range(cx.size - 1):
                        plt.annotate(
                            round(dist(cx[i], cx[i + 1], cy[i], cy[i + 1]), 2),
                            (cx[i], cy[i]), fontsize = 'xx-small',
                        )
        plt.gca().set_aspect('equal', adjustable = 'box')
        plt.legend()
        plt.show()
        plt.clf()
def get_next_subsection(a,b,c):
    newa = []
    newb = []
    newc = []
    neg = np.sign(a[0]-a[1])
    i = 0
    while(i < len(a)-1 and np.sign(a[i] - a[i+1]) == neg):
        newa.append(a[i])
        newb.append(b[i])
        newc.append(c[i])
        i+=1
    if(neg == 1):
        newa.reverse()
        newb.reverse()
        newc.reverse()
    return newa,newb,newc,i

def get_spline(f1,plot=False):
    j = 0
    xl = list(f1['x'])
    yl = list(f1['y'])
    tl = list(f1['t'])
    newx = []
    newy = []
    newt = []
    plt.plot(xl, yl)
    while(j < len(f1['x'])-1):
        xj, yj, tj, newj = get_next_subsection(xl[j:],yl[j:],tl[j:])
    #     print(np.shape(tj),np.shape(xj), np.shape(xl[j:]),np.shape(tl[j:]))
        if(len(xj) > 2):
            g = interpolate.CubicSpline(xj, yj, bc_type='natural')
            gt = interpolate.CubicSpline(xj, tj, bc_type='natural')
            xa, ya, fa = resample_euclid_equidist(np.asarray(xj), np.asarray(yj), 2, approx = 'spline', return_approx = True)
#             x_new = np.arange(min(xj), max(xj), 2)
#             y_new = g(x_new)
            t_new = g(xa)
            newx.append(xa)
            newy.append(ya)
            newt.append(t_new)
            if(plot):
                plt.plot(xa, ya)
        j += newj
    newx = np.concatenate([x.ravel() for x in newx])
    newy = np.concatenate([x.ravel() for x in newy])
    newt = np.concatenate([x.ravel() for x in newt])
    return newx, newy, newt



f = pd.read_csv("../../ants.txt", delimiter=',')



#671, 932,....., #1171, 1198, 1349
file = open("Spline_Data_X7.csv", "w")
file2 = open("Spline_Data_Y7.csv","w")
file3 = open("Spline_Data_T7.csv",'w')
writer = csv.writer(file)
writer2 = csv.writer(file2)
writer3=csv.writer(file3)
newdat =  {}
max_int = 428!
print(len(np.unique(f['id'])))
ind = 0
u = np.unique(f['id'])
for i in tqdm(u[497:]):
    if(ind > max_int):
        break
    f1 = f[f['id']==i]
    f1.drop_duplicates(subset=['x'],inplace=True)
    x,y,t = get_spline(f1)
    writer.writerow(x)
    writer2.writerow(y)
    writer3.writerow(t)
    ind +=1
file.close()   