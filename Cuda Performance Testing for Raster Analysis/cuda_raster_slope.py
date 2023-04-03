# from osgeo import gdal
import time

import numba
from numba import cuda
import numpy as np
import math


@numba.jit(nopython=True, nogil=True)
def get_slope_numba_simple(arrx, arry, Sx, Sx2, SxSx):
    n = arrx.shape[0]
    Sy = 0
    Sx2 = 0
    Sxy = 0
    for itemx, itemy in zip(arrx, arry):
        Sxy += itemx * itemy
        Sx2 += itemx * itemx
        Sy += itemy
    return (n*Sxy - Sx * Sy) / (n*Sx2 - SxSx)


def get_slope_simple(arrx, arry, Sx, Sx2, SxSx):
    n = arrx.shape[0]
    Sy = 0
    Sxy = 0
    for itemx, itemy in zip(arrx, arry):
        Sxy += itemx * itemy
        Sy += itemy
    return (n*Sxy - Sx * Sy) / (n*Sx2 - SxSx)
def get_slope_numpy(arrx, arry, Sx, Sx2, SxSx):
    n = arrx.shape[0]
    Sy = np.sum(arry)
    Sxy = arrx @ arry
    return (n*Sxy - Sx * Sy) / (n*Sx2 - SxSx)

def get_slope_polyfit(arr_vals):
    return np.polyfit(np.arange(len(arr_vals)), arr_vals, deg=1)[0]

@cuda.jit(device=True)
def get_slope_cuda_simple(arrx, arry, Sx, Sx2, SxSx):
    n = arrx.shape[0]
    Sy = 0
    Sxy = 0
    for itemx, itemy in zip(arrx, arry):
        Sxy += itemx * itemy
        Sy += itemy
    return (n*Sxy - Sx * Sy) / (n*Sx2 - SxSx)

@cuda.jit
def get_slope_cuda(arrx, arry, Sx, Sx2, SxSx, out):
    cx, cy = cuda.grid(2)
    if cx < arry.shape[0] and cy < arry.shape[1]:
        out[cx, cy] = get_slope_cuda_simple(arrx, arry[cx, cy], Sx, Sx2, SxSx)


if __name__ == '__main__':
    for s in [100, 100, 300, 1000, 3000, 5000, 8000, 10000]:
        np.random.seed(42)
        testdata = np.random.randint(1000, 5000, (s, s, 10))#.astype(float)
        # an_array = np.zeros(testdata.shape[:-1], dtype=float)

        print("Data ready.")

        t0 = time.time()
        # threadsperblock = (2, 2)
        # threadsperblock = (16, 16)
        threadsperblock = (32, 32)
        blockspergrid_x = math.ceil(testdata.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(testdata.shape[1] / threadsperblock[1])
        # blockspergrid_x = (testdata.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        # blockspergrid_y = (testdata.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # stream = cuda.stream()
        x = np.arange(testdata.shape[-1]).astype(np.int64)
        Sx = int(sum(x))
        SxSx = Sx * Sx
        Sx2 = int(sum(np.square(x)))

        x_dev = cuda.to_device(x)
        testdata_dev = cuda.to_device(np.ascontiguousarray(testdata)) # transfer to GPU manually, so it won't be transferred back
        an_array = cuda.device_array(testdata.shape[:-1], dtype=float)  # create on GPU directly, no need for copying

        get_slope_cuda[blockspergrid, threadsperblock](x_dev, testdata_dev, Sx, Sx2, SxSx, an_array)
        # cuda.synchronize()
        out_cuda = an_array.copy_to_host()
        t1 = time.time()
        print(f"Cuda:                   took {(t1-t0):7.4f}s. (Array dim: {testdata.shape})")
        del an_array, x_dev, testdata_dev  # free GPU memory

        t0 = time.time()
        n = testdata.shape[-1]
        x = np.arange(n)
        Sx = np.sum(x)
        SxSx = Sx * Sx
        Sx2 = np.sum(np.square(x))
        Sy = np.sum(testdata, axis=-1)
        sXY = np.einsum('ijk,k', testdata, x)
        out_vec = (n * sXY - Sx * Sy) / (n * Sx2 - SxSx)
        t1 = time.time()
        print(f"Vectorized:             took {(t1-t0):7.4f}s. (Array dim: {testdata.shape}) "
              f"- are results the same? {np.allclose(out_vec, out_cuda)}")


        t0 = time.time()
        n = testdata.shape[-1]
        x = np.arange(n)
        Sx = np.sum(x)
        SxSx = Sx * Sx
        Sx2 = np.sum(np.square(x))
        out_numba = np.apply_along_axis(lambda data: get_slope_numba_simple(x, data, Sx, Sx2, SxSx), -1, testdata)
        t1 = time.time()
        print(f"Numba apply along axis: took {(t1-t0):7.4f}s. (Array dim: {testdata.shape}) "
              f"- are results the same? {np.allclose(out_numba, out_cuda)}")

        t0 = time.time()
        n = testdata.shape[-1]
        x = np.arange(n)
        Sx = np.sum(x)
        SxSx = Sx * Sx
        Sx2 = np.sum(np.square(x))
        out_npy = np.apply_along_axis(lambda data: get_slope_numpy(x, data, Sx, Sx2, SxSx), -1, testdata)
        t1 = time.time()
        print(f"np apply along axis:    took {(t1-t0):7.4f}s. (Array dim: {testdata.shape}) "
              f"- are results the same? {np.allclose(out_npy, out_cuda)}")

        t0 = time.time()
        n = testdata.shape[-1]
        x = np.arange(n)
        Sx = np.sum(x)
        SxSx = Sx * Sx
        Sx2 = np.sum(np.square(x))
        out_std = np.apply_along_axis(lambda data: get_slope_simple(x, data, Sx, Sx2, SxSx), -1, testdata)
        t1 = time.time()
        print(f"Std. apply along axis:  took {(t1-t0):7.4f}s. (Array dim: {testdata.shape}) "
              f"- are results the same? {np.allclose(out_std, out_cuda)}")


        t0 = time.time()
        out_poly = np.apply_along_axis(get_slope_polyfit, -1, testdata)
        t1 = time.time()
        print(f"Polyfit:                took {(t1-t0):7.4f}s. (Array dim: {testdata.shape}) "
              f"- are results the same? {np.allclose(out_vec, out_cuda)}")
