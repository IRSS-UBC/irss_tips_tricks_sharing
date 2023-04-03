## Performance comparison of numba.cuda and other methods

In `cuda_raster_slope.py`, find different implementations to calculate the slope of a timeseries for each raster
cell in a 2D array. The results on a Lambda Tensorbook (early 2022) are shown here:

(*) the first call to `numba.cuda` and `numba.jit` includes the just-in-time compilation time for the kernels.
(**) these runtimes are comparable to the larger array sizes as they exclude the just-in-time compilation time.


| Array size  | Cnt  | numba.cuda | Vectorized | numba.jit | matrix mul | simple slope | np.polyfit |
|-------------|------|------------|------------|-----------|------------|--------------|------------|
| 100x100*    | 10K  | 0.3513s    | 0.0010s    | 0.2507s   | 0.0570s    | 0.0410s      | 0.3760s    |
| 100x100**   | 10K  | 0.0010s    | 0.0010s    | 0.0190s   | 0.0580s    | 0.0400s      | 0.3709s    |
| 300x300     | 90K  | 0.0020s    | 0.0010s    | 0.1640s   | 0.5180s    | 0.3617s      | 3.1471s    |
| 1000x1000   | 1M   | 0.0120s    | 0.0230s    | 1.9166s   | 5.7087s    | 4.0897s      | 35.6763s   |
| 3000x3000   | 9M   | 0.0910s    | 0.1910s    | 16.9781s  | 52.3257s   | 36.6744s     | 323.7616s  |
| 5000x5000   | 25M  | 0.2480s    | 0.5372s    | 47.7372s  | 149.9153s  | 104.8890s    | too long   |
| 8000x8000   | 64M  | 0.6110s    | 1.3757s    |           |            |              | too long   |
| 8700x8700   | 75M  | 0.7239s    | 1.6336s    |           |            |              | too long   |
| 10000x10000 | 100M | 0.9381s    | 2.1072s    | 191.8848s | 586.5449s  | 418.3852     | too long   |
| 12000x12000 | 144M | 1.6168s    | 3.1845s    | too long  |            |              |            |
| 14000x14000 | 196M | 2.0730s    | 4.3859s    | too long  |            |              |            |
| 16000x16000 | 256M | 2.6420s    | 5.6735s    | too long  |            |              |            |
| 18000x18000 | 324M | 3.2179s    | 7.0231s    | too long  |            |              |            |
| 20000x20000 | 400M | OUT OF MEM | 9.4737s    | too long  |            |              |            |
| 25000x25000 | 625M | OUT OF MEM | 16.1061    | too long  |            |              |            |
