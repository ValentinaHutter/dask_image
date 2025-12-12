import numpy as np
import xarray as xr


def lai(cube_10, cube_20):
    # ((2 * B03) - (B04 + B02))
    b02 = cube_10.blue.astype(int)*0.0001-0.1
    b03 = cube_10.green.astype(int)*0.0001-0.1
    b04 = cube_10.red.astype(int)*0.0001-0.1
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b05 = cube_20.rededge1.astype(int)*0.0001-0.1
    b06 = cube_20.rededge2.astype(int)*0.0001-0.1
    b07 = cube_20.rededge3.astype(int)*0.0001-0.1
    b8a = cube_20.nir08.astype(int)*0.0001-0.1
    b11 = cube_20.swir16.astype(int)*0.0001-0.1
    b12 = cube_20.swir22.astype(int)*0.0001-0.1

    return b02, b03, b04, b08, b05, b06, b07, b8a, b11, b12