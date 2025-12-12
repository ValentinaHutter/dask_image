import numpy as np
import xarray as xr

def evi(cube_10, cube_20=None):
    # (2.5*(B08 - B04)/((B08 + 6*B04-7.5 * B02) + 1))
    b02 = cube_10.blue.astype(int)*0.0001-0.1
    b04 = cube_10.red.astype(int)*0.0001-0.1
    b08 = cube_10.nir.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = np.where((cube_10.red == 0).values, False, True)
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)

    return np.where(NaN, (2.5*(b08 - b04)/((b08 + 6*b04-7.5*b02) + 1)), np.nan).astype("float16")


def nbr(cube_10, cube_20):
    # ((B08 - B12)/(B08 + B12))
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b12 = cube_20.swir22.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = np.where((cube_10.red == 0).values, False, True)
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)

    return np.where(NaN, ((b08 - b12)/(b08 + b12)), np.nan).astype("float16")


def ndmi(cube_10, cube_20):
    # ((B08 - B11)/(B08 + B11))
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b11 = cube_20.swir16.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = np.where((cube_10.red == 0).values, False, True)
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)

    return np.where(NaN, ((b08 - b11)/(b08 + b11)), np.nan).astype("float16")


def nmdi(cube_10, cube_20):
    # (B08 – (B11 – B12))/(B08 + (B11 – B12))
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b11 = cube_20.swir16.astype(int)*0.0001-0.1
    b12 = cube_20.swir22.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = np.where((cube_10.red == 0).values, False, True)
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)

    return np.where(NaN, ((b08 - (b11 - b12))/(b08 + (b11 - b12))), np.nan).astype("float16")


def ndwi(cube_10, cube_20=None):
    # ((B03 - B08)/(B08 + B03))
    b03 = cube_10.green.astype(int)*0.0001-0.1
    b08 = cube_10.nir.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = np.where((cube_10.red == 0).values, False, True)
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)

    return np.where(NaN, ((b03 - b08)/(b08 + b03)), np.nan).astype("float16")


def ndii(cube_10, cube_20):
    # ((B08 - B11)/(B08 + B11))
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b11 = cube_20.swir16.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = np.where((cube_10.red == 0).values, False, True)
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)

    return np.where(NaN, ((b08 - b11)/(b08 + b11)), np.nan).astype("float16")


def exg(cube_10, cube_20=None):
    # ((2 * B03) - (B04 + B02))
    b02 = cube_10.blue.astype(int)*0.0001-0.1
    b03 = cube_10.green.astype(int)*0.0001-0.1
    b04 = cube_10.red.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = np.where((cube_10.red == 0).values, False, True)
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)

    return np.where(NaN, ((2 * b03) - (b04 + b02)), np.nan).astype("float16")
    

    
def tcari_osavi(cube_10, cube_20):
    # (3*((B05 – B04) – 0.2 * (B05 – B03) * (B05/4))/(1.16 * B08 – (B04/B08) + B04 + 0.16)

    b03 = cube_10.green.astype(int)*0.0001-0.1
    b04 = cube_10.red.astype(int)*0.0001-0.1
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b05 = cube_20.rededge1.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = np.where((cube_10.red == 0).values, False, True)
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)

    return np.where(NaN, (3*((b05-b04) - 0.2*(b05-b03)*(b05/4))/(1.16*b08 - (b04/b08) + b04 + 0.16)), np.nan).astype("float16")
    
    
def ndvi(cube_10, cube_20=None):
    # ((B08 - B04)/(B08 + B04))
    b04 = cube_10.red.astype(int)*0.0001-0.1
    b08 = cube_10.nir.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = np.where((cube_10.red == 0).values, False, True)
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)

    return np.where(NaN, ((b08 - b04)/(b08 + b04)), np.nan).astype("float16")
    
    
def albedo(cube_10, cube_20):
    # B02 * 0.1836 + B03 * 0.1759 + B04 * 0.1456 + B05 * 0.1347 + B06 * 0.1233 + B07 * 0.1134 + B08 * 0.1001 + B11 * 0.0231 + B12 * 0.0003
    b02 = cube_10.blue.astype(int)*0.0001-0.1
    b03 = cube_10.green.astype(int)*0.0001-0.1
    b04 = cube_10.red.astype(int)*0.0001-0.1
    b08 = cube_10.nir.astype(int)*0.0001-0.1
    b05 = cube_20.rededge1.astype(int)*0.0001-0.1
    b06 = cube_20.rededge2.astype(int)*0.0001-0.1
    b07 = cube_20.rededge3.astype(int)*0.0001-0.1
    b11 = cube_20.swir16.astype(int)*0.0001-0.1
    b12 = cube_20.swir22.astype(int)*0.0001-0.1

    if isinstance(cube_10.red, xr.DataArray):
        NaN = np.where((cube_10.red == 0).values, False, True)
    elif isinstance(cube_10.red, np.ndarray):
        NaN = np.where((cube_10.red == 0), False, True)

    return np.where(NaN, (b02*0.1836 + b03*0.1759 + b04*0.1456 + b05*0.1347 + b06*0.1233 + b07*0.1134 + b08*0.1001 + b11*0.0231 + b12*0.0003), np.nan).astype("float16")
    
    
def get_bands(cube_10, cube_20):
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