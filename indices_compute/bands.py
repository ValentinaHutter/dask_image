from odc import stac as odc_stac
from pyproj import CRS


def get_crs(item):    
    if isinstance(item, list):
        item = item[0]
    if "proj:epsg" in item.properties:
        crs = CRS.from_epsg(item.properties["proj:epsg"])
    elif "proj:wkt" in item.properties:
        crs = CRS.from_wkt(item.properties["proj:wkt"])
    elif "proj:wkt2" in item.properties:
        crs = CRS.from_wkt(item.properties["proj:wkt2"])
    elif "proj:code" in item.properties:
        code = item.properties["proj:code"]
        if code.startswith("EPSG:"):
            code = code.split(":")[-1]
        crs = CRS.from_epsg(code)
    else:
        print("Could not find CRS from item properties: ", item.properties)
    return crs

def osavi(item):
    band_dim = "bands"
    crs = get_crs(item)
    if not isinstance(item, list):
        item = [item]
    data = odc_stac.load(item,
        crs=crs,
        bands=["red", "nir"],
        chunks={'time': -1, 'x': 1024, 'y': 1024},
        resolution=(10)).to_array(dim=band_dim)
    b04 = data.sel({band_dim: "red"}).astype(int)*0.0001-0.1
    b08 = data.sel({band_dim: "nir"}).astype(int)*0.0001-0.1

    osavi = (1 + 0.16)*(b08 - b04)/(b08 + b04 + 0.16)

    osavi = osavi.assign_coords(**{"index": "osavi"})
    osavi = osavi.expand_dims(dim="index")

    return osavi



def albedo(item):
    band_dim = "bands"
    crs = get_crs(item)
    if not isinstance(item, list):
        item = [item]
    data = odc_stac.load(item,
        crs=crs,
        bands=["blue", "red", "nir", "swir16", "swir22"],
        chunks={'time': -1, 'x': 1024, 'y': 1024},
        resolution=(10)).to_array(dim=band_dim)
    b02 = data.sel({band_dim: "blue"}).astype(int)*0.0001-0.1
    b04 = data.sel({band_dim: "red"}).astype(int)*0.0001-0.1
    b08 = data.sel({band_dim: "nir"}).astype(int)*0.0001-0.1
    b11 = data.sel({band_dim: "swir16"}).astype(int)*0.0001-0.1
    b12 = data.sel({band_dim: "swir22"}).astype(int)*0.0001-0.1

    albedo = 0.356*b02 + 0.130*b04 + 0.373*b08 + 0.085*b11 + 0.072*b12 - 0.0018

    albedo = albedo.assign_coords(**{"index": "albedo"})
    albedo = albedo.expand_dims(dim="index")

    return albedo

def evi(item):
    band_dim = "bands"
    crs = get_crs(item)
    if not isinstance(item, list):
        item = [item]
    data = odc_stac.load(item,
        crs=crs,
        bands=["blue", "red", "nir"],
        chunks={'time': -1, 'x': 1024, 'y': 1024},
        resolution=(10)).to_array(dim=band_dim)
    b02 = data.sel({band_dim: "blue"}).astype(int)*0.0001-0.1
    b04 = data.sel({band_dim: "red"}).astype(int)*0.0001-0.1
    b08 = data.sel({band_dim: "nir"}).astype(int)*0.0001-0.1

    evi = 2.5*(b08 - b04)/((b08 + 6*b04-7.5*b02) + 1)

    evi = evi.assign_coords(**{"index": "evi"})
    evi = evi.expand_dims(dim="index")

    return evi

def exg(item):
    band_dim = "bands"
    crs = get_crs(item)
    if not isinstance(item, list):
        item = [item]
    data = odc_stac.load(item,
        crs=crs,
        bands=["blue", "green", "red"],
        chunks={'time': -1, 'x': 1024, 'y': 1024},
        resolution=(10)).to_array(dim=band_dim)
    b02 = data.sel({band_dim: "blue"}).astype(int)*0.0001-0.1
    b03 = data.sel({band_dim: "green"}).astype(int)*0.0001-0.1
    b04 = data.sel({band_dim: "red"}).astype(int)*0.0001-0.1

    ExG = (2 * b03) - (b04 + b02)

    ExG = ExG.assign_coords(**{"index": "exg"})
    ExG = ExG.expand_dims(dim="index")

    return ExG