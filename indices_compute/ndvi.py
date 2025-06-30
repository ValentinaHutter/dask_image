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


def ndvi(item):
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

    ndvi = (b08 - b04)/(b08 + b04)

    ndvi = ndvi.assign_coords(**{"index": "ndvi"})
    ndvi = ndvi.expand_dims(dim="index")

    return ndvi


def ndwi(item):
    band_dim = "bands"
    crs = get_crs(item)
    if not isinstance(item, list):
        item = [item]
    data = odc_stac.load(item,
        crs=crs,
        bands=["green", "nir"],
        chunks={'time': -1, 'x': 1024, 'y': 1024},
        resolution=(10)).to_array(dim=band_dim)
    b03 = data.sel({band_dim: "green"}).astype(int)*0.0001-0.1
    b08 = data.sel({band_dim: "nir"}).astype(int)*0.0001-0.1

    ndwi = (b03 - b08)/(b08 + b03)

    ndwi = ndwi.assign_coords(**{"index": "ndwi"})
    ndwi = ndwi.expand_dims(dim="index")

    return ndwi


def ndmi(item):
    band_dim = "bands"
    crs = get_crs(item)
    if not isinstance(item, list):
        item = [item]
    data = odc_stac.load(item,
        crs=crs,
        bands=["nir", "swir16"],
        chunks={'time': -1, 'x': 1024, 'y': 1024},
        resolution=(10)).to_array(dim=band_dim)
    b08 = data.sel({band_dim: "nir"}).astype(int)*0.0001-0.1
    b11 = data.sel({band_dim: "swir16"}).astype(int)*0.0001-0.1
   
    ndmi = (b08 - b11)/(b08 + b11)

    ndmi = ndmi.assign_coords(**{"index": "ndmi"})
    ndmi = ndmi.expand_dims(dim="index")

    return ndmi


def nbr(item):
    band_dim = "bands"
    crs = get_crs(item)
    if not isinstance(item, list):
        item = [item]
    data = odc_stac.load(item,
        crs=crs,
        bands=["nir", "swir22"],
        chunks={'time': -1, 'x': 1024, 'y': 1024},
        resolution=(10)).to_array(dim=band_dim)
    b08 = data.sel({band_dim: "nir"}).astype(int)*0.0001-0.1
    b12 = data.sel({band_dim: "swir22"}).astype(int)*0.0001-0.1
   
    nbr = (b08 - b12)/(b08 + b12)

    nbr = nbr.assign_coords(**{"index": "nbr"})
    nbr = nbr.expand_dims(dim="index")

    return nbr