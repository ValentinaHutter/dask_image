from odc import stac as odc_stac
from pyproj import CRS


def ndvi(item):
    band_dim = "bands"
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
    data = odc_stac.load([item],
        crs=crs,
        bands=["red", "nir"],
        chunks={'time': -1, 'x': 1024, 'y': 1024},
        resolution=(10)).to_array(dim=band_dim)
    b04 = data.sel({band_dim: "red"})
    b08 = data.sel({band_dim: "nir"})

    ndvi = (b08 - b04)/(b08 + b04)

    ndvi = ndvi.assign_coords(**{"index": "ndvi"})
    ndvi = ndvi.expand_dims(dim="index")

    return ndvi
