import numpy as np
from odc import stac as odc_stac
from pyproj import CRS
import requests
import xml.etree.ElementTree as ET


degToRad = np.pi / 180

bands = ["green", "red", "rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]
names = ["B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
band_names = {name: bands[i] for i, name in enumerate(names)}

bands_10m = ["red", "green", "blue", "nir"]
bands_20m = ["swir22", "rededge2", "rededge3", "rededge1", "swir16", "nir08"]
bands_60m = ["coastal", "nir09"]
bands_none = ["visual", "wvp", "scl", "aot", "cloud", "snow"]


def normalize(unnormalized, min, max):
    return 2 * (unnormalized - min) / (max - min) - 1


def denormalize(normalized, min, max):
    return 0.5 * (normalized + 1) * (max - min) + min


def tansig(input):
    return 2 / (1 + np.exp(-2 * input)) - 1


def load_data(items, bands=bands, chunks={'time': -1, 'x': 512, 'y': 512}, resolution=10):
    if not isinstance(items, list):
        items = [items]
    item = items[0]
    
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
    cube = odc_stac.load(items,
        crs=crs,
        bands=bands,
        chunks=chunks,
        resolution=(resolution))

    return cube


def get_viewing_angles(item):
    # viewing angles can only be read from MTD.xml
    # solar angles are in the stac item properties
    metadata = requests.get(item.assets['granule_metadata'].href).text

    root = ET.fromstring(metadata)
    child = root.find(root.tag.split('}')[0]+'}Geometric_Info')
    in_root = child.find('Tile_Angles').find('Mean_Viewing_Incidence_Angle_List')
    for angle in in_root.iter('Mean_Viewing_Incidence_Angle'):
        if angle.attrib:
            if angle.get('bandId') == '4':
                zenith4 = angle.find('ZENITH_ANGLE')
                azimuth4 = angle.find('AZIMUTH_ANGLE')
                if not (zenith4.get('unit') == 'deg' and azimuth4.get('unit') == 'deg'):
                    print('Warning: angle unit: ', zenith4.get('unit'), azimuth4.get('unit'))
                saa = item.properties["view:sun_azimuth"]
                sza = item.properties["view:sun_elevation"]

                return float(zenith4.text), float(azimuth4.text), sza, saa
            

def load_data_10(item):
    cube = load_data(item, bands=bands_10m, resolution=10)

    return cube

def load_data_20(item):
    cube = load_data(item, bands=bands_20m, resolution=20)

    return cube

def load_data_60(item):
    cube = load_data(item, bands=bands_60m, resolution=60)

    return cube

def load_data_none(item):
    cube = load_data(item, bands=bands_none, resolution=10)

    return cube
