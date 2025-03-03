import numpy as np
from .utils import *


def fapar(item):

    def neuron1(
        b03_norm,
        b04_norm,
        b05_norm,
        b06_norm,
        b07_norm,
        b8a_norm,
        b11_norm,
        b12_norm,
        viewZen_norm,
        sunZen_norm,
        relAzim_norm,
    ):
        s = (
            -0.887068364040280
            + 0.268714454733421 * b03_norm
            - 0.205473108029835 * b04_norm
            + 0.281765694196018 * b05_norm
            + 1.337443412255980 * b06_norm
            + 0.390319212938497 * b07_norm
            - 3.612714342203350 * b8a_norm
            + 0.222530960987244 * b11_norm
            + 0.821790549667255 * b12_norm
            - 0.093664567310731 * viewZen_norm
            + 0.019290146147447 * sunZen_norm
            + 0.037364446377188 * relAzim_norm
        )
        return tansig(s)

    def neuron2(
        b03_norm,
        b04_norm,
        b05_norm,
        b06_norm,
        b07_norm,
        b8a_norm,
        b11_norm,
        b12_norm,
        viewZen_norm,
        sunZen_norm,
        relAzim_norm,
    ):
        s = (
            0.320126471197199
            - 0.248998054599707 * b03_norm
            - 0.571461305473124 * b04_norm
            - 0.369957603466673 * b05_norm
            + 0.246031694650909 * b06_norm
            + 0.332536215252841 * b07_norm
            + 0.438269896208887 * b8a_norm
            + 0.819000551890450 * b11_norm
            - 0.934931499059310 * b12_norm
            + 0.082716247651866 * viewZen_norm
            - 0.286978634108328 * sunZen_norm
            - 0.035890968351662 * relAzim_norm
        )
        return tansig(s)

    def neuron3(
        b03_norm,
        b04_norm,
        b05_norm,
        b06_norm,
        b07_norm,
        b8a_norm,
        b11_norm,
        b12_norm,
        viewZen_norm,
        sunZen_norm,
        relAzim_norm,
    ):
        s = (
            0.610523702500117
            - 0.164063575315880 * b03_norm
            - 0.126303285737763 * b04_norm
            - 0.253670784366822 * b05_norm
            - 0.321162835049381 * b06_norm
            + 0.067082287973580 * b07_norm
            + 2.029832288655260 * b8a_norm
            - 0.023141228827722 * b11_norm
            - 0.553176625657559 * b12_norm
            + 0.059285451897783 * viewZen_norm
            - 0.034334454541432 * sunZen_norm
            - 0.031776704097009 * relAzim_norm
        )
        return tansig(s)

    def neuron4(
        b03_norm,
        b04_norm,
        b05_norm,
        b06_norm,
        b07_norm,
        b8a_norm,
        b11_norm,
        b12_norm,
        viewZen_norm,
        sunZen_norm,
        relAzim_norm,
    ):
        s = (
            -0.379156190833946
            + 0.130240753003835 * b03_norm
            + 0.236781035723321 * b04_norm
            + 0.131811664093253 * b05_norm
            - 0.250181799267664 * b06_norm
            - 0.011364149953286 * b07_norm
            - 1.857573214633520 * b8a_norm
            - 0.146860751013916 * b11_norm
            + 0.528008831372352 * b12_norm
            - 0.046230769098303 * viewZen_norm
            - 0.034509608392235 * sunZen_norm
            + 0.031884395036004 * relAzim_norm
        )
        return tansig(s)

    def neuron5(
        b03_norm,
        b04_norm,
        b05_norm,
        b06_norm,
        b07_norm,
        b8a_norm,
        b11_norm,
        b12_norm,
        viewZen_norm,
        sunZen_norm,
        relAzim_norm,
    ):
        s = (
            +1.353023396690570
            - 0.029929946166941 * b03_norm
            + 0.795804414040809 * b04_norm
            + 0.348025317624568 * b05_norm
            + 0.943567007518504 * b06_norm
            - 0.276341670431501 * b07_norm
            - 2.946594180142590 * b8a_norm
            + 0.289483073507500 * b11_norm
            + 1.044006950440180 * b12_norm
            - 0.000413031960419 * viewZen_norm
            + 0.403331114840215 * sunZen_norm
            + 0.068427130526696 * relAzim_norm
        )
        return tansig(s)

    def layer2(neuron1, neuron2, neuron3, neuron4, neuron5):
        s = (
            -0.336431283973339
            + 2.126038811064490 * neuron1
            - 0.632044932794919 * neuron2
            + 5.598995787206250 * neuron3
            + 1.770444140578970 * neuron4
            - 0.267879583604849 * neuron5
        )
        return s

    band_dim = "bands"
    data = load_data(item).to_array(dim=band_dim)

    b03 = data.sel({band_dim: band_names["B03"]})
    b04 = data.sel({band_dim: band_names["B04"]})
    b05 = data.sel({band_dim: band_names["B05"]})
    b06 = data.sel({band_dim: band_names["B06"]})
    b07 = data.sel({band_dim: band_names["B07"]})
    b8a = data.sel({band_dim: band_names["B8A"]})
    b11 = data.sel({band_dim: band_names["B11"]})
    b12 = data.sel({band_dim: band_names["B12"]})

    viewZen, viewAzim, sunZen, sunAzim = get_viewing_angles(item)

    b03_norm = normalize(b03, 0, 0.253061520471542)
    b04_norm = normalize(b04, 0, 0.290393577911328)
    b05_norm = normalize(b05, 0, 0.305398915248555)
    b06_norm = normalize(b06, 0.006637972542253, 0.608900395797889)
    b07_norm = normalize(b07, 0.013972727018939, 0.753827384322927)
    b8a_norm = normalize(b8a, 0.026690138082061, 0.782011770669178)
    b11_norm = normalize(b11, 0.016388074192258, 0.493761397883092)
    b12_norm = normalize(b12, 0, 0.493025984460231)
    viewZen_norm = normalize(np.cos(viewZen * degToRad), 0.918595400582046, 1)
    sunZen_norm = normalize(
        np.cos(sunZen * degToRad), 0.342022871159208, 0.936206429175402
    )
    relAzim_norm = np.cos((sunAzim - viewAzim) * degToRad)

    n1 = neuron1(
        b03_norm,
        b04_norm,
        b05_norm,
        b06_norm,
        b07_norm,
        b8a_norm,
        b11_norm,
        b12_norm,
        viewZen_norm,
        sunZen_norm,
        relAzim_norm,
    )
    n2 = neuron2(
        b03_norm,
        b04_norm,
        b05_norm,
        b06_norm,
        b07_norm,
        b8a_norm,
        b11_norm,
        b12_norm,
        viewZen_norm,
        sunZen_norm,
        relAzim_norm,
    )
    n3 = neuron3(
        b03_norm,
        b04_norm,
        b05_norm,
        b06_norm,
        b07_norm,
        b8a_norm,
        b11_norm,
        b12_norm,
        viewZen_norm,
        sunZen_norm,
        relAzim_norm,
    )
    n4 = neuron4(
        b03_norm,
        b04_norm,
        b05_norm,
        b06_norm,
        b07_norm,
        b8a_norm,
        b11_norm,
        b12_norm,
        viewZen_norm,
        sunZen_norm,
        relAzim_norm,
    )
    n5 = neuron5(
        b03_norm,
        b04_norm,
        b05_norm,
        b06_norm,
        b07_norm,
        b8a_norm,
        b11_norm,
        b12_norm,
        viewZen_norm,
        sunZen_norm,
        relAzim_norm,
    )

    l2 = layer2(n1, n2, n3, n4, n5)

    f = denormalize(l2, 0.000153013463222, 0.977135096979553)

    f.attrs = data.attrs

    f = f.assign_coords(**{"index": "fapar"})
    f = f.expand_dims(dim="index")

    return f
