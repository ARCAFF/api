from datetime import datetime, timedelta
from pathlib import Path

import astropy.units as u
import numpy as np
import torch
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.map import Map
from sunpy.net import Fido
from sunpy.net import attrs as a

from app.fulldisk_model import yolo_detection
from app.hale_model import hale_classification
from app.mcintosh_encoders import decode_predicted_classes_to_original
from app.mcintosh_model import mcintosh_classification

CUTOUT = [800, 400] * u.pix


def download_magnetogram(time):
    r"""
    Download magnetogram and prepare for use

    Parameters
    ----------
    time : datetime.datetime

    """
    query = Fido.search(
        a.Time(time, time + timedelta(minutes=1)),
        (a.Instrument.hmi & a.Physobs("LOS_magnetic_field") & a.Provider("JSOC"))
        | (a.Instrument.mdi & a.Physobs("LOS_magnetic_field") & a.Provider("SHA")),
    )
    if not query:
        raise MagNotFoundError()
    files = Fido.fetch(query["vso"][0])
    if not files:
        raise MagDownloadError()
    mag_map = Map(files[0])
    mag_map = mag_map.rotate()
    return mag_map


def classify(time: datetime, hgs_latitude: float, hgs_longitude: float):
    r"""
    Create and classify an AR cutout from time and position.

    Parameters
    ----------
    time
        Date and time to use for classification.
    hgs_latitude
        Cutout centre latitude.
    hgs_longitude
        Cutout centre longitude.
    Returns
    -------
    Classification result

    """
    mag_map = download_magnetogram(time)
    if "hmi" in mag_map.detector.casefold():
        size = CUTOUT
    else:
        size = CUTOUT / 4

    pos_hgs = SkyCoord(
        hgs_longitude * u.deg,
        hgs_latitude * u.deg,
        obstime=mag_map.date,
        frame="heliographic_stonyhurst",
    )
    pos_hpc = pos_hgs.transform_to(mag_map.coordinate_frame)
    pos_pixels = mag_map.world_to_pixel(pos_hpc)

    top_right = [
        (pos_pixels[0] + (size[0] - 1 * u.pix) / 2).to_value(u.pix),
        (pos_pixels[1] + (size[1] - 1 * u.pix) / 2).to_value(u.pix),
    ]
    bottom_left = [
        (pos_pixels[0] - (size[0] - 1 * u.pix) / 2).to_value(u.pix),
        (pos_pixels[1] - (size[1] - 1 * u.pix) / 2).to_value(u.pix),
    ]
    cutout = mag_map.submap(bottom_left * u.pix, top_right=top_right * u.pix)

    # Get Hale classification
    hale_result = hale_classification(cutout.data)
    hale_class = hale_result["predicted_class"]

    # Get McIntosh classification
    mcintosh_result = None
    mcintosh_components = None
    mcintosh_class = None

    if hale_class == "QS" or hale_class == "IA":
        # For quiet sun or inactive regions, use Hale class
        mcintosh_class = hale_class
    else:
        # For active regions, run McIntosh classification
        mcintosh_result = mcintosh_classification(cutout.data)
        mcintosh_components = mcintosh_result["components"]

        # Get the predicted classes from components
        z_predicted = mcintosh_components["z"]["predicted_class"]
        p_predicted = mcintosh_components["p"]["predicted_class"]
        c_predicted = mcintosh_components["c"]["predicted_class"]

        # Decode to original forms
        mcintosh_class = decode_predicted_classes_to_original(
            z_predicted, p_predicted, c_predicted
        )

    # Ensure mcintosh_class is never None
    if mcintosh_class is None:
        mcintosh_class = "Unknown"

    # Prepare final result
    result = {
        "time": time,
        "hale_class": hale_result["predicted_class"],
        "hale_probs": hale_result["probabilities"],
        "mcintosh_class": mcintosh_class,
        "mcintosh_components": mcintosh_components,
        "hgs_latitude": hgs_latitude,
        "hgs_longitude": hgs_longitude,
    }

    return result


def detect(time: datetime):
    r"""
    Detect and classify all active regions in a full-disk magnetogram.

    Parameters
    ----------
    time : datetime.datetime
        Date and time for detection

    Returns
    -------
    list
        List of detection results with bounding boxes and classifications
    """
    # Download magnetogram
    mag_map = download_magnetogram(time)
    mag_map.data
    # ML model here

    detections = yolo_detection(mag_map)
    return detections


class MagNotFoundError(Exception):
    r"""No SRS file found"""

    pass


class MagDownloadError(Exception):
    r"""Error downloading SRS file"""

    pass
