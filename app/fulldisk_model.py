import logging
from pathlib import Path

import astropy.units as u
import numpy as np
import torch
from astropy.coordinates import SkyCoord
from ultralytics import YOLO

from app.model_utils import (
    download_and_extract_model,
    logger,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def download_yolo_model():
    """
    Downloads, extracts, and loads the YOLO detection model for full-disk magnetogram AR detection.
    """
    model_url = "https://www.comet.com/api/registry/model/item/download?modelItemId=9iPvriGnFaFjE6dNzGYYYZoEG"

    # Specify the correct filename for YOLO models
    weights_path = download_and_extract_model(model_url, "yolo_detection", "best.pt")

    try:
        # Load YOLO model with downloaded weights
        model = YOLO(weights_path)
        model.to(device)

        logger.info(f"YOLO model loaded successfully on device: {device}")
        return model

    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        raise


def pixel_to_hgs_coords(mag_map, pixel_coords):
    """
    Convert pixel coordinates to heliographic Stonyhurst coordinates.

    Parameters
    ----------
    mag_map : sunpy.map.Map
        The magnetogram map
    pixel_coords : tuple
        Pixel coordinates (x, y)

    Returns
    -------
    dict
        Dictionary with latitude and longitude in degrees
    """
    try:
        from sunpy.coordinates import SphericalScreen

        # Get world coordinates
        world_coord = mag_map.pixel_to_world(
            pixel_coords[0] * u.pix, pixel_coords[1] * u.pix
        )

        # Define the center of the solar disk as the observer's position
        observer = mag_map.observer_coordinate

        # Check if the coordinate is on the solar disk
        solar_radius = mag_map.rsun_obs
        distance_from_center = np.sqrt(world_coord.Tx**2 + world_coord.Ty**2)

        if distance_from_center > solar_radius:
            logger.warning(
                f"Coordinates {pixel_coords} are off-disk, using SphericalScreen assumption"
            )

        # Use SphericalScreen context manager with the observer as center
        with SphericalScreen(center=observer):
            # Transform to heliographic Stonyhurst
            hgs_coord = world_coord.transform_to("heliographic_stonyhurst")

            # Get latitude and longitude values
            lat = float(hgs_coord.lat.to(u.deg).value)
            lon = float(hgs_coord.lon.to(u.deg).value)

            # Ensure coordinates are within valid ranges expected by ARDetection model
            # Clamp latitude between -90 and 90
            lat = max(-90.0, min(90.0, lat))

            # Clamp longitude between -90 and 90
            lon = max(-90.0, min(90.0, lon))

            return {
                "latitude": lat,
                "longitude": lon,
            }
    except Exception as e:
        logger.warning(f"Failed to convert pixel coordinates {pixel_coords}: {e}")
        # Return valid default values instead of NaN to avoid validation errors
        return {"latitude": 0.0, "longitude": 0.0}


def preprocess_magnetogram(mag_map, target_size=1024):
    """
    Preprocess magnetogram for YOLO inference.

    Parameters
    ----------
    mag_map : sunpy.map.Map
        The magnetogram map
    target_size : int
        Target size for YOLO input (default: 1024)

    Returns
    -------
    numpy.ndarray
        Preprocessed magnetogram data
    """
    try:
        # Get the data
        data = mag_map.data

        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)

        # Normalize to 0-255 range for YOLO
        data_min, data_max = np.percentile(data, [1, 99])
        data = np.clip((data - data_min) / (data_max - data_min) * 255, 0, 255).astype(
            np.uint8
        )

        # Convert to RGB (duplicate channels for YOLO)
        if len(data.shape) == 2:
            data = np.stack([data, data, data], axis=-1)

        return data

    except Exception as e:
        logger.error(f"Error preprocessing magnetogram: {e}")
        raise


def yolo_detection(mag_map, model=None, confidence_threshold=0.6):
    """
    Run YOLO detection on a full-disk magnetogram to find active regions.
    """
    if model is None:
        model = download_yolo_model()

    try:
        processed_data = preprocess_magnetogram(mag_map)
        results = model(processed_data, conf=confidence_threshold, verbose=False)

        # Log YOLO inference results
        logger.info(
            f"YOLO inference completed. Number of result objects: {len(results)}"
        )

        detections = []
        on_disk_count = 0
        off_disk_count = 0

        for i, result in enumerate(results):
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                logger.info(f"Result {i}: Found {len(boxes)} boxes")

                for j, (box, conf) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = box

                    # Check if detection center is on the solar disk
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Convert center to world coordinates to check if on disk
                    try:
                        world_coord = mag_map.pixel_to_world(
                            center_x * u.pix, center_y * u.pix
                        )
                        solar_radius = mag_map.rsun_obs
                        distance_from_center = np.sqrt(
                            world_coord.Tx**2 + world_coord.Ty**2
                        )

                        if distance_from_center > solar_radius:
                            off_disk_count += 1
                            logger.debug(
                                f"Skipping off-disk detection at ({center_x:.1f}, {center_y:.1f})"
                            )
                            continue  # Skip off-disk detections

                        on_disk_count += 1

                    except Exception as e:
                        logger.warning(
                            f"Could not check disk position for detection at ({center_x:.1f}, {center_y:.1f}): {e}"
                        )
                        continue

                    # Convert pixel coordinates to HGS coordinates
                    bottom_left_hgs = pixel_to_hgs_coords(mag_map, (x1, y1))
                    top_right_hgs = pixel_to_hgs_coords(mag_map, (x2, y2))
                    center_hgs = pixel_to_hgs_coords(mag_map, (center_x, center_y))

                    hale_class = ""
                    mcintosh_class = ""

                    detection = {
                        "time": mag_map.date.datetime,
                        "bbox": {
                            "bottom_left": {
                                "latitude": bottom_left_hgs["latitude"],
                                "longitude": bottom_left_hgs["longitude"],
                            },
                            "top_right": {
                                "latitude": top_right_hgs["latitude"],
                                "longitude": top_right_hgs["longitude"],
                            },
                        },
                        "hale_class": hale_class,
                        "mcintosh_class": mcintosh_class,
                        "confidence": float(conf),
                        "bbox_pixels": {
                            "bottom_left": {"x": float(x1), "y": float(y1)},
                            "top_right": {"x": float(x2), "y": float(y2)},
                        },
                    }

                    detections.append(detection)

        logger.info(
            f"YOLO detection complete. On-disk: {on_disk_count}, Off-disk (filtered): {off_disk_count}, Final detections: {len(detections)}"
        )
        return detections

    except Exception as e:
        logger.error(f"Error during YOLO detection: {e}")
        raise
