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

    weights_path = download_and_extract_model(model_url, "yolo_detection")

    try:
        # Look for best.pt file specifically for YOLO models
        if weights_path.name != "best.pt":
            # If the extracted file isn't best.pt, look for it in the same directory
            best_pt_path = weights_path.parent / "best.pt"
            if best_pt_path.exists():
                weights_path = best_pt_path
            else:
                logger.warning(
                    f"Expected YOLO weights file 'best.pt' not found. Using: {weights_path}"
                )

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
        # Convert pixel to world coordinates
        world_coord = mag_map.pixel_to_world(
            pixel_coords[0] * u.pix, pixel_coords[1] * u.pix
        )

        # Transform to heliographic Stonyhurst
        hgs_coord = world_coord.transform_to("heliographic_stonyhurst")

        return {
            "latitude": float(hgs_coord.lat.to(u.deg).value),
            "longitude": float(hgs_coord.lon.to(u.deg).value),
        }
    except Exception as e:
        logger.warning(f"Failed to convert pixel coordinates {pixel_coords}: {e}")
        return {"latitude": np.nan, "longitude": np.nan}


def preprocess_magnetogram(mag_map, target_size=640):
    """
    Preprocess magnetogram for YOLO inference.

    Parameters
    ----------
    mag_map : sunpy.map.Map
        The magnetogram map
    target_size : int
        Target size for YOLO input (default: 640)

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


def yolo_detection(mag_map, model=None, confidence_threshold=0.1):
    """
    Run YOLO detection on a full-disk magnetogram to find active regions.

    Parameters
    ----------
    mag_map : sunpy.map.Map
        The full-disk magnetogram
    model : YOLO, optional
        Pre-loaded YOLO model
    confidence_threshold : float
        Confidence threshold for detections

    Returns
    -------
    list
        List of detection dictionaries with bounding boxes and confidence scores
    """
    if model is None:
        model = download_yolo_model()

    try:
        processed_data = preprocess_magnetogram(mag_map)

        results = model(processed_data, conf=confidence_threshold, verbose=False)

        detections = []

        for result in results:
            if result.boxes is not None:
                boxes = (
                    result.boxes.xyxy.cpu().numpy()
                )  # Get bounding boxes in xyxy format
                confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores

                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = box

                    # Convert pixel coordinates to HGS coordinates
                    bottom_left_hgs = pixel_to_hgs_coords(mag_map, (x1, y1))
                    top_right_hgs = pixel_to_hgs_coords(mag_map, (x2, y2))

                    detection = {
                        "bbox_pixels": {
                            "bottom_left": {"x": float(x1), "y": float(y1)},
                            "top_right": {"x": float(x2), "y": float(y2)},
                        },
                        "bbox_hgs": {
                            "bottom_left": bottom_left_hgs,
                            "top_right": top_right_hgs,
                        },
                        "confidence": float(conf),
                    }

                    detections.append(detection)

        logger.info(f"YOLO detection complete. Found {len(detections)} detections")
        return detections

    except Exception as e:
        logger.error(f"Error during YOLO detection: {e}")
        raise
