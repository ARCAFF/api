from pathlib import Path

import astropy.units as u
import numpy as np
import torch
from arccnet.visualisation import utils as ut_v
from PIL import Image
from ultralytics import YOLO

from app.config import settings
from app.model_utils import download_and_extract_model, logger

device = "cuda" if torch.cuda.is_available() else "cpu"


def download_yolo_model():
    """
    Downloads, extracts, and loads the YOLO detection model for full-disk magnetogram AR detection.
    """
    model_url = "https://www.comet.com/api/registry/model/item/download?modelItemId=9iPvriGnFaFjE6dNzGYYYZoEG"

    weights_path = download_and_extract_model(
        model_url,
        "yolo_detection",
        extracted_weights_filename="best.pt",
        model_data_path=settings.model_path,
    )

    try:
        # Load YOLO model with downloaded weights
        model = YOLO(weights_path)
        model.to(device)

        logger.info(f"YOLO model loaded successfully on device: {device}")
        return model

    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        raise


def preprocess_magnetogram(mag_map, target_size=1024):
    """
    Preprocess a full-disk magnetogram for YOLO-based active region detection.

    This function performs several preprocessing steps on the provided magnetogram:
      1. Converts any invalid data (NaN) to zero.
      2. Applies a hardtanh transformation to constrain data values.
      3. Normalizes the data to the range [0, 255] and converts it to 8-bit unsigned integers.
      4. Resizes the image to the target dimensions for YOLO input.
      5. Vertically flips the resized image for correct orientation.
      6. Saves the processed image to a local cache directory for reuse.

    Parameters
    ----------
    mag_map : sunpy.map.Map
        The magnetogram map to preprocess.
    target_size : int, optional
        The width and height (in pixels) of the output image. Default is 1024.

    Returns
    -------
    numpy.ndarray
        The preprocessed magnetogram data array.
    """
    try:
        data = mag_map.data

        data = np.nan_to_num(data, nan=0.0)
        data = ut_v.hardtanh_transform_npy(data)

        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data = (data * 255).astype(np.uint8)

        savedir = Path(".cache")
        savedir.mkdir(parents=True, exist_ok=True)
        save_path = savedir / "prepr_mag.png"

        img = Image.fromarray(data)
        img_resized = img.resize((target_size, target_size))
        img_flip = img_resized.transpose(Image.FLIP_TOP_BOTTOM)
        img_flip.save(save_path)

        return save_path

    except Exception as e:
        logger.error(f"Error preprocessing magnetogram: {e}")
        raise


def yolo_detection(
    mag_map,
    model=None,
    confidence_threshold=0.5,
):
    """
    Perform YOLO detection on a full-disk magnetogram.

    This function uses a YOLO model to detect active regions in the provided magnetogram.
    The processing steps include:
      1. Preprocessing the magnetogram with `preprocess_magnetogram` to match the YOLO input size.
      2. Running the YOLO model to obtain detection results.
      3. Extracting bounding box coordinates, confidence scores, and class labels.
      4. Scaling the bounding box coordinates from the resized image back to the original image dimensions.
      5. Converting the pixel coordinates into heliographic coordinates using a World Coordinate System (WCS) transformation.

    Parameters
    ----------
    mag_map : sunpy.map.Map
        The full-disk magnetogram map on which to perform region detection.
    model : YOLO, optional
        A preloaded YOLO model. If None, the model is automatically downloaded and loaded.
    confidence_threshold : float, optional
        Minimum confidence threshold for detections. (Default is 0.6)

    Returns
    -------
    list
        A list of detection dictionaries. Each dictionary contains:
            - time: the observation time of the magnetogram.
            - bbox: a dictionary with 'bottom_left' and 'top_right' heliographic coordinates (latitude and longitude).
            - hale_class: the detected active region's class label.
            - confidence: the detection confidence score.
    """
    if model is None:
        model = download_yolo_model()

    img_path = preprocess_magnetogram(mag_map.data)
    results = model.predict(img_path, conf=confidence_threshold)
    bboxes = results[0].boxes
    boxes = bboxes.xyxy.cpu().numpy()
    confidences = bboxes.conf.cpu().numpy()
    class_names = [results[0].names.get(int(cls)) for cls in bboxes.cls]

    original_dims = mag_map.dimensions
    original_width = original_dims.x.value
    original_height = original_dims.y.value

    resized_width = 1024
    resized_height = 1024

    scale_x = original_width / resized_width
    scale_y = original_height / resized_height

    detections = []
    for j, (box, conf) in enumerate(zip(boxes, confidences)):
        x1_resized, y1_resized, x2_resized, y2_resized = box

        x1_original = x1_resized * scale_x
        x2_original = x2_resized * scale_x

        # Invert the Y-coordinate from top-left to bottom-left origin
        y1_original = original_height - (y2_resized * scale_y)
        y2_original = original_height - (y1_resized * scale_y)

        # Perform WCS Conversion
        bottom_left_world = mag_map.pixel_to_world(
            x1_original * u.pix, y1_original * u.pix
        )
        top_right_world = mag_map.pixel_to_world(
            x2_original * u.pix, y2_original * u.pix
        )

        # Transform from the native system (Helioprojective) to Heliographic
        bottom_left_hgs_coord = bottom_left_world.transform_to(
            "heliographic_stonyhurst"
        )
        top_right_hgs_coord = top_right_world.transform_to("heliographic_stonyhurst")

        bottom_left_hgs = {
            "latitude": bottom_left_hgs_coord.lat.to_value(u.deg),
            "longitude": bottom_left_hgs_coord.lon.to_value(u.deg),
        }
        top_right_hgs = {
            "latitude": top_right_hgs_coord.lat.to_value(u.deg),
            "longitude": top_right_hgs_coord.lon.to_value(u.deg),
        }

        hale_class = class_names[j]

        detection = {
            "time": mag_map.date.datetime,
            "bbox": {
                "bottom_left": bottom_left_hgs,
                "top_right": top_right_hgs,
            },
            "hale_class": hale_class,
            "confidence": float(conf),
        }
        detections.append(detection)
    return detections
