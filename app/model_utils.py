import logging
import zipfile

import numpy as np
import requests
import torch
from arccnet.visualisation import utils as ut_v

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_and_extract_model(
    model_url,
    model_name,
    *,
    model_data_path,
    extracted_weights_filename="model-data/comet-torch-model.pth",
):
    """
    Downloads and extracts a model archive.

    Parameters
    ----------
    model_url : str
        URL to download the model from
    model_name : str
        Name of the model for cache naming
    extracted_weights_filename : str
        Expected filename of the weights in the archive
    model_data_path : Path
        Path to the model data directory

    Returns
    -------
    Path
        Path to the extracted weights file
    """
    model_data = model_data_path
    model_data.mkdir(parents=True, exist_ok=True)

    model_cache_dir = model_data / model_name
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path = model_cache_dir / f"{model_name}_archive.zip"

    # Download if needed
    if not archive_path.exists():
        logger.info(f"Downloading model archive to {archive_path}...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Download complete: {archive_path}")

    # Extract if needed
    extracted_path = model_cache_dir / extracted_weights_filename
    if not extracted_path.exists() and archive_path.exists():
        logger.info(f"Extracting weights from {archive_path}...")
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            files = zip_ref.namelist()
            logger.info(f"Files in archive: {files}")

            # Find weights file
            target_file = None
            if extracted_weights_filename in files:
                target_file = extracted_weights_filename
            else:
                # Look for any .pth, .pt, or .bin file
                for name in files:
                    if name.endswith((".pth", ".pt", ".bin")):
                        target_file = name
                        break

            if not target_file:
                raise FileNotFoundError("No weights file found in archive")

            logger.info(f"Extracting {target_file} to {model_cache_dir}")
            zip_ref.extract(target_file, model_cache_dir)

            # If extracted file has different name, create expected path
            actual_extracted_path = model_cache_dir / target_file
            if actual_extracted_path != extracted_path:
                actual_extracted_path.replace(extracted_path)

    if not extracted_path.exists():
        raise FileNotFoundError(f"Weights file not found at {extracted_path}")

    return extracted_path


def load_state_dict(weights_path, device="cpu"):
    """
    Load and clean state dict from weights file.

    Parameters
    ----------
    weights_path : Path
        Path to the weights file
    device : str
        Device to load weights to

    Returns
    -------
    dict
        Cleaned state dict
    """
    logger.info(f"Loading model weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)

    # Handle potential nesting in state_dict
    if "model_state_dict" in state_dict:
        logger.info("Found 'model_state_dict' key, extracting...")
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        logger.info("Found 'state_dict' key, extracting...")
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        logger.info("Found 'model' key, extracting...")
        state_dict = state_dict["model"]
    else:
        logger.info("Using state_dict as-is (no nesting detected)")

    return state_dict


def preprocess_data(image_data, hardtanh=True, target_height=224, target_width=224):
    """
    Preprocess the input data for the model.

    Parameters
    ----------
    image_data : numpy.ndarray
        The input data to preprocess.
    hardtanh : bool, optional
        Whether to apply hardtanh transformation, by default True.
    target_height : int, optional
        The target height for resizing, by default 224.
    target_width : int, optional
        The target width for resizing, by default 224.

    Returns
    -------
    torch.Tensor
        The preprocessed data as a PyTorch tensor with shape (1, 1, target_height, target_width).
    """
    image_data = np.nan_to_num(image_data, nan=0.0)
    if hardtanh:
        image_data = ut_v.hardtanh_transform_npy(
            image_data, divisor=800, min_val=-1.0, max_val=1.0
        )
    image_data = ut_v.pad_resize_normalize(
        image_data, target_height=target_height, target_width=target_width
    )
    image_data = image_data.astype(np.float32)
    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
    image_data = np.expand_dims(image_data, axis=0)  # Add channel dimension
    return torch.from_numpy(image_data)


def safe_inference(model, input_tensor, device="cpu"):
    """
    Run inference with error handling.

    Parameters
    ----------
    model : torch.nn.Module
        The model to run inference with
    input_tensor : torch.Tensor
        Input tensor for the model
    device : str
        Device to run inference on

    Returns
    -------
    torch.Tensor or tuple
        Model output
    """
    try:
        model.eval()
        model.to(device)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
        return output
    except Exception:
        logger.exception("An error occurred during model inference.")
        raise
