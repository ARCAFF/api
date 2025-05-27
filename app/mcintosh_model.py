import logging
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn

from arccnet.models import train_utils as ut_t
from arccnet.models.mcintosh import HierarchicalResNet

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "resnet18"
num_classes_Z = 5  # A, B, C, H, LG
num_classes_P = 3  # 0, x, frag
num_classes_C = 4  # asym, r, sym, x

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_model():
    """
    Downloads, extracts, and loads the Hale classification model.

    Returns:
        torch.nn.Module: The loaded PyTorch model in evaluation mode.
    """
    # Default Model
    model_url = "https://www.comet.com/api/registry/model/item/download?modelItemId=ZkTcrrYWpJwlQ3Kmlp6GCJGiK"
    CACHE_DIR = Path(".cache")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = CACHE_DIR / f"{model_name}_mcintosh_archive.zip"
    extracted_weights_filename = "model-data/comet-torch-model.pth"
    current_weights_path = CACHE_DIR / extracted_weights_filename

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
    if not current_weights_path.exists() and archive_path.exists():
        logger.info(f"Extracting weights from {archive_path}...")
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            files = zip_ref.namelist()
            logger.info(f"Files in archive: {files}")

            # Find weights file
            target_file = None
            if extracted_weights_filename in files:
                target_file = extracted_weights_filename
            else:
                for name in files:
                    if name.endswith((".pth", ".pt", ".bin")):
                        target_file = name
                        break

            if not target_file:
                raise FileNotFoundError("No weights file found in archive")

            logger.info(f"Extracting {target_file}")
            zip_ref.extract(target_file, CACHE_DIR)
            current_weights_path = CACHE_DIR / target_file

    # Load the model
    if not current_weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at {current_weights_path}")

    logger.info(f"Loading model weights from: {current_weights_path}")

    # Load state dict first to examine architecture
    state_dict = torch.load(current_weights_path, map_location=device)

    # Handle potential nesting in state_dict
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    # Load Model
    try:
        model = HierarchicalResNet(
            num_classes_Z=num_classes_Z,
            num_classes_P=num_classes_P,
            num_classes_C=num_classes_C,
            resnet_version=model_name,
        ).to(device)
        ut_t.replace_activations(model, nn.ReLU, nn.LeakyReLU, negative_slope=0.01)

        model.load_state_dict(state_dict)
        model.to(device)

        logger.info(f"Model loaded successfully with architecture: {model_name}")

        model.eval()
        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
