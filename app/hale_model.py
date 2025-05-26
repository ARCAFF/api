import logging
import zipfile
from pathlib import Path

import numpy as np
import requests
import timm
import torch
from arccnet.models import train_utils as ut_t
from arccnet.visualisation import utils as ut_v

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model():
    """
    Downloads, extracts, and loads the Hale classification model.
    
    Returns:
        torch.nn.Module: The loaded PyTorch model in evaluation mode.
    """
    # Default Model
    model_name = "vit_small_patch16_224"  
    num_classes = 5  # qs, ia, a, b, bg
    model_url = 'https://www.comet.com/api/registry/model/item/download?modelItemId=2Y3HZMoq3XXffVgzkL9wE9IZb'
    CACHE_DIR = Path(".cache")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = CACHE_DIR / f"{model_name}_archive.zip"
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
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            logger.info(f"Files in archive: {files}")
            
            # Find weights file
            target_file = None
            if extracted_weights_filename in files:
                target_file = extracted_weights_filename
            else:
                for name in files:
                    if name.endswith(('.pth', '.pt', '.bin')):
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
    state_dict = torch.load(current_weights_path, map_location='cpu')
    
    # Handle potential nesting in state_dict
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    # Load Model    
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, in_chans=1)
        ut_t.replace_activations(model, torch.nn.ReLU, torch.nn.LeakyReLU, negative_slope=0.01)
        
        model.load_state_dict(state_dict, strict=False)
        
        logger.info(f"Model loaded successfully with architecture: {model_name}")
        model.eval()
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

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
        image_data = ut_v.hardtanh_transform_npy(image_data, divisor=800, min_val=-1.0, max_val=1.0)
    image_data = ut_v.pad_resize_normalize(image_data, target_height=target_height, target_width=target_width)
    image_data = image_data.astype(np.float32) 
    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
    image_data = np.expand_dims(image_data, axis=0)  # Add channel dimension
    return torch.from_numpy(image_data)

def run_inference(model, cutout, device = 'cpu'):
    try:
        model.eval()
        with torch.no_grad():
            output = model(cutout)
        return output.cpu().numpy()
    except Exception:
        logger.exception("An error occurred during model inference.")
        raise

def hale_classification(cutout, model=None, device='cpu'):
    """
    Classify the input cutout with Hale classification scheme.

    Parameters
    ----------
    cutout : numpy.ndarray
        The input cutout to classify.
    model : torch.nn.Module, optional
        The pre-trained model to use for classification, by default None.
    device : str, optional
        The device to run the model on ('cpu' or 'cuda'), by default 'cpu'.

    Returns
    -------
    numpy.ndarray
        The classification result.
    """
    if model is None:
        model = download_model()
    
    cutout = preprocess_data(cutout)
    cutout = cutout.to(device)
    
    result = run_inference(model, cutout, device)
    probabilities = torch.softmax(torch.tensor(result), dim=1).numpy()
    predicted_class = np.argmax(result)
    hale_classes = ['QS', 'IA', 'Alpha', 'Beta', 'Beta-Gamma']
    hale_probs = ", ".join(
        f"{cls}: {float(p):.4f}" for cls, p in zip(hale_classes, probabilities[0])
    )
    result = {
        'predicted_class': hale_classes[predicted_class],
        'probabilities': hale_probs,
        'predicted_class_index': predicted_class
    }
    return result