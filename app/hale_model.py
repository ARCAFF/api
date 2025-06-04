import numpy as np
import timm
import torch

from app.model_utils import (
    download_and_extract_model,
    load_state_dict,
    logger,
    preprocess_data,
    safe_inference,
)
from arccnet.models import train_utils as ut_t


def download_model():
    """
    Downloads, extracts, and loads the Hale classification model.

    Returns:
        torch.nn.Module: The loaded PyTorch model in evaluation mode.
    """
    model_name = "vit_small_patch16_224"
    num_classes = 5  # qs, ia, a, b, bg
    model_url = "https://www.comet.com/api/registry/model/item/download?modelItemId=2Y3HZMoq3XXffVgzkL9wE9IZb"

    weights_path = download_and_extract_model(model_url, model_name)
    state_dict = load_state_dict(weights_path)

    # Load and create model
    try:
        model = timm.create_model(
            model_name, pretrained=False, num_classes=num_classes, in_chans=1
        )
        ut_t.replace_activations(
            model, torch.nn.ReLU, torch.nn.LeakyReLU, negative_slope=0.01
        )

        model.load_state_dict(state_dict, strict=False)

        logger.info(f"Model loaded successfully with architecture: {model_name}")
        model.eval()
        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def hale_classification(cutout, model=None, device="cpu"):
    """
    Classify the input cutout with Hale classification scheme.
    """
    if model is None:
        model = download_model()

    cutout_tensor = preprocess_data(cutout)

    result = safe_inference(model, cutout_tensor, device)

    result = result.cpu().numpy()
    probabilities = torch.softmax(torch.tensor(result), dim=1).numpy()
    predicted_class = np.argmax(result)
    hale_classes = ["QS", "IA", "Alpha", "Beta", "Beta-Gamma"]
    hale_probs = ", ".join(
        f"{cls}: {float(p):.4f}" for cls, p in zip(hale_classes, probabilities[0])
    )
    result = {
        "predicted_class": hale_classes[predicted_class],
        "probabilities": hale_probs,
        "predicted_class_index": predicted_class,
    }
    return result
