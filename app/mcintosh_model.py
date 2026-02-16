import torch
import torch.nn as nn
from arccnet.models import train_utils as ut_t
from arccnet.models.cutouts.mcintosh.models import HierarchicalResNet

from app.config import settings
from app.mcintosh_encoders import (
    c_classes,
    create_encoders,
    p_classes,
    z_classes,
)
from app.model_utils import (
    download_and_extract_model,
    load_state_dict,
    logger,
    preprocess_data,
    safe_inference,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "resnet18"
num_classes_Z = 5  # A, B, C, H, LG
num_classes_P = 4  # asym, r, sym, x
num_classes_C = 3  # o, x, frag


def download_model():
    """
    Downloads, extracts, and loads the McIntosh classification model.
    """
    model_url = "https://www.comet.com/api/registry/model/item/download?modelItemId=ZkTcrrYWpJwlQ3Kmlp6GCJGiK"

    weights_path = download_and_extract_model(
        model_url,
        f"{model_name}_mcintosh",
        model_data_path=settings.models_path,
    )
    state_dict = load_state_dict(weights_path, device)

    # Load and create model
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


def run_inference(model, magnetogram_tensor, device, encoders=None):
    """Run inference on a single magnetogram"""
    # Add batch dimension if needed
    if magnetogram_tensor.dim() == 3:
        magnetogram_tensor = magnetogram_tensor.unsqueeze(0)

    outputs_z, outputs_p, outputs_c = safe_inference(model, magnetogram_tensor, device)

    # Predictions
    _, pred_z = torch.max(outputs_z, 1)
    _, pred_p = torch.max(outputs_p, 1)
    _, pred_c = torch.max(outputs_c, 1)

    # Probabilities
    probs_z = torch.softmax(outputs_z, dim=1)
    probs_p = torch.softmax(outputs_p, dim=1)
    probs_c = torch.softmax(outputs_c, dim=1)

    results = {
        "predictions": {
            "z": pred_z.cpu().numpy()[0],
            "p": pred_p.cpu().numpy()[0],
            "c": pred_c.cpu().numpy()[0],
        },
        "probabilities": {
            "z": probs_z.cpu().numpy()[0],
            "p": probs_p.cpu().numpy()[0],
            "c": probs_c.cpu().numpy()[0],
        },
    }

    # Convert to class labels
    if encoders:
        results["class_labels"] = {
            "z": encoders["Z_encoder"].inverse_transform([pred_z.cpu().numpy()[0]])[0],
            "p": encoders["p_encoder"].inverse_transform([pred_p.cpu().numpy()[0]])[0],
            "c": encoders["c_encoder"].inverse_transform([pred_c.cpu().numpy()[0]])[0],
        }
        # Combine into McIntosh class
        mcintosh_class = (
            results["class_labels"]["z"]
            + results["class_labels"]["p"]
            + results["class_labels"]["c"]
        )
        results["mcintosh_class"] = mcintosh_class

    return results


def mcintosh_classification(cutout, model=None, device="cpu"):
    """
    Classify the input cutout with McIntosh classification scheme.
    """
    if model is None:
        model = download_model()

    encoders, _ = create_encoders()

    cutout_tensor = preprocess_data(cutout)
    result = run_inference(model, cutout_tensor, device, encoders)

    z_probs = ", ".join(
        f"{cls}: {float(p):.4f}"
        for cls, p in zip(z_classes, result["probabilities"]["z"])
    )
    p_probs = ", ".join(
        f"{cls}: {float(p):.4f}"
        for cls, p in zip(p_classes, result["probabilities"]["p"])
    )
    c_probs = ", ".join(
        f"{cls}: {float(p):.4f}"
        for cls, p in zip(c_classes, result["probabilities"]["c"])
    )

    # Format the final result
    result = {
        "mcintosh_class": result.get("mcintosh_class", ""),
        "components": {
            "z": {
                "predicted_class": result["class_labels"]["z"],
                "probabilities": z_probs,
            },
            "p": {
                "predicted_class": result["class_labels"]["p"],
                "probabilities": p_probs,
            },
            "c": {
                "predicted_class": result["class_labels"]["c"],
                "probabilities": c_probs,
            },
        },
    }

    return result
