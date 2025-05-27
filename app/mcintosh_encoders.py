from sklearn.preprocessing import LabelEncoder

z_classes = ["A", "B", "C", "H", "LG"]
p_classes = ["asym", "r", "sym", "x"]
c_classes = ["frag", "o", "x"]

z_component_mapping = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "LG",  # Merge D, E, F into LG (LargeGroup)
    "E": "LG",
    "F": "LG",
    "H": "H",
}

p_component_mapping = {
    "x": "x",
    "r": "r",
    "s": "sym",  # Merge s and h into sym
    "h": "sym",
    "a": "asym",  # Merge a and k into asym
    "k": "asym",
}

c_component_mapping = {
    "x": "x",
    "o": "o",
    "i": "frag",  # Merge i and c into frag
    "c": "frag",
}


def create_encoders():
    """
    Create encoders from McIntosh component mappings.
    """
    # Create and fit encoders
    z_encoder = LabelEncoder()
    p_encoder = LabelEncoder()
    c_encoder = LabelEncoder()

    z_encoder.fit(z_classes)
    p_encoder.fit(p_classes)
    c_encoder.fit(c_classes)

    # Create encoders dictionary
    encoders = {"Z_encoder": z_encoder, "p_encoder": p_encoder, "c_encoder": c_encoder}

    # Create mappings dictionary
    mappings = {
        "Z_component": z_component_mapping,
        "p_component": p_component_mapping,
        "c_component": c_component_mapping,
    }

    return encoders, mappings


def decode_predicted_classes_to_original(z_class, p_class, c_class):
    """
    Convert predicted class names directly to original McIntosh classification with alternatives.

    Args:
        z_class: Predicted Z component class name (e.g., "LG")
        p_class: Predicted P component class name (e.g., "asym")
        c_class: Predicted C component class name (e.g., "frag")

    Returns:
        String representation showing original alternatives
    """
    # Reverse mappings to show original alternatives
    z_reverse_mapping = {"A": "A", "B": "B", "C": "C", "H": "H", "LG": "D/E/F"}
    p_reverse_mapping = {"asym": "a/k", "r": "r", "sym": "s/h", "x": "x"}
    c_reverse_mapping = {"frag": "i/c", "o": "o", "x": "x"}

    # Convert to original form representations
    z_original = z_reverse_mapping.get(z_class, z_class)
    p_original = p_reverse_mapping.get(p_class, p_class)
    c_original = c_reverse_mapping.get(c_class, c_class)

    return f"{z_original}-{p_original}-{c_original}"


def decode_mcintosh_classification(predicted_classes_list, encoders=None):
    """
    Decode McIntosh classifications using predicted class names.

    Args:
        predicted_classes_list: List of tuples [(z_class, p_class, c_class), ...]
        encoders: Not needed for this function, kept for compatibility

    Returns:
        List of decoded classification strings
    """
    results = []
    for z_class, p_class, c_class in predicted_classes_list:
        decoded = decode_predicted_classes_to_original(z_class, p_class, c_class)
        results.append(decoded)
    return results
